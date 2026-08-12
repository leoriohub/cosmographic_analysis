#!/usr/bin/env python3
"""
Run isotropic synthetic data simulations.

Replicates the ISO simulation section of main.ipynb.
Usage: python scripts/run_synthetic_iso.py [--config config.yaml]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

import numpy as np
import healpy as hp

from cosmographic_analysis.config import load_config
from cosmographic_analysis.data_loader import load_pantheon_data
from cosmographic_analysis.coordinates import get_healpix_vectors
from cosmographic_analysis.cosmology import MODEL_CODES
from cosmographic_analysis.hemispheric_comparison import exec_map_numba
from cosmographic_analysis.statistics import fit_gaussian
from cosmographic_analysis.plotting.histograms import plot_histograms


def run_iso(config_path: str, n_workers: int = 1, optimizer: str = 'woodbury'):
    config = load_config(config_path)
    p = config.parameters
    d = config.data
    model_code = MODEL_CODES[p.model]

    print(f"Loading data (z range: {p.zdown} < z < {p.zup})...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    pts = hp.nside2npix(p.nside)
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])

    healpix_dirs = get_healpix_vectors(p.nside)

    print(f"Generating {p.repetitions} isotropic realizations (optimizer={optimizer})...")
    v1_iso = np.zeros((p.repetitions, len(ra), 3))
    for i in range(p.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        v1_iso[i] = vecti

    # Parallel setup: exec_map_numba parallelizes internally via a pool
    # (spawn context for GPU methods, fork otherwise — mirrors run_pipeline)
    if n_workers > 1:
        from multiprocessing import get_context
        n_workers = min(n_workers, 8, os.cpu_count() or 8)
        if optimizer in ('grid', 'woodbury', 'woodbury-cholesky'):
            pool = get_context('spawn').Pool(n_workers)
        else:
            from multiprocessing import Pool
            pool = Pool(n_workers)
        print(f"  Using pool of {n_workers} workers")
    else:
        pool = None

    h0u_all, h0d_all, q0u_all, q0d_all = [], [], [], []
    for i, v1_it in enumerate(v1_iso):
        if i == 0 or (i + 1) % 50 == 0 or i == p.repetitions - 1:
            print(f"  [{i+1}/{p.repetitions}]")
        datos_lcdm = [r1, v1_it, hostyn, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy, model_code]
        res_h0, res_q0 = exec_map_numba(healpix_dirs, tuple(datos_lcdm), pool=pool, n_workers=n_workers, method=optimizer)
        h0u_all.append(np.array(res_h0[0]))
        h0d_all.append(np.array(res_h0[1]))
        q0u_all.append(np.array(res_q0[0]))
        q0d_all.append(np.array(res_q0[1]))

    if pool is not None:
        pool.close()
        pool.join()

    h0m = np.concatenate([np.array(h0u_all), np.array(h0d_all)], axis=1)
    q0m = np.concatenate([np.array(q0u_all), np.array(q0d_all)], axis=1)
    delta_h0_max = np.max(h0m, axis=1) - np.min(h0m, axis=1)
    delta_q0_max = np.max(q0m, axis=1) - np.min(q0m, axis=1)

    x_h0, y_h0 = fit_gaussian(delta_h0_max)
    x_q0, y_q0 = fit_gaussian(delta_q0_max)
    plot_histograms(
        [delta_h0_max, 0, x_h0, y_h0],
        [delta_q0_max, 0, x_q0, y_q0],
        filename=f"{config.output.histograms}[ISO](hf={p.h0f}_qf={p.q0f})(model={p.model}).png",
    )
    print("ISO simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run ISO synthetic simulations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers (default: 1 = serial)")
    parser.add_argument("--optimizer", choices=["golden", "scipy", "grid", "woodbury", "woodbury-cholesky"], default="woodbury",
                        help="Optimizer for hemispheric comparison")
    args = parser.parse_args()
    run_iso(args.config, args.n_workers, args.optimizer)


if __name__ == "__main__":
    main()
