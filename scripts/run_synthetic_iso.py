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
from cosmographic_analysis.hemispheric_comparison import exec_map
from cosmographic_analysis.statistics import fit_gaussian
from cosmographic_analysis.plotting.histograms import plot_histograms


def run_iso(config_path: str, n_workers: int = 1):
    config = load_config(config_path)
    p = config.parameters
    d = config.data

    print(f"Loading data (z range: {p.zdown} < z < {p.zup})...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    pts = hp.nside2npix(p.nside)
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])

    healpix_dirs = get_healpix_vectors(p.nside)

    print(f"Generating {p.repetitions} isotropic realizations...")
    v1_iso = np.zeros((p.repetitions, len(ra), 3))
    for i in range(p.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        v1_iso[i] = vecti

    h0u_all, h0d_all, q0u_all, q0d_all = [], [], [], []
    for i, v1_it in enumerate(v1_iso):
        if i == 0 or (i + 1) % 50 == 0 or i == p.repetitions - 1:
            print(f"  [{i+1}/{p.repetitions}]")
        datos_lcdm = [r1, v1_it, hostyn, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm), n_workers=n_workers)
        h0u_all.append(np.array(res_h0[0]))
        h0d_all.append(np.array(res_h0[1]))
        q0u_all.append(np.array(res_q0[0]))
        q0d_all.append(np.array(res_q0[1]))

    h0m = np.concatenate(h0u_all + h0d_all, axis=1)
    q0m = np.concatenate(q0u_all + q0d_all, axis=1)
    delta_h0_max = np.max(h0m, axis=1) - np.min(h0m, axis=1)
    delta_q0_max = np.max(q0m, axis=1) - np.min(q0m, axis=1)

    x_h0, y_h0 = fit_gaussian(delta_h0_max)
    x_q0, y_q0 = fit_gaussian(delta_q0_max)
    plot_histograms(
        [delta_h0_max, 0, x_h0, y_h0],
        [delta_q0_max, 0, x_q0, y_q0],
        filename=f"{config.output.histograms}[ISO](hf={p.h0f}_qf={p.q0f}).png",
    )
    print("ISO simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run ISO synthetic simulations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers (default: 1 = serial)")
    args = parser.parse_args()
    run_iso(args.config, args.n_workers)


if __name__ == "__main__":
    main()
