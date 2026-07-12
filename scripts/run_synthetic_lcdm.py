#!/usr/bin/env python3
"""
Run LCDM synthetic data simulations.

Replicates the LCDM simulation section of main.ipynb.
Usage: python scripts/run_synthetic_lcdm.py [--config config.yaml]
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
from cosmographic_analysis.coordinates import DecRa2Cartesian, get_healpix_vectors
from cosmographic_analysis.cosmology import mu
from cosmographic_analysis.hemispheric_comparison import exec_map_numba
from cosmographic_analysis.statistics import fit_gaussian
from cosmographic_analysis.plotting.histograms import plot_histograms


def run_lcdm(config_path: str, n_workers: int = 1, optimizer: str = 'woodbury'):
    config = load_config(config_path)
    p = config.parameters
    d = config.data

    print(f"Loading data (z range: {p.zdown} < z < {p.zup})...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    v1 = DecRa2Cartesian(dec, ra)
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])
    pts = hp.nside2npix(p.nside)

    healpix_dirs = get_healpix_vectors(p.nside)

    print(f"Generating {p.repetitions} LCDM realizations...")
    mu_fid = np.array([mu(zi, p.h0f, p.q0f) for zi in zz])
    r1_lcdm = np.tile(r1, (p.repetitions, 1, 1))

    for i in range(p.repetitions):
        mu_sample = np.random.normal(mu_fid, sigmuz)
        r1_lcdm[i, :, 5] = mu_sample

    h0um, h0dm, q0um, q0dm = [], [], [], []
    for i, r1_it in enumerate(r1_lcdm):
        if i == 0 or (i + 1) % 50 == 0 or i == p.repetitions - 1:
            print(f"  [{i+1}/{p.repetitions}]")
        datos_lcdm = [r1_it, v1, hostyn, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy]
        res_h0, res_q0 = exec_map_numba(healpix_dirs, tuple(datos_lcdm), n_workers=n_workers, method=optimizer)
        h0um.append(np.array(res_h0[0]))
        h0dm.append(np.array(res_h0[1]))
        q0um.append(np.array(res_q0[0]))
        q0dm.append(np.array(res_q0[1]))

    h0m = np.concatenate(h0um + h0dm, axis=1)
    q0m = np.concatenate(q0um + q0dm, axis=1)
    delta_h0_max = np.max(h0m, axis=1) - np.min(h0m, axis=1)
    delta_q0_max = np.max(q0m, axis=1) - np.min(q0m, axis=1)

    x_h0, y_h0 = fit_gaussian(delta_h0_max)
    x_q0, y_q0 = fit_gaussian(delta_q0_max)
    plot_histograms(
        [delta_h0_max, 0, x_h0, y_h0],
        [delta_q0_max, 0, x_q0, y_q0],
        titlemarker="LCDM",
        filename=f"{config.output.histograms}[LCDM](hf={p.h0f}_qf={p.q0f}).png",
    )
    print("LCDM simulation complete.")


def main():
    parser = argparse.ArgumentParser(description="Run LCDM synthetic simulations")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers (default: 1 = serial)")
    parser.add_argument("--optimizer", choices=["golden", "scipy", "grid", "woodbury", "woodbury-cholesky"], default="woodbury",
                        help="Optimizer for hemispheric comparison")
    args = parser.parse_args()
    run_lcdm(args.config, args.n_workers, args.optimizer)


if __name__ == "__main__":
    main()
