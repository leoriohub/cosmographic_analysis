#!/usr/bin/env python3
"""
Full cosmographic analysis pipeline.

Reproduces the entire main.ipynb pipeline as a CLI script.
Usage: python scripts/run_pipeline.py [--config config.yaml]
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import healpy as hp
from IPython.display import clear_output
except ImportError:
    def clear_output(wait=False):
        pass

from cosmographic_analysis.config import load_config, ensure_output_dirs
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors, IndexToDecRa
from cosmographic_analysis.hemispheric_comparison import exec_map, precompute_hemisphere_data, exec_map_fixed
from cosmographic_analysis.anisotropy import get_max_anisotropy
from cosmographic_analysis.maps import generate_map
from cosmographic_analysis.dw_statistic import hemispheric_dw, total_dw
from cosmographic_analysis.statistics import fit_gaussian, mc_statistics
from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms
from cosmographic_analysis.plotting.skymaps import plot_h0_q0_maps
from cosmographic_analysis.plotting.summary import save_summary_tables


def run_pipeline(config_path: str):
    config = load_config(config_path)
    ensure_output_dirs(config)
    p = config.parameters
    d = config.data
    o = config.output

    print("=" * 60)
    print("Cosmographic Analysis Pipeline")
    print("=" * 60)
    print(f"  nside={p.nside}, h0f={p.h0f}, q0f={p.q0f}")
    print(f"  redshift range: {p.zdown} < z < {p.zup}")
    print(f"  repetitions: {p.repetitions}")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/9] Loading Pantheon+ data...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    print(f"  {len(zz)} SNe with {p.zdown} < z < {p.zup}")
    print(f"  Cov matrix shape: {cov_mat.shape}")

    pts = hp.nside2npix(p.nside)
    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown,
    )
    r1, v1, hostyn_arr, _, _, _, _, _, _ = datos

    # Step 2: Get HEALPix vectors
    print("\n[2/9] Computing HEALPix vectors...")
    npix = hp.nside2npix(p.nside)
    pixel_indices = np.arange(npix)
    healpix_ra, healpix_dec = IndexToDecRa(p.nside, pixel_indices)
    healpix_dirs = get_healpix_vectors(p.nside)

    # Create reusable multiprocessing pool for all iterations
    pool = Pool(min(os.cpu_count() or 10, 10))

    # Step 3: Hemispheric comparison
    print(f"\n[3/9] Running hemispheric comparison ({len(healpix_dirs)} directions)...")
    results_h0, results_q0 = exec_map(healpix_dirs, datos, pool=pool)
    h0u, h0d, h0u_err, h0d_err = results_h0
    q0u, q0d, q0u_err, q0d_err = results_q0

    h0 = np.concatenate((h0u, h0d))
    h0_err = np.concatenate((h0u_err, h0d_err))
    q0 = np.concatenate((q0u, q0d))
    q0_err = np.concatenate((q0u_err, q0d_err))

    # Step 4: Max anisotropy
    print("\n[4/9] Computing maximum anisotropy...")
    hdirs = np.concatenate((np.array(healpix_dirs), -np.array(healpix_dirs)))

    h0_max_err = h0_err[np.argmax(h0)]
    h0_min_err = h0_err[np.argmin(h0)]
    q0_max_err = q0_err[np.argmax(q0)]
    q0_min_err = q0_err[np.argmin(q0)]

    delta_h0_data_err = np.sqrt(h0_max_err**2 + h0_min_err**2)
    delta_q0_data_err = np.sqrt(q0_max_err**2 + q0_min_err**2)

    delta_h0_data = np.abs(np.array(h0u) - np.array(h0d))
    delta_q0_data = np.abs(np.array(q0u) - np.array(q0d))
    delta_h0_data_max = np.max(delta_h0_data)
    delta_q0_data_max = np.max(delta_q0_data)

    print(f"  delta_h0 = {delta_h0_data_max:.6f} +/- {delta_h0_data_err:.6f}")
    print(f"  delta_q0 = {delta_q0_data_max:.6f} +/- {delta_q0_data_err:.6f}")

    bestfit_data = [h0u, h0d, q0u, q0d]
    max_anis_dir = get_max_anisotropy(bestfit_data, healpix_dirs)
    print(f"  Max anisotropy direction (dec, ra) for q0: {max_anis_dir[0]}")

    # Step 5: Generate maps
    print("\n[5/9] Generating sky maps...")
    theta = np.arccos(hdirs[:, 2])
    phi = np.radians(180) - np.arctan2(hdirs[:, 1], hdirs[:, 0])

    h0map, q0map = generate_map(p.nside, theta, phi, h0, q0)
    print(f"  Maps generated: h0map ({h0map.size} pixels), q0map ({q0map.size} pixels)")

    # Step 6: Durbin-Watson statistic
    print("\n[6/9] Computing Durbin-Watson statistics...")
    max_q0_anis_vec = healpix_dirs[np.argmax(np.abs(delta_q0_data))]
    dw_up, dw_down = hemispheric_dw(max_q0_anis_vec, datos)
    print(f"  DW for max anisotropy direction: North={dw_up:.4f}, South={dw_down:.4f}")

    total_dw_up, total_dw_down, data_dw = total_dw(healpix_dirs, datos)
    print(f"  Mean DW all directions: North={np.mean(total_dw_up):.4f}, South={np.mean(total_dw_down):.4f}")

    # Step 7: Synthetic ISO simulation
    print(f"\n[7/9] ISO simulation ({p.repetitions} repetitions)...")
    v1_iso = np.zeros((p.repetitions, len(ra), 3))
    for i in range(p.repetitions):
        vecti = np.random.randn(len(ra), 3)
        vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
        thetai = np.arccos(vecti[:, 2])
        phii = np.arctan2(vecti[:, 1], vecti[:, 0])
        deci = thetai
        rai = np.pi - phii
        v1i = np.column_stack([
            np.sin(rai)*np.cos(deci),
            np.sin(rai)*np.sin(deci),
            np.cos(rai),
        ])
        v1_iso[i] = v1i

    h0u_iso, h0d_iso, q0u_iso, q0d_iso = [], [], [], []
    for i, v1_it in enumerate(v1_iso):
        clear_output(wait=True)
        print(f"ISO iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1, v1_it, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map(healpix_dirs, tuple(datos_lcdm), pool=pool)
        h0u_iso.append(np.array(res_h0[0]))
        h0d_iso.append(np.array(res_h0[1]))
        q0u_iso.append(np.array(res_q0[0]))
        q0d_iso.append(np.array(res_q0[1]))

    h0u_iso = np.array(h0u_iso)
    h0d_iso = np.array(h0d_iso)
    q0u_iso = np.array(q0u_iso)
    q0d_iso = np.array(q0d_iso)

    h0m_iso = np.concatenate((h0u_iso, h0d_iso), axis=1)
    q0m_iso = np.concatenate((q0u_iso, q0d_iso), axis=1)
    delta_h0_iso_max = np.max(h0m_iso, axis=1) - np.min(h0m_iso, axis=1)
    delta_q0_iso_max = np.max(q0m_iso, axis=1) - np.min(q0m_iso, axis=1)

    header_iso = (
        f"This is the data for ISO distribution with new set of data for each repetition\n"
        f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
        f"delta_h0_iso_max  delta_q0_iso_max"
    )
    filename_iso = (
        f"{o.compilations}{p.prefix_name}[ISO]({p.h0f}=h0f_{p.q0f}=q0f)"
        f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}).txt"
    )
    np.savetxt(filename_iso, np.column_stack([delta_h0_iso_max, delta_q0_iso_max]), header=header_iso)

    # Step 8: Synthetic LCDM simulation
    print(f"\n[8/9] LCDM simulation ({p.repetitions} repetitions)...")
    from cosmographic_analysis.cosmology import mu

    mu_fid = np.array([mu(zi, p.h0f, p.q0f) for zi in zz])
    r1_lcdm = np.tile(r1, (p.repetitions, 1, 1))

    for i in range(p.repetitions):
        mu_sample = np.random.normal(mu_fid, sigmuz)
        r1_lcdm[i, :, 5] = mu_sample

    # Precompute hemisphere data once for all LCDM iterations
    print("  Precomputing hemisphere data for LCDM...")
    lcdm_precomputed = precompute_hemisphere_data(healpix_dirs, datos)

    h0um_lcdm, h0dm_lcdm, q0um_lcdm, q0dm_lcdm = [], [], [], []
    for i, r1_it in enumerate(r1_lcdm):
        clear_output(wait=True)
        print(f"LCDM iteration {i+1}/{p.repetitions}")
        datos_lcdm = [r1_it, v1, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown]
        res_h0, res_q0 = exec_map_fixed(healpix_dirs, tuple(datos_lcdm), lcdm_precomputed)
        h0um_lcdm.append(np.array(res_h0[0]))
        h0dm_lcdm.append(np.array(res_h0[1]))
        q0um_lcdm.append(np.array(res_q0[0]))
        q0dm_lcdm.append(np.array(res_q0[1]))

    h0um_lcdm = np.array(h0um_lcdm)
    h0dm_lcdm = np.array(h0dm_lcdm)
    q0um_lcdm = np.array(q0um_lcdm)
    q0dm_lcdm = np.array(q0dm_lcdm)

    h0m_lcdm = np.concatenate((h0um_lcdm, h0dm_lcdm), axis=1)
    q0m_lcdm = np.concatenate((q0um_lcdm, q0dm_lcdm), axis=1)
    delta_h0_lcdm_max = np.max(h0m_lcdm, axis=1) - np.min(h0m_lcdm, axis=1)
    delta_q0_lcdm_max = np.max(q0m_lcdm, axis=1) - np.min(q0m_lcdm, axis=1)

    header_lcdm = (
        f"This is the data for LCDM distribution with new set of data for each repetition\n"
        f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
        f"delta_h0_lcdm_max  delta_q0_lcdm_max"
    )
    filename_lcdm = (
        f"{o.compilations}{p.prefix_name}[LCDM]({p.h0f}=h0f_{p.q0f}=q0f)"
        f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}).txt"
    )
    np.savetxt(
        filename_lcdm,
        np.column_stack([delta_h0_lcdm_max, delta_q0_lcdm_max]),
        header=header_lcdm,
    )

    # Step 9: Fit Gaussians + plot histograms
    print("\n[9/9] Fitting Gaussians and plotting histograms...")

    x_gauss_h0_iso, y_gauss_h0_iso = fit_gaussian(delta_h0_iso_max)
    x_gauss_q0_iso, y_gauss_q0_iso = fit_gaussian(delta_q0_iso_max)

    data_h0_iso_hist = [delta_h0_iso_max, delta_h0_data_max, x_gauss_h0_iso, y_gauss_h0_iso]
    data_q0_iso_hist = [delta_q0_iso_max, delta_q0_data_max, x_gauss_q0_iso, y_gauss_q0_iso]
    file_name_iso = f"{o.histograms}[ISO](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    plot_histograms(data_h0_iso_hist, data_q0_iso_hist, filename=file_name_iso)

    x_gauss_h0_lcdm, y_gauss_h0_lcdm = fit_gaussian(delta_h0_lcdm_max)
    x_gauss_q0_lcdm, y_gauss_q0_lcdm = fit_gaussian(delta_q0_lcdm_max)

    data_h0_lcdm_hist = [delta_h0_lcdm_max, delta_h0_data_max, x_gauss_h0_lcdm, y_gauss_h0_lcdm]
    data_q0_lcdm_hist = [delta_q0_lcdm_max, delta_q0_data_max, x_gauss_q0_lcdm, y_gauss_q0_lcdm]
    file_name_lcdm = f"{o.histograms}[LCDM](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    plot_histograms(data_h0_lcdm_hist, data_q0_lcdm_hist, titlemarker="LCDM", filename=file_name_lcdm)

    data_h0_both = [delta_h0_iso_max, delta_h0_lcdm_max, delta_h0_data_max]
    data_q0_both = [delta_q0_iso_max, delta_q0_lcdm_max, delta_q0_data_max]
    file_name_both = f"{o.histograms}[BOTH](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}).png"
    plot_both_histograms(data_h0_both, data_q0_both, filename=file_name_both)

    maximum_anisotropy_data = np.array([delta_h0_data_max, delta_q0_data_max])
    maximum_anisotropy_mc = np.array([
        delta_h0_lcdm_max, delta_q0_lcdm_max,
        delta_h0_iso_max, delta_q0_iso_max,
    ])
    p_values = mc_statistics(maximum_anisotropy_data, maximum_anisotropy_mc)
    p_h0_iso_max, p_q0_iso_max, p_h0_lcdm_max, p_q0_lcdm_max = p_values

    # Step 10: Sky maps
    print("\n[10/11] Generating sky maps...")
    map_path = plot_h0_q0_maps(p.nside, theta, phi, h0, q0, p.h0f, p.q0f, config)
    print(f"  Map saved: {map_path}")

    # Step 11: Summary tables
    print("\n[11/11] Saving summary tables...")
    save_summary_tables(
        delta_h0_data_max=delta_h0_data_max,
        delta_q0_data_max=delta_q0_data_max,
        p_h0_iso_max=p_h0_iso_max,
        p_q0_iso_max=p_q0_iso_max,
        p_h0_lcdm_max=p_h0_lcdm_max,
        p_q0_lcdm_max=p_q0_lcdm_max,
        prefix_name=p.prefix_name,
        h0f=p.h0f,
        q0f=p.q0f,
        zup=p.zup,
        zdown=p.zdown,
        pts=pts,
        n_rep=p.repetitions,
        tables_dir=o.tables,
    )
    print(f"  Tables saved to {o.tables}")

    # Clean up multiprocessing pool
    pool.close()
    pool.join()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Cosmographic analysis pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
