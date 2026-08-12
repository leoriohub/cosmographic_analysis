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
import time
from typing import Optional
from multiprocessing import Pool

import numpy as np
import healpy as hp

from cosmographic_analysis.hemispheric_comparison import exec_map_numba, exec_map_fixed_numba


# Module-level shared data for ISO parallel workers (fork copy-on-write)
_WORKER_ISO_DATA = None

def _init_iso_worker(data):
    global _WORKER_ISO_DATA
    _WORKER_ISO_DATA = data

def _iso_worker_task(v1_it):
    from cosmographic_analysis.hemispheric_comparison import exec_map_numba as _exec_numba
    r1, hostyn_arr, cov_mat, h0f, q0f, pts, zup, zdown, healpix_dirs, method, cov_numpy, model_code = _WORKER_ISO_DATA
    try:
        datos = [r1, v1_it, hostyn_arr, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model_code]
        res_h0, res_q0 = _exec_numba(healpix_dirs, tuple(datos), n_workers=1, method=method)
        h0u = np.array(res_h0[0])
        h0d = np.array(res_h0[1])
        q0u = np.array(res_q0[0])
        q0d = np.array(res_q0[1])
        h0m = np.concatenate((h0u, h0d))
        q0m = np.concatenate((q0u, q0d))
        return (np.max(h0m) - np.min(h0m), np.max(q0m) - np.min(q0m))
    except Exception:
        return (np.nan, np.nan)


# Module-level shared data for LCDM parallel workers
_WORKER_LCDM_DATA = None

def _init_lcdm_worker(data):
    global _WORKER_LCDM_DATA
    _WORKER_LCDM_DATA = data

def _lcdm_worker_task(r1_it):
    from cosmographic_analysis.hemispheric_comparison import exec_map_fixed_numba as _exec_fixed_numba
    v1, hostyn_arr, cov_mat, h0f, q0f, pts, zup, zdown, healpix_dirs, precomputed, method, cov_np, model_code = _WORKER_LCDM_DATA
    try:
        datos = [r1_it, v1, hostyn_arr, cov_mat, h0f, q0f, pts, zup, zdown, cov_np, model_code]
        res_h0, res_q0 = _exec_fixed_numba(healpix_dirs, tuple(datos), precomputed, n_workers=1, method=method)
        h0u = np.array(res_h0[0])
        h0d = np.array(res_h0[1])
        q0u = np.array(res_q0[0])
        q0d = np.array(res_q0[1])
        h0m = np.concatenate((h0u, h0d))
        q0m = np.concatenate((q0u, q0d))
        return (np.max(h0m) - np.min(h0m), np.max(q0m) - np.min(q0m))
    except Exception:
        return (np.nan, np.nan)


from cosmographic_analysis.config import load_config, ensure_output_dirs
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors, IndexToDecRa
from cosmographic_analysis.hemispheric_comparison import exec_map_numba, precompute_hemisphere_data, exec_map_fixed_numba
from cosmographic_analysis.cosmology import mu_model, MODEL_CODES
from cosmographic_analysis.anisotropy import get_max_anisotropy
from cosmographic_analysis.maps import generate_map, dirs_to_theta_phi
from cosmographic_analysis.dw_statistic import hemispheric_dw, total_dw
from cosmographic_analysis.statistics import fit_gaussian, mc_statistics
from cosmographic_analysis.plotting.histograms import plot_histograms, plot_both_histograms
from cosmographic_analysis.plotting.skymaps import plot_h0_q0_maps
from cosmographic_analysis.plotting.summary import save_summary_tables


def run_pipeline(config_path: str, n_workers: int = 1, optimizer: str = 'woodbury', run_sims: bool = True, n_reps: Optional[int] = None, nside: Optional[int] = None, model: Optional[str] = None):
    config = load_config(config_path)
    if n_reps is not None:
        config.parameters.repetitions = n_reps
    if nside is not None:
        config.parameters.nside = nside
    if model is not None:
        config.parameters.model = model
    if config.parameters.model not in MODEL_CODES:
        raise ValueError(f"Unknown model '{config.parameters.model}'. Valid: {sorted(MODEL_CODES)}")
    model_code = MODEL_CODES[config.parameters.model]
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
    print(f"  n_workers: {n_workers}, optimizer: {optimizer}, model: {p.model}")
    print(f"  simulations: {'yes' if run_sims else 'no (--no-sims)'}")
    print("=" * 60)
    _opt_tag = f"(method={optimizer})"
    _model_tag = f"(model={p.model})"

    # Step 1: Load data
    print("\n[1/9] Loading Pantheon+ data...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    print(f"  {len(zz)} SNe with {p.zdown} < z < {p.zup}")
    print(f"  Cov matrix shape: {cov_mat.shape}")

    pts = hp.nside2npix(p.nside)
    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy, model_code,
    )
    r1, v1, hostyn_arr, _, _, _, _, _, _, _, _ = datos

    # Step 2: Get HEALPix vectors
    print("\n[2/9] Computing HEALPix vectors...")
    npix = hp.nside2npix(p.nside)
    pixel_indices = np.arange(npix)
    healpix_ra, healpix_dec = IndexToDecRa(p.nside, pixel_indices)
    healpix_dirs = get_healpix_vectors(p.nside)

    # Step 3: Hemispheric comparison (single execution, use inner parallelism)
    print(f"\n[3/9] Running hemispheric comparison ({len(healpix_dirs)} directions)...")
    pool = Pool(min(n_workers, 8, os.cpu_count() or 8)) if n_workers > 1 else None
    results_h0, results_q0 = exec_map_numba(healpix_dirs, datos, pool=pool, n_workers=n_workers, method=optimizer)
    if pool is not None:
        pool.close()
        pool.join()
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

    bestfit_data = [np.array(h0u), np.array(h0d), np.array(q0u), np.array(q0d)]
    max_anis_dir = get_max_anisotropy(bestfit_data, healpix_dirs)
    print(f"  Max anisotropy direction (dec, ra) for q0: {max_anis_dir[0]}")

    # Step 5: Generate maps
    print("\n[5/9] Generating sky maps...")
    theta, phi = dirs_to_theta_phi(hdirs)

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
    if run_sims:
        print(f"\n[7/9] ISO simulation ({p.repetitions} repetitions)...")
        v1_iso = np.zeros((p.repetitions, len(ra), 3))
        for i in range(p.repetitions):
            vecti = np.random.randn(len(ra), 3)
            vecti /= np.linalg.norm(vecti, axis=1)[:, np.newaxis]
            v1_iso[i] = vecti

        if n_workers > 1:
            iso_shared = (r1, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, healpix_dirs, optimizer, cov_numpy, model_code)
            n_iso_workers = min(n_workers, p.repetitions)

            if optimizer in ('grid', 'woodbury', 'woodbury-cholesky'):
                from multiprocessing import get_context
                n_iso_workers = min(n_iso_workers, 4)
                pool_class = get_context('spawn').Pool
            else:
                pool_class = Pool

            with pool_class(n_iso_workers, initializer=_init_iso_worker, initargs=(iso_shared,)) as iso_pool:
                print(f"  Running {p.repetitions} ISO iterations with {n_iso_workers} workers "
                      f"({'spawn' if optimizer in ('grid', 'woodbury', 'woodbury-cholesky') else 'fork'} context)...")
                results = list(iso_pool.starmap(_iso_worker_task, [(v1_iso[i],) for i in range(p.repetitions)]))
            for i in range(0, p.repetitions, 50):
                print(f"  ISO [{min(i+50, p.repetitions)}/{p.repetitions}]")
            delta_h0_iso_max = np.array([r[0] for r in results])
            delta_q0_iso_max = np.array([r[1] for r in results])
        else:
            delta_h0_list, delta_q0_list = [], []
            for i, v1_it in enumerate(v1_iso):
                try:
                    if i == 0 or (i + 1) % 50 == 0 or i == p.repetitions - 1:
                        print(f"  ISO [{i+1}/{p.repetitions}]")
                    datos_lcdm = [r1, v1_it, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy, model_code]
                    res_h0, res_q0 = exec_map_numba(healpix_dirs, tuple(datos_lcdm), n_workers=1, method=optimizer)
                    h0u = np.array(res_h0[0])
                    h0d = np.array(res_h0[1])
                    q0u = np.array(res_q0[0])
                    q0d = np.array(res_q0[1])
                    h0m = np.concatenate((h0u, h0d))
                    q0m = np.concatenate((q0u, q0d))
                    delta_h0_list.append(np.max(h0m) - np.min(h0m))
                    delta_q0_list.append(np.max(q0m) - np.min(q0m))
                except Exception as e:
                    print(f"  [WARN] ISO iteration {i+1} failed: {e}")
                    delta_h0_list.append(np.nan)
                    delta_q0_list.append(np.nan)
            delta_h0_iso_max = np.array(delta_h0_list)
            delta_q0_iso_max = np.array(delta_q0_list)

        header_iso = (
            f"This is the data for ISO distribution with new set of data for each repetition\n"
            f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
            f"delta_h0_iso_max  delta_q0_iso_max"
        )
        filename_iso = (
            f"{o.compilations}{p.prefix_name}[ISO]({p.h0f}=h0f_{p.q0f}=q0f)"
            f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}){_opt_tag}{_model_tag}.txt"
        )
        np.savetxt(filename_iso, np.column_stack([delta_h0_iso_max, delta_q0_iso_max]), header=header_iso)

        # Step 8: Synthetic LCDM simulation
        print(f"\n[8/9] LCDM simulation ({p.repetitions} repetitions)...")
        mu_fid = np.array([mu_model(zi, p.h0f, p.q0f, model_code) for zi in zz])
        r1_lcdm = np.tile(r1, (p.repetitions, 1, 1))

        for i in range(p.repetitions):
            mu_sample = np.random.normal(mu_fid, sigmuz)
            r1_lcdm[i, :, 5] = mu_sample

        lcdm_precomputed = precompute_hemisphere_data(healpix_dirs, datos, n_workers=n_workers, cov_numpy=cov_numpy)

        if n_workers > 1:
            lcdm_shared = (v1, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, healpix_dirs, lcdm_precomputed, optimizer, cov_numpy, model_code)
            n_lcdm_workers = min(n_workers, p.repetitions)
            with Pool(n_lcdm_workers, initializer=_init_lcdm_worker, initargs=(lcdm_shared,)) as lcdm_pool:
                print(f"  Running {p.repetitions} iterations with {n_lcdm_workers} workers...")
                results = list(lcdm_pool.starmap(_lcdm_worker_task, [(r1_lcdm[i],) for i in range(p.repetitions)]))
            for i in range(0, p.repetitions, 50):
                print(f"  LCDM [{min(i+50, p.repetitions)}/{p.repetitions}]")
            delta_h0_lcdm_max = np.array([r[0] for r in results])
            delta_q0_lcdm_max = np.array([r[1] for r in results])
        else:
            delta_h0_list, delta_q0_list = [], []
            for i, r1_it in enumerate(r1_lcdm):
                try:
                    if i == 0 or (i + 1) % 50 == 0 or i == p.repetitions - 1:
                        print(f"  LCDM [{i+1}/{p.repetitions}]")
                    datos_lcdm = [r1_it, v1, hostyn_arr, cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy, model_code]
                    res_h0, res_q0 = exec_map_fixed_numba(healpix_dirs, tuple(datos_lcdm), lcdm_precomputed, n_workers=1, method=optimizer)
                    h0u = np.array(res_h0[0])
                    h0d = np.array(res_h0[1])
                    q0u = np.array(res_q0[0])
                    q0d = np.array(res_q0[1])
                    h0m = np.concatenate((h0u, h0d))
                    q0m = np.concatenate((q0u, q0d))
                    delta_h0_list.append(np.max(h0m) - np.min(h0m))
                    delta_q0_list.append(np.max(q0m) - np.min(q0m))
                except Exception as e:
                    print(f"  [WARN] LCDM iteration {i+1} failed: {e}")
                    delta_h0_list.append(np.nan)
                    delta_q0_list.append(np.nan)
            delta_h0_lcdm_max = np.array(delta_h0_list)
            delta_q0_lcdm_max = np.array(delta_q0_list)

        header_lcdm = (
            f"This is the data for LCDM distribution with new set of data for each repetition\n"
            f"{p.repetitions} repetitions, {pts} points q0f= {p.q0f} h0f= {p.h0f}\n\n"
            f"delta_h0_lcdm_max  delta_q0_lcdm_max"
        )
        filename_lcdm = (
            f"{o.compilations}{p.prefix_name}[LCDM]({p.h0f}=h0f_{p.q0f}=q0f)"
            f"({p.repetitions}_rep)({pts}_pts)({p.zup}>z>{p.zdown}){_opt_tag}{_model_tag}.txt"
        )
        np.savetxt(
            filename_lcdm,
            np.column_stack([delta_h0_lcdm_max, delta_q0_lcdm_max]),
            header=header_lcdm,
        )

        # Step 9: Fit Gaussians + plot histograms
        print("\n[9/9] Fitting Gaussians and plotting histograms...")

        delta_h0_iso_max_clean = delta_h0_iso_max[~np.isnan(delta_h0_iso_max)]
        delta_q0_iso_max_clean = delta_q0_iso_max[~np.isnan(delta_q0_iso_max)]
        delta_h0_lcdm_max_clean = delta_h0_lcdm_max[~np.isnan(delta_h0_lcdm_max)]
        delta_q0_lcdm_max_clean = delta_q0_lcdm_max[~np.isnan(delta_q0_lcdm_max)]

        x_gauss_h0_iso, y_gauss_h0_iso = fit_gaussian(delta_h0_iso_max_clean)
        x_gauss_q0_iso, y_gauss_q0_iso = fit_gaussian(delta_q0_iso_max_clean)

        data_h0_iso_hist = [delta_h0_iso_max, delta_h0_data_max, x_gauss_h0_iso, y_gauss_h0_iso]
        data_q0_iso_hist = [delta_q0_iso_max, delta_q0_data_max, x_gauss_q0_iso, y_gauss_q0_iso]
        file_name_iso = f"{o.histograms}[ISO](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}){_opt_tag}{_model_tag}.png"
        plot_histograms(data_h0_iso_hist, data_q0_iso_hist, filename=file_name_iso)

        x_gauss_h0_lcdm, y_gauss_h0_lcdm = fit_gaussian(delta_h0_lcdm_max_clean)
        x_gauss_q0_lcdm, y_gauss_q0_lcdm = fit_gaussian(delta_q0_lcdm_max_clean)

        data_h0_lcdm_hist = [delta_h0_lcdm_max, delta_h0_data_max, x_gauss_h0_lcdm, y_gauss_h0_lcdm]
        data_q0_lcdm_hist = [delta_q0_lcdm_max, delta_q0_data_max, x_gauss_q0_lcdm, y_gauss_q0_lcdm]
        file_name_lcdm = f"{o.histograms}[LCDM](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}){_opt_tag}{_model_tag}.png"
        plot_histograms(data_h0_lcdm_hist, data_q0_lcdm_hist, titlemarker="LCDM", filename=file_name_lcdm)

        data_h0_both = [delta_h0_iso_max, delta_h0_lcdm_max, delta_h0_data_max]
        data_q0_both = [delta_q0_iso_max, delta_q0_lcdm_max, delta_q0_data_max]
        file_name_both = f"{o.histograms}[BOTH](hf={p.h0f}_qf={p.q0f})({p.repetitions})_rep_({p.zup}>z>{p.zdown}){_opt_tag}{_model_tag}.png"
        plot_both_histograms(data_h0_both, data_q0_both, filename=file_name_both)

        maximum_anisotropy_data = np.array([delta_h0_data_max, delta_q0_data_max])
        maximum_anisotropy_mc = np.array([
            delta_h0_lcdm_max, delta_q0_lcdm_max,
            delta_h0_iso_max, delta_q0_iso_max,
        ])
        p_values = mc_statistics(maximum_anisotropy_data, maximum_anisotropy_mc)
        p_h0_iso_max, p_q0_iso_max, p_h0_lcdm_max, p_q0_lcdm_max = p_values
    else:
        p_h0_iso_max = p_q0_iso_max = p_h0_lcdm_max = p_q0_lcdm_max = np.nan

    # Step 10: Sky maps
    print("\n[10/11] Generating sky maps...")
    map_path = plot_h0_q0_maps(p.nside, theta, phi, h0, q0, p.h0f, p.q0f, config, optimizer=optimizer, model=p.model)
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
        optimizer=optimizer,
        model=p.model,
    )
    print(f"  Tables saved to {o.tables}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


VALID_OPTIMIZERS = frozenset({'golden', 'scipy', 'grid', 'woodbury', 'woodbury-cholesky'})


def run_comparison(config_path, n_workers, optimizer_list):
    """Run multiple optimizers and print comparison table."""
    if not optimizer_list:
        optimizer_list = list(VALID_OPTIMIZERS)
    else:
        unknown = set(optimizer_list) - VALID_OPTIMIZERS
        if unknown:
            print(f"  [Error] Unknown optimizer(s): {sorted(unknown)}")
            print(f"  Valid optimizers: {sorted(VALID_OPTIMIZERS)}")
            return

    config = load_config(config_path)
    ensure_output_dirs(config)
    p = config.parameters
    d = config.data
    if p.model not in MODEL_CODES:
        raise ValueError(f"Unknown model '{p.model}'. Valid: {sorted(MODEL_CODES)}")
    model_code = MODEL_CODES[p.model]

    print("=" * 60)
    print("Multi-Method Comparison")
    print("=" * 60)
    print(f"  nside={p.nside}, h0f={p.h0f}, q0f={p.q0f}")
    print(f"  redshift range: {p.zdown} < z < {p.zup}")
    print(f"  optimizers: {optimizer_list}, model: {p.model}")
    print("=" * 60)

    # Load data once
    print("\nLoading Pantheon+ data...")
    zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z, cov_numpy = \
        load_pantheon_data(d.lcparam, d.cov_matrix, p.zup, p.zdown)
    print(f"  {len(zz)} SNe with {p.zdown} < z < {p.zup}")

    pts = hp.nside2npix(p.nside)
    datos = build_datos_tuple(
        ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn,
        cov_mat, p.h0f, p.q0f, pts, p.zup, p.zdown, cov_numpy, model_code,
    )

    healpix_dirs = get_healpix_vectors(p.nside)

    # Comparison mode forces serial execution so GPU methods (grid, woodbury,
    # woodbury-cholesky) use the GPU path. Parallel execution would silently
    # fall back to golden section via multi_hem_map_numba, making the
    # comparison meaningless.
    if n_workers > 1:
        print(f"  [Note] Forcing serial execution (n_workers=1) to enable GPU method comparison.")
    results = {}
    for opt in optimizer_list:
        print(f"\n  Running optimizer '{opt}'...")
        t0 = time.perf_counter()
        r_h0, r_q0 = exec_map_numba(healpix_dirs, datos, pool=None, n_workers=1, method=opt)
        elapsed = time.perf_counter() - t0

        h0u, h0d = np.array(r_h0[0]), np.array(r_h0[1])
        q0u, q0d = np.array(r_q0[0]), np.array(r_q0[1])
        max_dh0 = np.max(np.abs(h0u - h0d))
        max_dq0 = np.max(np.abs(q0u - q0d))
        results[opt] = {'time': elapsed, 'max_dh0': max_dh0, 'max_dq0': max_dq0}
        print(f"    done in {elapsed*1000:.1f}ms | max|Δh₀|={max_dh0:.4f} | max|Δq₀|={max_dq0:.4f}")

    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)
    print(f"{'Optimizer':20s} {'Time (ms)':>10s} {'max|Δh₀|':>10s} {'max|Δq₀|':>10s}")
    print("-" * 54)
    for opt in optimizer_list:
        r = results[opt]
        print(f"{opt:20s} {r['time']*1000:>10.1f} {r['max_dh0']:>10.4f} {r['max_dq0']:>10.4f}")
    print("=" * 54)


def main():
    parser = argparse.ArgumentParser(description="Cosmographic analysis pipeline")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers (default: 1 = serial)")
    parser.add_argument("--optimizer", choices=["golden", "scipy", "grid", "woodbury", "woodbury-cholesky"], default="woodbury",
                        help="Optimizer (default: woodbury — fastest on GPU):"
                             " 'golden' (numba golden section, CPU), 'grid' (GPU grid search), "
                             "'woodbury' (CPU Woodbury + Woodbury χ² q₀), "
                             "'woodbury-cholesky' (CPU Woodbury + Cholesky χ² q₀), "
                             "or 'scipy' (generic nD)")
    parser.add_argument("--nside", type=int, default=None,
                        help="HEALPix Nside parameter (default: from config.yaml, usually 8)")
    parser.add_argument("--model", choices=["taylor2", "pade11", "pade21"], default=None,
                        help="Distance model (default: from config.yaml)")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="Override number of MC simulation repetitions (default: from config.yaml, usually 500)")
    parser.add_argument("--no-sims", action="store_true",
                        help="Skip MC simulations (ISO + LCDM). Only runs real data + maps + tables.")
    parser.add_argument("--compare", nargs="*", default=None,
                        help="Run multiple optimizers and print comparison table. "
                             "If no arguments, runs all available. "
                             "Example: --compare grid woodbury woodbury-cholesky")
    args = parser.parse_args()
    if args.repetitions is not None:
        override_reps = args.repetitions
    else:
        override_reps = None
    override_nside = args.nside
    if args.compare is not None:
        run_comparison(args.config, args.n_workers, args.compare)
    else:
        run_pipeline(args.config, args.n_workers, args.optimizer, run_sims=not args.no_sims, n_reps=override_reps, nside=override_nside, model=args.model)


if __name__ == "__main__":
    main()
