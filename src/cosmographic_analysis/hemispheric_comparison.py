import os
# Prevent BLAS thread oversubscription in multiprocessing workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from multiprocessing import Pool
from typing import Tuple, Optional
from scipy.optimize import minimize

import pandas as pd
import numpy as np
from numba import njit
# import distance modulus from cosmology.py
from cosmographic_analysis.cosmology import mu


# ---- Numba helpers for 1D optimization (golden section search) ----


@njit(cache=True)
def _chi2_1par(theta_val, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0):
    """Chi2 for 1-parameter fit. fit_h0=True fits h0, False fits q0."""
    if fit_h0:
        mu_model = mu(z, theta_val, theta_fixed)
    else:
        mu_model = mu(z, theta_fixed, theta_val)

    resid = np.empty(len(z))
    for i in range(len(z)):
        if hostyn[i] == 1:
            resid[i] = mu_ceph[i] - mu_model[i]
        else:
            resid[i] = mu_sh0es[i] - mu_model[i]

    Ar = np.dot(resid, np.dot(inv_cov, resid))
    return Ar


@njit(cache=True)
def _golden_fit(z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, a, b, tol=1e-6):
    """Golden section search for 1-parameter fit. Returns (optimum, error)."""
    phi = (np.sqrt(5) - 1) / 2

    if a > b:
        a, b = b, a

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _chi2_1par(c, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)
    fd = _chi2_1par(d, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _chi2_1par(c, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _chi2_1par(d, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)

    x_opt = (a + b) / 2.0

    # Error via central difference on chi2 at minimum
    h = max(tol * 10, abs(x_opt) * 1e-4)
    fp = _chi2_1par(x_opt + h, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)
    fm = _chi2_1par(x_opt - h, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)
    f0 = _chi2_1par(x_opt, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0)
    hessian = (fp - 2 * f0 + fm) / (h * h)
    error = np.sqrt(1.0 / hessian) if hessian > 0 else 1.0

    return x_opt, error


# ---- Generic chi2 for n-D optimization (scipy fallback) ----


def _fit_hemisphere_scipy(z, mu_ceph, mu_sh0es, hostyn, inv_cov, h0f, q0f, free_mask):
    """Fit using scipy.minimize. free_mask[0]=fit h0, free_mask[1]=fit q0."""
    n_free = int(free_mask[0]) + int(free_mask[1])
    if n_free == 0:
        return h0f, q0f, 0.0, 0.0

    x0 = []
    bounds = []
    if free_mask[0]:
        x0.append(h0f)
        bounds.append((0.3, 1.5))
    if free_mask[1]:
        x0.append(q0f)
        bounds.append((-1.5, 0.5))
    x0 = np.array(x0)

    def chi2_func(theta):
        idx = 0
        h0 = theta[idx] if free_mask[0] else h0f
        idx = idx + 1 if free_mask[0] else idx
        q0 = theta[idx] if free_mask[1] else q0f
        mu_model = mu(z, h0, q0)
        resid = np.empty(len(z))
        for i in range(len(z)):
            resid[i] = (mu_ceph[i] - mu_model[i]) if hostyn[i] == 1 else (mu_sh0es[i] - mu_model[i])
        return np.dot(resid, np.dot(inv_cov, resid))

    res = minimize(chi2_func, x0, method='L-BFGS-B', bounds=bounds)

    h0_val = res.x[0] if free_mask[0] else h0f
    q0_val = res.x[1 if free_mask[0] else 0] if free_mask[1] else q0f

    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, 'todense'):
            hess_inv = hess_inv.todense()
        errs = np.sqrt(np.abs(np.diag(hess_inv)))
    except Exception:
        errs = np.ones(n_free) * 0.1

    h0_err = errs[0] if free_mask[0] else 0.0
    q0_err = errs[-1] if free_mask[1] else 0.0

    return h0_val, q0_val, h0_err, q0_err


# Module-level shared data for parallel precompute workers
_WORKER_PRECOMPUTE_DIRS: Optional[np.ndarray] = None
_WORKER_PRECOMPUTE_V1: Optional[np.ndarray] = None
_WORKER_PRECOMPUTE_COV: Optional[pd.DataFrame] = None


def _init_precompute_worker(dirs, v1, cov):
    global _WORKER_PRECOMPUTE_DIRS, _WORKER_PRECOMPUTE_V1, _WORKER_PRECOMPUTE_COV
    _WORKER_PRECOMPUTE_DIRS = dirs
    _WORKER_PRECOMPUTE_V1 = v1
    _WORKER_PRECOMPUTE_COV = cov


def _precompute_worker(idx: int) -> tuple:
    healpix_dir = _WORKER_PRECOMPUTE_DIRS[idx]
    dot_products = np.dot(_WORKER_PRECOMPUTE_V1, healpix_dir)
    mask_up = dot_products >= 0
    upi = np.where(mask_up)[0]
    downi = np.where(~mask_up)[0]
    inv_cov_up = np.linalg.inv(_WORKER_PRECOMPUTE_COV.iloc[upi, upi].values)
    inv_cov_down = np.linalg.inv(_WORKER_PRECOMPUTE_COV.iloc[downi, downi].values)
    return idx, {
        "up_indices": upi,
        "down_indices": downi,
        "inv_cov_up": inv_cov_up,
        "inv_cov_down": inv_cov_down,
    }


def precompute_hemisphere_data(
    healpix_dirs: np.ndarray,
    datos: tuple,
    n_workers: int = 1,
) -> dict:
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    n = len(healpix_dirs)
    n_workers = min(n_workers, 8, os.cpu_count() or 8)
    print(f"  Precomputing hemisphere data for {n} directions (n_workers={n_workers})...")

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        init_args = (healpix_dirs, v1, cov_mat)
        precomputed = {}
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_precompute_worker,
            initargs=init_args,
        ) as executor:
            futures = {executor.submit(_precompute_worker, i): i for i in range(n)}
            for future in as_completed(futures):
                idx, data = future.result()
                precomputed[idx] = data
                if len(precomputed) % 100 == 0 or len(precomputed) == n:
                    print(f"    [{len(precomputed)}/{n}] directions")
    else:
        precomputed = {}
        for idx in range(n):
            if idx > 0 and (idx % 100 == 0 or idx == n - 1):
                print(f"    [{idx+1}/{n}] directions")
            dot_products = np.dot(v1, healpix_dirs[idx])
            mask_up = dot_products >= 0
            upi = np.where(mask_up)[0]
            downi = np.where(~mask_up)[0]
            inv_cov_up = np.linalg.inv(cov_mat.iloc[upi, upi].values)
            inv_cov_down = np.linalg.inv(cov_mat.iloc[downi, downi].values)
            precomputed[idx] = {
                "up_indices": upi,
                "down_indices": downi,
                "inv_cov_up": inv_cov_up,
                "inv_cov_down": inv_cov_down,
            }

    return precomputed


# ---- Numba-accelerated public functions ----


def multi_hem_map_numba(healpix_vec: np.ndarray, datos: tuple, method: str = 'golden'):
    """Fit h0/q0 for a HEALPix direction.

    Args:
        method: 'golden' (fast 1D golden section) or 'scipy' (generic scipy, any-D).
    """
    r1 = datos[0]
    v1 = datos[1]
    hostyn = datos[2]
    cov_mat = datos[3]
    q0f = datos[5]
    h0f = datos[4]

    dot_products = np.dot(v1, healpix_vec)
    mask_up = dot_products >= 0
    upi = np.where(mask_up)[0]
    downi = np.where(~mask_up)[0]

    hostyn_up = hostyn[mask_up].astype(np.int64)
    hostyn_down = hostyn[~mask_up].astype(np.int64)

    up = r1[mask_up]
    down = r1[~mask_up]
    z_up = up[:, 2].astype(np.float64)
    z_down = down[:, 2].astype(np.float64)
    mu_sh0es_up = up[:, 5].astype(np.float64)
    muceph_up = up[:, 7].astype(np.float64)
    mu_sh0es_down = down[:, 5].astype(np.float64)
    muceph_down = down[:, 7].astype(np.float64)

    inv_cov_up = np.linalg.inv(cov_mat.iloc[upi, upi].values).astype(np.float64)
    inv_cov_down = np.linalg.inv(cov_mat.iloc[downi, downi].values).astype(np.float64)

    if method == 'golden':
        h0u, h0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, q0f, True, 0.3, 1.5)
        h0d, h0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, q0f, True, 0.3, 1.5)
        q0u, q0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, False, -1.5, 0.5)
        q0d, q0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, False, -1.5, 0.5)
    elif method == 'scipy':
        h0u, _, h0u_err, _ = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (True, False))
        h0d, _, h0d_err, _ = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (True, False))
        _, q0u, _, q0u_err = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (False, True))
        _, q0d, _, q0d_err = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (False, True))
    else:
        raise ValueError(f"Unknown method: {method}")

    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_numba(healpix_dirs: np.ndarray, datos: tuple, save=None, pool: Pool = None, n_workers=1, method: str = 'golden'):
    """Run multi_hem_map_numba across all directions."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    n = len(healpix_dirs)

    if n_workers > 1 and pool is not None:
        args_list = [(h, datos, method) for h in healpix_dirs]
        results_map = list(pool.starmap(multi_hem_map_numba, args_list))
    else:
        print(f"  Processing {n} directions (method={method})...")
        results_map = []
        for i, h in enumerate(healpix_dirs, 1):
            if i % 100 == 0:
                print(f"  [{i}/{n}] directions")
            results_map.append(multi_hem_map_numba(h, datos, method))

    h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err = zip(*results_map)
    results_h0 = (h0u, h0d, h0u_err, h0d_err)
    results_q0 = (q0u, q0d, q0u_err, q0d_err)

    if save is not None:
        header_map = (
            f'Data for Hubble and q0 maps:\n'
            f'{pts} points, q0f={q0f}, h0f={h0f}, zup={zup}, zdown={zdown}\n\n'
            f'h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err'
        )
        filename_map = (
            f'compilations/[NEW][MAP][SH0ES_CALIB]'
            f'(pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt'
        )
        save_data_map = np.column_stack([h0u, h0u_err, h0d, h0d_err, q0u, q0u_err, q0d, q0d_err])
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


def multi_hem_map_fixed_numba(healpix_vec: np.ndarray, dir_idx: int, datos: tuple, precomputed: dict, method: str = 'golden'):
    """Precomputed version for LCDM."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    pc = precomputed[dir_idx]

    hostyn_up = hostyn[pc["up_indices"]].astype(np.int64)
    hostyn_down = hostyn[pc["down_indices"]].astype(np.int64)

    z_up = r1[pc["up_indices"], 2].astype(np.float64)
    z_down = r1[pc["down_indices"], 2].astype(np.float64)
    mu_sh0es_up = r1[pc["up_indices"], 5].astype(np.float64)
    muceph_up = r1[pc["up_indices"], 7].astype(np.float64)
    mu_sh0es_down = r1[pc["down_indices"], 5].astype(np.float64)
    muceph_down = r1[pc["down_indices"], 7].astype(np.float64)
    inv_cov_up = pc["inv_cov_up"].astype(np.float64)
    inv_cov_down = pc["inv_cov_down"].astype(np.float64)

    if method == 'golden':
        h0u, h0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, q0f, True, 0.3, 1.5)
        h0d, h0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, q0f, True, 0.3, 1.5)
        q0u, q0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, False, -1.5, 0.5)
        q0d, q0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, False, -1.5, 0.5)
    elif method == 'scipy':
        h0u, _, h0u_err, _ = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (True, False))
        h0d, _, h0d_err, _ = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (True, False))
        _, q0u, _, q0u_err = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (False, True))
        _, q0d, _, q0d_err = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (False, True))
    else:
        raise ValueError(f"Unknown method: {method}")

    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_fixed_numba(healpix_dirs: np.ndarray, datos: tuple, precomputed: dict, pool: Pool = None, n_workers=1, method: str = 'golden'):
    """Run multi_hem_map_fixed_numba across all directions (for LCDM)."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    n = len(healpix_dirs)

    if n_workers > 1 and pool is not None:
        args_list = [(h, idx, datos, precomputed, method) for idx, h in enumerate(healpix_dirs)]
        results_map = list(pool.starmap(multi_hem_map_fixed_numba, args_list))
    else:
        print(f"  Processing {n} LCDM directions (method={method})...")
        results_map = []
        for idx in range(n):
            if idx > 0 and (idx % 100 == 0 or idx == n - 1):
                print(f"    [{idx+1}/{n}] directions")
            results_map.append(multi_hem_map_fixed_numba(healpix_dirs[idx], idx, datos, precomputed, method))

    h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err = zip(*results_map)
    return (h0u, h0d, h0u_err, h0d_err), (q0u, q0d, q0u_err, q0d_err)
