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
# import distance modulus from cosmology.py
from cosmographic_analysis.cosmology import mu


# Parallel mapping implementation.

# Healpix_dirs is a list of directions which represent each pixel in the healpix pixelation scheme.

def multi_hem_map(healpix_vec: np.ndarray, datos: Tuple, save=None):
    """
    Fit h0 and q0 for a HEALPix direction, sharing covariance inversions.

    Merges hem_h0 + hem_q0 to avoid redundant np.linalg.inv calls
    (they computed identical inverse matrices independently).
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

    up = r1[mask_up]
    down = r1[~mask_up]
    hostyn_up = hostyn[mask_up]
    hostyn_down = hostyn[~mask_up]
    z_up = up[:, 2]
    z_down = down[:, 2]
    mu_sh0es_up = up[:, 5]
    muceph_up = up[:, 7]
    mu_sh0es_down = down[:, 5]
    muceph_down = down[:, 7]

    # Shared cov matrix inversions (done once instead of twice)
    inv_newcovu = np.linalg.inv(cov_mat.iloc[upi, upi].values)
    inv_newcovd = np.linalg.inv(cov_mat.iloc[downi, downi].values)

    # ---------- h0 fit (q0 fixed) ----------
    def chi2uh0(theta):
        h0 = theta[0]
        mu_model_up = mu(z_up, h0, q0f)
        resid_up = np.zeros(len(up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]
        return np.dot(resid_up, np.dot(inv_newcovu, resid_up))

    chi2umin = minimize(chi2uh0, [0.7], method='L-BFGS-B')
    h0u = chi2umin.x[0]
    h0u_err = np.sqrt(chi2umin.hess_inv([1])[0])

    def chi2dh0(theta):
        h0 = theta[0]
        mu_model_down = mu(z_down, h0, q0f)
        resid_down = np.zeros(len(down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down == 0] - mu_model_down[hostyn_down == 0]
        return np.dot(resid_down, np.dot(inv_newcovd, resid_down))

    chi2dmin = minimize(chi2dh0, [0.7], method='L-BFGS-B')
    h0d = chi2dmin.x[0]
    h0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])

    # ---------- q0 fit (h0 fixed) ----------
    def chi2uq0(theta):
        q0 = theta[0]
        mu_model_up = mu(z_up, h0f, q0)
        resid_up = np.zeros(len(up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]
        return np.dot(resid_up, np.dot(inv_newcovu, resid_up))

    chi2umin_q0 = minimize(chi2uq0, [-0.5], method='L-BFGS-B')
    q0u = chi2umin_q0.x[0]
    q0u_err = np.sqrt(chi2umin_q0.hess_inv([1])[0])

    def chi2dq0(theta):
        q0 = theta[0]
        mu_model_down = mu(z_down, h0f, q0)
        resid_down = np.zeros(len(down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down == 0] - mu_model_down[hostyn_down == 0]
        return np.dot(resid_down, np.dot(inv_newcovd, resid_down))

    chi2dmin = minimize(chi2dq0, [-0.5], method='L-BFGS-B')
    q0d = chi2dmin.x[0]
    q0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])

    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


# Exec_map is a function that receives a list of healpix_dirs and maps the hemispheric comparison function to each healpix_dir in parallel.

def exec_map(healpix_dirs: np.ndarray, datos: Tuple, save=None, pool: Pool = None, n_workers=1):
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    n = len(healpix_dirs)
    args_list = [(healpix_dir, datos) for healpix_dir in healpix_dirs]

    if n_workers > 1 and pool is not None:
        results_map = list(
            pool.starmap(multi_hem_map, args_list)
        )
    else:
        print(f"  Processing {n} directions (serial)...")
        results_map = []
        for i, h in enumerate(healpix_dirs, 1):
            if i % 100 == 0:
                print(f"  [{i}/{n}] directions")
            results_map.append(multi_hem_map(h, datos))

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
        save_data_map = np.column_stack(
            [h0u, h0u_err, h0d, h0d_err, q0u, q0u_err, q0d, q0d_err]
        )
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


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


def hem_h0_fixed(healpix_dir: np.ndarray, datos: tuple, precomputed: dict, dir_idx: int):
    """Same as hem_h0 but uses precomputed indices and covariance inversions."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos

    pc = precomputed[dir_idx]
    up = r1[pc["up_indices"]]
    down = r1[pc["down_indices"]]
    z_up = up[:, 2]
    z_down = down[:, 2]

    hostyn_up = hostyn[pc["up_indices"]]
    hostyn_down = hostyn[pc["down_indices"]]
    mu_sh0es_up = up[:, 5]
    muceph_up = up[:, 7]
    mu_sh0es_down = down[:, 5]
    muceph_down = down[:, 7]

    def chi2uh0(theta):
        h0 = theta[0]
        mu_model_up = mu(z_up, h0, q0f)
        resid_up = np.zeros(len(up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]
        return np.dot(resid_up, np.dot(pc["inv_cov_up"], resid_up))

    chi2umin = minimize(chi2uh0, [0.7], method='L-BFGS-B')
    h0u = chi2umin.x[0]
    h0u_err = np.sqrt(chi2umin.hess_inv([1])[0])

    def chi2dh0(theta):
        h0 = theta[0]
        mu_model_down = mu(z_down, h0, q0f)
        resid_down = np.zeros(len(down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down == 0] - mu_model_down[hostyn_down == 0]
        return np.dot(resid_down, np.dot(pc["inv_cov_down"], resid_down))

    chi2dmin = minimize(chi2dh0, [0.7], method='L-BFGS-B')
    h0d = chi2dmin.x[0]
    h0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])

    return h0u, h0d, h0u_err, h0d_err


def hem_q0_fixed(healpix_dir: np.ndarray, datos: tuple, precomputed: dict, dir_idx: int):
    """Same as hem_q0 but uses precomputed indices and covariance inversions."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos

    pc = precomputed[dir_idx]
    up = r1[pc["up_indices"]]
    down = r1[pc["down_indices"]]
    z_up = up[:, 2]
    z_down = down[:, 2]

    hostyn_up = hostyn[pc["up_indices"]]
    hostyn_down = hostyn[pc["down_indices"]]
    muceph_up = up[:, 7]
    muceph_down = down[:, 7]
    mu_sh0es_up = up[:, 5]
    mu_sh0es_down = down[:, 5]

    def chi2uq0(theta):
        q0 = theta[0]
        mu_model_up = mu(z_up, h0f, q0)
        resid_up = np.zeros(len(up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]
        return np.dot(resid_up, np.dot(pc["inv_cov_up"], resid_up))

    chi2umin = minimize(chi2uq0, [-0.5], method='L-BFGS-B')
    q0u = chi2umin.x[0]
    q0u_err = np.sqrt(chi2umin.hess_inv([1])[0])

    def chi2dq0(theta):
        q0 = theta[0]
        mu_model_down = mu(z_down, h0f, q0)
        resid_down = np.zeros(len(down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down == 0] - mu_model_down[hostyn_down == 0]
        return np.dot(resid_down, np.dot(pc["inv_cov_down"], resid_down))

    chi2dmin = minimize(chi2dq0, [-0.5], method='L-BFGS-B')
    q0d = chi2dmin.x[0]
    q0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])

    return q0u, q0d, q0u_err, q0d_err


# Module-level cache for precomputed hemisphere data.
# Set before Pool creation so forked workers inherit it via copy-on-write,
# avoiding gigabytes of pickle overhead on every starmap call.
_WORKER_PRECOMPUTED: Optional[dict] = None


def _init_worker_precomputed(precomputed: dict):
    """Store precomputed data in module-level cache for worker processes."""
    global _WORKER_PRECOMPUTED
    _WORKER_PRECOMPUTED = precomputed


def multi_hem_map_fixed_worker(healpix_vec: np.ndarray, dir_idx: int, datos: tuple):
    """Worker function that reads precomputed data from module-level cache."""

    h0u, h0d, h0u_err, h0d_err = hem_h0_fixed(healpix_vec, datos, _WORKER_PRECOMPUTED, dir_idx)
    q0u, q0d, q0u_err, q0d_err = hem_q0_fixed(healpix_vec, datos, _WORKER_PRECOMPUTED, dir_idx)
    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_fixed(healpix_dirs: np.ndarray, datos: tuple, precomputed: dict, pool: Pool = None, n_workers=1):
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    n = len(healpix_dirs)
    args_list = [(healpix_dir, idx, datos) for idx, healpix_dir in enumerate(healpix_dirs)]

    # Note: in parallel mode, the caller must have called _init_worker_precomputed()
    # BEFORE creating the pool, so forked workers inherit the data via copy-on-write.
    _init_worker_precomputed(precomputed)

    if n_workers > 1 and pool is not None:
        results_map = list(pool.starmap(multi_hem_map_fixed_worker, args_list))
    else:
        print(f"  Processing {n} LCDM directions (serial)...")
        results_map = []
        for i, (h, idx, d) in enumerate(args_list, 1):
            if i % 100 == 0:
                print(f"    [{i}/{n}] directions")
            results_map.append(multi_hem_map_fixed_worker(h, idx, d))

    h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err = zip(*results_map)
    return (h0u, h0d, h0u_err, h0d_err), (q0u, q0d, q0u_err, q0d_err)
