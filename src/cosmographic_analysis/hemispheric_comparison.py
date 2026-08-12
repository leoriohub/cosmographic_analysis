import os

# Track which warnings/status messages have been printed (avoids per-iteration spam)
_WARNED_GPU_FALLBACK: set[str] = set()
_PRINTED_ONCE: set[str] = set()


def _print_once(key: str, msg: str) -> None:
    """Print a message only once per key (silences repeat noise in loops)."""
    if key not in _PRINTED_ONCE:
        _PRINTED_ONCE.add(key)
        print(msg)

# Prevent BLAS thread oversubscription in multiprocessing workers
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from multiprocessing import Pool

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize

# import distance modulus from cosmology.py
from cosmographic_analysis.cosmology import mu_model

try:
    import cupy as cp
    import cupyx.scipy.linalg  # for solve_triangular
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

from cosmographic_analysis.covariance import (
    MAX_WOODBURY_RANK,
    _build_woodbury_M,
    _check_condition,
    _woodbury_h0_solve,
    decompose_systematic_covariance,
)

# ---- Numba helpers for 1D optimization (golden section search) ----


@njit(cache=True)
def _chi2_1par(theta_val, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model=0):
    """Chi2 for 1-parameter fit. fit_h0=True fits h0, False fits q0."""
    if fit_h0:
        model_mu = mu_model(z, theta_val, theta_fixed, model)
    else:
        model_mu = mu_model(z, theta_fixed, theta_val, model)

    resid = np.empty(len(z))
    for i in range(len(z)):
        if hostyn[i] == 1:
            resid[i] = mu_ceph[i] - model_mu[i]
        else:
            resid[i] = mu_sh0es[i] - model_mu[i]

    Ar = np.dot(resid, np.dot(inv_cov, resid))
    return Ar


@njit(cache=True)
def _golden_fit(z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, a, b, tol=1e-6, model=0):
    """Golden section search for 1-parameter fit. Returns (optimum, error)."""
    phi = (np.sqrt(5) - 1) / 2

    if a > b:
        a, b = b, a

    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = _chi2_1par(c, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)
    fd = _chi2_1par(d, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)

    while abs(b - a) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - phi * (b - a)
            fc = _chi2_1par(c, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = _chi2_1par(d, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)

    x_opt = (a + b) / 2.0

    # Error via central difference on chi2 at minimum
    h = max(tol * 10, abs(x_opt) * 1e-4)
    fp = _chi2_1par(x_opt + h, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)
    fm = _chi2_1par(x_opt - h, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)
    f0 = _chi2_1par(x_opt, z, mu_ceph, mu_sh0es, hostyn, inv_cov, theta_fixed, fit_h0, model)
    hessian = (fp - 2 * f0 + fm) / (h * h)
    error = np.sqrt(2.0 / hessian) if hessian > 0 else 1.0

    return x_opt, error


# ---- Generic chi2 for n-D optimization (scipy fallback) ----


def _fit_hemisphere_scipy(z, mu_ceph, mu_sh0es, hostyn, inv_cov, h0f, q0f, free_mask, model=0):
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
        mu_model = mu_model(z, h0, q0, model)
        resid = np.empty(len(z))
        for i in range(len(z)):
            resid[i] = (mu_ceph[i] - mu_model[i]) if hostyn[i] == 1 else (mu_sh0es[i] - mu_model[i])
        return np.dot(resid, np.dot(inv_cov, resid))

    res = minimize(chi2_func, x0, method="L-BFGS-B", bounds=bounds)

    h0_val = res.x[0] if free_mask[0] else h0f
    q0_val = res.x[1 if free_mask[0] else 0] if free_mask[1] else q0f

    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, "todense"):
            hess_inv = hess_inv.todense()
        errs = np.sqrt(np.abs(np.diag(hess_inv)))
    except Exception:
        errs = np.ones(n_free) * 0.1

    h0_err = errs[0] if free_mask[0] else 0.0
    q0_err = errs[-1] if free_mask[1] else 0.0

    return h0_val, q0_val, h0_err, q0_err


# Module-level shared data for parallel precompute workers
_WORKER_PRECOMPUTE_DIRS: np.ndarray | None = None
_WORKER_PRECOMPUTE_V1: np.ndarray | None = None
_WORKER_PRECOMPUTE_COV: pd.DataFrame | None = None
_WORKER_PRECOMPUTE_COV_NUMPY: np.ndarray | None = None


def _init_precompute_worker(dirs, v1, cov, cov_numpy):
    global _WORKER_PRECOMPUTE_DIRS, _WORKER_PRECOMPUTE_V1, _WORKER_PRECOMPUTE_COV, _WORKER_PRECOMPUTE_COV_NUMPY
    _WORKER_PRECOMPUTE_DIRS = dirs
    _WORKER_PRECOMPUTE_V1 = v1
    _WORKER_PRECOMPUTE_COV = cov
    _WORKER_PRECOMPUTE_COV_NUMPY = cov_numpy


def _precompute_worker(idx: int) -> tuple:
    healpix_dir = _WORKER_PRECOMPUTE_DIRS[idx]
    dot_products = np.dot(_WORKER_PRECOMPUTE_V1, healpix_dir)
    mask_up = dot_products >= 0
    upi = np.where(mask_up)[0]
    downi = np.where(~mask_up)[0]
    inv_cov_up = np.linalg.inv(_WORKER_PRECOMPUTE_COV_NUMPY[np.ix_(upi, upi)])
    inv_cov_down = np.linalg.inv(_WORKER_PRECOMPUTE_COV_NUMPY[np.ix_(downi, downi)])
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
    cov_numpy: np.ndarray | None = None,
) -> dict:
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, _, _ = datos
    if cov_numpy is None:
        cov_numpy = datos[9]
    n = len(healpix_dirs)
    n_workers = min(n_workers, 8, os.cpu_count() or 8)
    print(f"  Precomputing hemisphere data for {n} directions (n_workers={n_workers})...")

    if n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        init_args = (healpix_dirs, v1, cov_mat, cov_numpy)
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
            inv_cov_up = np.linalg.inv(cov_numpy[np.ix_(upi, upi)])
            inv_cov_down = np.linalg.inv(cov_numpy[np.ix_(downi, downi)])
            precomputed[idx] = {
                "up_indices": upi,
                "down_indices": downi,
                "inv_cov_up": inv_cov_up,
                "inv_cov_down": inv_cov_down,
            }

    return precomputed


# ---- GPU-accelerated batched matrix inversion ----


def gpu_batch_invert_hemispheres(
    cov_numpy: np.ndarray,
    direction_indices: list[tuple[np.ndarray, np.ndarray]],
    bin_width: int = 25,
    return_gpu: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not HAVE_CUPY:
        raise ImportError("CuPy not available")

    cov_gpu = cp.asarray(cov_numpy)
    n_dirs = len(direction_indices)
    results = [None] * (n_dirs * 2)

    bins = {}
    for dir_idx, (upi, downi) in enumerate(direction_indices):
        for hemi_idx, h_indices in [(dir_idx * 2, upi), (dir_idx * 2 + 1, downi)]:
            sz = len(h_indices)
            bin_key = sz // bin_width
            bins.setdefault(bin_key, []).append((hemi_idx, h_indices))

    for bin_key in sorted(bins.keys()):
        group = bins[bin_key]
        if not group:
            continue

        sizes = [len(idx) for _, idx in group]
        max_sz = max(sizes)
        n_batch = len(group)

        batch = cp.zeros((n_batch, max_sz, max_sz), dtype=cp.float64)
        for i in range(n_batch):
            batch[i, range(max_sz), range(max_sz)] = 1.0

        for i, (_, h_indices) in enumerate(group):
            sz = len(h_indices)
            if sz == max_sz:
                h_cp = cp.array(h_indices)
                batch[i] = cov_gpu[cp.ix_(h_cp, h_cp)]
            else:
                h_cp = cp.array(h_indices)
                sub = cov_gpu[cp.ix_(h_cp, h_cp)]
                batch[i, :sz, :sz] = sub

        inv_batch = cp.linalg.inv(batch)

        for i, (flat_idx, h_indices) in enumerate(group):
            sz = len(h_indices)
            if return_gpu:
                results[flat_idx] = inv_batch[i, :sz, :sz].copy()
            else:
                inv_cpu = cp.asnumpy(inv_batch[i, :sz, :sz])
                results[flat_idx] = inv_cpu

    result_pairs = [(results[i * 2], results[i * 2 + 1]) for i in range(n_dirs)]
    return result_pairs


def gpu_batch_cholesky_hemispheres(
    cov_numpy: np.ndarray,
    direction_indices: list[tuple[np.ndarray, np.ndarray]],
    bin_width: int = 25,
) -> list[tuple["cp.ndarray", "cp.ndarray"]]:
    """Batched Cholesky factorization of hemisphere covariance submatrices.

    Same grouping and padding logic as gpu_batch_invert_hemispheres but
    returns lower-triangular Cholesky factors instead of inverses.

    Cholesky is ~4× faster than inversion (n³/3 vs 4n³/3 FLOPs).
    Chi² can then be evaluated via triangular solve (O(n²) per grid point).

    Returns: list of (L_up, L_down) tuples, each L is a GPU-resident
             lower-triangular (n_hemi, n_hemi) Cholesky factor.
    """
    if not HAVE_CUPY:
        raise ImportError("CuPy not available")

    n_dirs = len(direction_indices)
    results = [None] * (n_dirs * 2)

    bins = {}
    for dir_idx, (upi, downi) in enumerate(direction_indices):
        for hemi_idx, h_indices in [(dir_idx * 2, upi), (dir_idx * 2 + 1, downi)]:
            sz = len(h_indices)
            bin_key = sz // bin_width
            bins.setdefault(bin_key, []).append((hemi_idx, h_indices))

    # Memory budget: 200 MB per batch (keeps GPU memory under 6GB)
    _MAX_BATCH_BYTES = 200 * 1024 * 1024

    for bin_key in sorted(bins.keys()):
        group = bins[bin_key]
        if not group:
            continue

        sizes = [len(idx) for _, idx in group]
        max_sz = max(sizes)
        n_batch = len(group)
        batch_bytes = n_batch * max_sz * max_sz * 8

        # Free GPU blocks from previous bin before allocating
        cp.get_default_memory_pool().free_all_blocks()

        if batch_bytes > _MAX_BATCH_BYTES:
            max_per_sub = max(1, _MAX_BATCH_BYTES // (max_sz * max_sz * 8))
            for start in range(0, n_batch, max_per_sub):
                sub_group = group[start:start + max_per_sub]
                sub_sizes = [len(idx) for _, idx in sub_group]
                sub_max = max(sub_sizes)
                # Extract submatrices on CPU, then transfer to GPU as a batch
                sub_covs = np.zeros((len(sub_group), sub_max, sub_max), dtype=np.float64)
                for i in range(len(sub_group)):
                    sub_covs[i, range(sub_max), range(sub_max)] = 1.0
                for i, (_, h_indices) in enumerate(sub_group):
                    sz = len(h_indices)
                    sub_covs[i, :sz, :sz] = cov_numpy[np.ix_(h_indices, h_indices)]
                sub_batch = cp.asarray(sub_covs)
                del sub_covs
                sub_chol = cp.linalg.cholesky(sub_batch)
                for i, (flat_idx, h_indices) in enumerate(sub_group):
                    sz = len(h_indices)
                    results[flat_idx] = cp.asnumpy(sub_chol[i, :sz, :sz])
                del sub_batch, sub_chol
        else:
            # Extract submatrices on CPU, batch-transfer to GPU
            covs = np.zeros((n_batch, max_sz, max_sz), dtype=np.float64)
            for i in range(n_batch):
                covs[i, range(max_sz), range(max_sz)] = 1.0
            for i, (_, h_indices) in enumerate(group):
                sz = len(h_indices)
                covs[i, :sz, :sz] = cov_numpy[np.ix_(h_indices, h_indices)]
            batch = cp.asarray(covs)
            del covs
            chol_batch = cp.linalg.cholesky(batch)
            for i, (flat_idx, h_indices) in enumerate(group):
                sz = len(h_indices)
                results[flat_idx] = cp.asnumpy(chol_batch[i, :sz, :sz])
            del batch, chol_batch

    result_pairs = [(results[i * 2], results[i * 2 + 1]) for i in range(n_dirs)]
    return result_pairs


# ---- GPU grid search (batched chi² via single GPU matmul) ----


def _inner_gpu(y, q0, model):
    """Model-dependent inner factor: mu = 5*log10((c/h0)*inner) + 25.

    Mirrors dl_model's inner formulas exactly (all models keep the
    (c/h0)*inner factorization so the analytical-h0 shortcut applies).
    """
    if model == 1:
        return y / (1.0 - (3.0 - q0) * y / 2.0)
    if model == 2:
        f2 = (3.0 - q0) / 2.0
        f3 = (10.0 - 5.0 * q0 + 3.0 * q0 * q0) / 6.0
        return y * (1.0 + (f2 - f3 / f2) * y) / (1.0 - (f3 / f2) * y)
    return y + (3.0 - q0) * y * y / 2.0


def _mu_grid_gpu(
    z_2d: "cp.ndarray",
    h0: "cp.ndarray",
    q0: "cp.ndarray",
    model: int = 0,
) -> "cp.ndarray":
    """Distance modulus on GPU with broadcasting for the grid.

    Computes mu(z, h0, q0) where h0 and q0 can each be either a scalar
    or a 2D column vector (n_grid, 1). Broadcasting handles the rest.

    For h0 fit: call with h0=(n_grid,1), q0=scalar
    For q0 fit: call with h0=scalar, q0=(n_grid,1)
    """
    y = z_2d / (z_2d + 1.0)
    dl_val = (2997.92458 / h0) * _inner_gpu(y, q0, model)
    return 5.0 * cp.log10(dl_val) + 25.0


def _chi2_grid_gpu(
    resid_grid: "cp.ndarray",
    inv_cov_gpu: "cp.ndarray",
) -> "cp.ndarray":
    """Compute chi2 for all grid points: diag(resid @ inv_cov @ resid.T)."""
    x = resid_grid @ inv_cov_gpu
    return cp.sum(resid_grid * x, axis=1)


def _chi2_cholesky_gpu(
    resid_grid: "cp.ndarray",
    L_gpu: "cp.ndarray",
) -> "cp.ndarray":
    """Compute chi2 for all grid points using Cholesky factor.

    Instead of resid @ inv(cov) @ resid^T, solves L @ x = resid via
    triangular solve (O(n²) instead of O(n³)), then chi2 = x·x.

    Args:
        resid_grid: (n_grid, n_hemi) residuals at each grid point
        L_gpu: (n_hemi, n_hemi) lower triangular Cholesky factor of cov

    Returns:
        chi2: (n_grid,) chi² values at each grid point

    """
    x = cupyx.scipy.linalg.solve_triangular(L_gpu, resid_grid.T, lower=True)
    return cp.sum(x * x, axis=0)


def _chi2_woodbury_gpu(
    resid_grid: "cp.ndarray",
    D_inv_H: "cp.ndarray",
    V_H: "cp.ndarray",
    L_M: "cp.ndarray",
) -> "cp.ndarray":
    """Compute chi² for all grid points using the Woodbury identity.

    Uses C⁻¹ = D⁻¹ - D⁻¹·V·M⁻¹·Vᵀ·D⁻¹ where C = D + V·Λ·Vᵀ.

    χ²(r) = rᵀ·D⁻¹·r  -  (rᵀ·D⁻¹·V) · M⁻¹ · (Vᵀ·D⁻¹·r)
          = r·D⁻¹·r    -  zᵀ · v                     where z = Vᵀ·D⁻¹·r, M·v = z

    This replaces the O(n²) triangular solve per grid point (Cholesky path)
    with O(k·n) matvec + O(k²) solve (Woodbury path). For n=315, k=15, this
    is ~20× cheaper per hemisphere.

    Args:
        resid_grid: (n_grid, n_H) residuals at each grid point
        D_inv_H: (n_H,) inverse statistical variances for hemisphere
        V_H: (n_H, k) eigenvectors of C_sys for hemisphere
        L_M: (k, k) lower Cholesky factor of cap matrix M

    Returns:
        chi2: (n_grid,) chi² values at each grid point

    """
    # rDr = diag(R · D⁻¹ · Rᵀ) — diagonal part
    rDr = cp.sum(resid_grid ** 2 * D_inv_H[None, :], axis=1)
    # z = V_Hᵀ · D⁻¹ · Rᵀ — project to k-space, (k, n_grid)
    z = V_H.T @ (D_inv_H[:, None] * resid_grid.T)
    # v = M⁻¹ · z — solve k×k system via two triangular solves, (k, n_grid)
    # M = L_M · L_Mᵀ, so L_M · L_Mᵀ · v = z  →  w = L_M⁻¹ · z, v = L_M⁻ᵀ · w
    w = cupyx.scipy.linalg.solve_triangular(L_M, z, lower=True)
    v = cupyx.scipy.linalg.solve_triangular(L_M.T, w, lower=False)
    # Full chi² = rDr - z·v  (elementwise dot over k)
    return rDr - cp.sum(z * v, axis=0)


def _parabolic_min(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find minimum x via 3-point parabolic interpolation on uniform grid.

    Returns (x_opt, hessian) where hessian = d²(chi²)/dx² at the optimum.

    Uses the analytic formula for a parabola through 3 equally-spaced points,
    avoiding general-purpose polyfit. chi2 is approximately quadratic near
    the minimum, making this both faster and more numerically stable.
    """
    n = len(x)
    if n < 3:
        return float(x[0]), 1.0

    dx = x[1] - x[0]  # uniform grid spacing
    min_idx = int(np.argmin(y))

    if 0 < min_idx < n - 1:
        y0, y1, y2 = y[min_idx - 1], y[min_idx], y[min_idx + 1]
        x1 = x[min_idx]
        lo, hi = x[min_idx - 1], x[min_idx + 1]
    elif min_idx == 0:
        y0, y1, y2 = y[0], y[1], y[2]
        x1 = x[1]
        lo, hi = x[0], x[1]
    else:
        y0, y1, y2 = y[-3], y[-2], y[-1]
        x1 = x[-2]
        lo, hi = x[-2], x[-1]

    # Parabola a*(x - x1)^2 + b*(x - x1) + c through 3 equally-spaced points
    # a = (y0 + y2 - 2*y1) / (2*dx^2),  b = (y2 - y0) / (2*dx)
    denom = y0 + y2 - 2.0 * y1
    if abs(denom) < 1e-15:
        return float(x[min_idx]), 1e-10
    a = denom / (2.0 * dx * dx)
    b = (y2 - y0) / (2.0 * dx)
    x_opt = x1 - b / (2.0 * a)
    x_opt = float(np.clip(x_opt, lo, hi))
    hessian = max(2.0 * a, 1e-10)
    return x_opt, hessian


def _parabolic_min_from_gpu(
    chi2_gpu: "cp.ndarray",
    grid_np: np.ndarray,
) -> tuple[float, float]:
    """GPU-accelerated parabolic minimum: downloads only 3 scalars.

    Like _parabolic_min, but works on a GPU-resident chi² array.
    Finds argmin on GPU, downloads only the 3 neighboring chi² values,
    and computes the parabola analytically on CPU.

    This eliminates downloading the full (n_grid,) chi² array to CPU
    (saves ~98% of PCIe traffic per fit).
    """
    n = len(chi2_gpu)
    min_idx = int(cp.argmin(chi2_gpu))  # 1 int download

    if n < 3:
        return float(grid_np[0]), 1.0

    dx = grid_np[1] - grid_np[0]

    if 0 < min_idx < n - 1:
        idxs = cp.array([min_idx - 1, min_idx, min_idx + 1])
        x1 = grid_np[min_idx]
        lo, hi = grid_np[min_idx - 1], grid_np[min_idx + 1]
    elif min_idx == 0:
        idxs = cp.array([0, 1, 2])
        x1 = grid_np[1]
        lo, hi = grid_np[0], grid_np[1]
    else:
        idxs = cp.array([-3, -2, -1])
        x1 = grid_np[-2]
        lo, hi = grid_np[-2], grid_np[-1]

    # Single batch download: 3 scalars in one call (1 sync instead of 3)
    y0, y1, y2 = [float(v) for v in cp.asnumpy(chi2_gpu[idxs])]

    denom = y0 + y2 - 2.0 * y1
    if abs(denom) < 1e-15:
        return float(grid_np[min_idx]), 1e-10
    a = denom / (2.0 * dx * dx)
    b = (y2 - y0) / (2.0 * dx)
    x_opt = x1 - b / (2.0 * a)
    x_opt = float(np.clip(x_opt, lo, hi))
    hessian = max(2.0 * a, 1e-10)
    return x_opt, hessian


def _exec_map_numba_gpu_grid(
    healpix_dirs: np.ndarray,
    datos: tuple,
    method: str,
    save: str | None = None,
    n_grid: int = 150,
    use_cholesky: bool = True,
    analytical_h0: bool = True,
) -> tuple:
    """GPU grid search: exact h0 via analytical formula + grid search for q0.

    For h0 with q0 fixed: chi²(log h0) is quadratic, so the minimum is
    computed exactly via closed form (no grid search, no interpolation).
    For q0 with h0 fixed: the parameter enters nonlinearly, so grid search
    + parabolic interpolation is used.

    Two factorization strategies:
    - Cholesky (default): cp.linalg.cholesky + triangular solve.
    - Inverse (use_cholesky=False): cp.linalg.inv + matmul.

    Algorithm:
    1. Compute constant part A of the distance modulus, then b = mu_obs - A
    2. For each direction: solve L·z₁ = 1 and L·z₂ = b (for Cholesky path)
    3. α* = -(1ᵀ·z₂) / (1ᵀ·z₁), h₀* = 10^(α*/5) — exact, iteration-free
    4. For q0: batched grid evaluation + parabolic interpolation

    Args:
        n_grid: Number of grid points for q0 search.
        use_cholesky: If True (default), batched Cholesky + triangular solve.
        analytical_h0: If True (default), exact closed-form h0 solution.

    """
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model = datos
    n = len(healpix_dirs)

    # 1. Hemisphere split
    direction_indices = []
    for h in healpix_dirs:
        dot_products = np.dot(v1, h)
        mask_up = dot_products >= 0
        upi = np.where(mask_up)[0]
        downi = np.where(~mask_up)[0]
        direction_indices.append((upi, downi))

    # 2. Factorize hemisphere cov matrices — keep on GPU
    if use_cholesky:
        factors = gpu_batch_cholesky_hemispheres(cov_numpy, direction_indices)
    else:
        factors = gpu_batch_invert_hemispheres(
            cov_numpy, direction_indices, return_gpu=True,
        )

    # 3. Observed mu and redshift on GPU
    mu_obs = np.where(hostyn == 1, r1[:, 7], r1[:, 5]).astype(np.float64)
    mu_obs_gpu = cp.array(mu_obs)
    z_gpu = cp.array(r1[:, 2].astype(np.float64))

    # 4. Grids: h0 uses analytical formula (exact quadratic), q0 uses grid
    q0_grid_np = np.linspace(-1.5, 0.5, n_grid, dtype=np.float64)

    if analytical_h0:
        # b = mu_obs - A where A is the h0-independent part of mu.
        # mu(h0, q0f, z) = 5*log10(c/h0 * inner(y, q0f)) + 25
        #                 = [5*log10(c) + 5*log10(inner(y, q0f)) + 25] - 5*log10(h0)
        #                 = A - 5*log10(h0)
        # r(h0) = mu_obs - mu(h0) = (mu_obs - A) + 5*log10(h0) = b + α·1
        # chi²(α) = (b+α·1)^T C⁻¹ (b+α·1) is QUADRATIC in α → exact minimum
        # (valid for every model: all keep the (c/h0)·inner factorization)
        y_gpu = z_gpu / (z_gpu + 1.0)
        inner = _inner_gpu(y_gpu, cp.float64(q0f), model)
        A_gpu = 5.0 * cp.log10(cp.float64(2997.92458) * inner) + 25.0
        b_gpu = mu_obs_gpu - A_gpu
    else:
        h0_grid_np = np.linspace(0.3, 1.5, n_grid, dtype=np.float64)
        z_2d_h0 = z_gpu[None, :]
        h0_2d = cp.array(h0_grid_np)[:, None]
        mu_h0 = _mu_grid_gpu(z_2d_h0, h0_2d, cp.float64(q0f), model)
        resid_h0 = mu_obs_gpu[None, :] - mu_h0

    # 5. Precompute q0 model grid
    z_2d = z_gpu[None, :]
    q0_2d = cp.array(q0_grid_np)[:, None]
    mu_q0 = _mu_grid_gpu(z_2d, cp.float64(h0f), q0_2d, model)
    resid_q0 = mu_obs_gpu[None, :] - mu_q0

    # 6. Result arrays
    results_h0_u = np.empty(n)
    results_h0_d = np.empty(n)
    results_q0_u = np.empty(n)
    results_q0_d = np.empty(n)
    results_h0_u_err = np.empty(n)
    results_h0_d_err = np.empty(n)
    results_q0_u_err = np.empty(n)
    results_q0_d_err = np.empty(n)

    chi2_fn = _chi2_cholesky_gpu if use_cholesky else _chi2_grid_gpu

    # Track directions with empty hemispheres (guard in Phase B too)
    _empty_hemi = np.zeros(n, dtype=bool)

    # 7. Phase A: evaluate all chi² values (no GPU syncs in the loop)
    if analytical_h0:
        q0_chi2_list = [None] * (n * 2)
        for dir_idx in range(n):
            upi, downi = direction_indices[dir_idx]
            f_up, f_down = factors[dir_idx]
            # Upload to GPU if stored on CPU (from memory-constrained batched Cholesky)
            if not isinstance(f_up, cp.ndarray):
                f_up = cp.asarray(f_up)
            if not isinstance(f_down, cp.ndarray):
                f_down = cp.asarray(f_down)

            # Empty-hemisphere guard: skip directions with no SNe on one side
            if len(upi) == 0 or len(downi) == 0:
                _empty_hemi[dir_idx] = True
                results_h0_u[dir_idx] = h0f
                results_h0_u_err[dir_idx] = 1.0
                results_h0_d[dir_idx] = h0f
                results_h0_d_err[dir_idx] = 1.0
                results_q0_u[dir_idx] = q0f
                results_q0_u_err[dir_idx] = 1.0
                results_q0_d[dir_idx] = q0f
                results_q0_d_err[dir_idx] = 1.0
                continue

            # -- Exact h0 from quadratic in α = 5*log10(h0) --
            b_up, b_down = b_gpu[upi], b_gpu[downi]
            ones_up = cp.ones(len(upi), dtype=cp.float64)
            ones_down = cp.ones(len(downi), dtype=cp.float64)

            if use_cholesky:
                rhs_up = cp.column_stack([ones_up, b_up])
                W_up = cupyx.scipy.linalg.solve_triangular(f_up, rhs_up, lower=True)
                Z_up = cupyx.scipy.linalg.solve_triangular(f_up.T, W_up, lower=False)
                z1_up, z2_up = Z_up[:, 0], Z_up[:, 1]

                rhs_down = cp.column_stack([ones_down, b_down])
                W_down = cupyx.scipy.linalg.solve_triangular(f_down, rhs_down, lower=True)
                Z_down = cupyx.scipy.linalg.solve_triangular(f_down.T, W_down, lower=False)
                z1_down, z2_down = Z_down[:, 0], Z_down[:, 1]
            else:
                z1_up, z2_up = f_up @ ones_up, f_up @ b_up
                z1_down, z2_down = f_down @ ones_down, f_down @ b_down

            o_dot_z1_up = float(cp.dot(ones_up, z1_up))
            o_dot_z2_up = float(cp.dot(ones_up, z2_up))
            a_up = -o_dot_z2_up / o_dot_z1_up if o_dot_z1_up > 0 else 0.0
            results_h0_u[dir_idx] = 10.0 ** (a_up / 5.0)
            results_h0_u_err[dir_idx] = (
                results_h0_u[dir_idx] * np.log(10) / 5.0 / np.sqrt(o_dot_z1_up)
                if o_dot_z1_up > 0 else 1.0
            )

            o_dot_z1_down = float(cp.dot(ones_down, z1_down))
            o_dot_z2_down = float(cp.dot(ones_down, z2_down))
            a_down = -o_dot_z2_down / o_dot_z1_down if o_dot_z1_down > 0 else 0.0
            results_h0_d[dir_idx] = 10.0 ** (a_down / 5.0)
            results_h0_d_err[dir_idx] = (
                results_h0_d[dir_idx] * np.log(10) / 5.0 / np.sqrt(o_dot_z1_down)
                if o_dot_z1_down > 0 else 1.0
            )

            # q0 via grid search
            q0_chi2_list[dir_idx * 2] = chi2_fn(resid_q0[:, upi], f_up)
            q0_chi2_list[dir_idx * 2 + 1] = chi2_fn(resid_q0[:, downi], f_down)
    else:
        h0_grid_np = np.linspace(0.3, 1.5, n_grid, dtype=np.float64)
        chi2_list = [None] * (n * 4)
        for dir_idx in range(n):
            upi, downi = direction_indices[dir_idx]
            f_up, f_down = factors[dir_idx]
            base = dir_idx * 4
            chi2_list[base + 0] = chi2_fn(resid_h0[:, upi], f_up)
            chi2_list[base + 1] = chi2_fn(resid_h0[:, downi], f_down)
            chi2_list[base + 2] = chi2_fn(resid_q0[:, upi], f_up)
            chi2_list[base + 3] = chi2_fn(resid_q0[:, downi], f_down)

    # 8. Phase B: sync, then parabolic interpolation (q0 only if analytical_h0)
    cp.cuda.Stream.null.synchronize()

    if analytical_h0:
        for dir_idx in range(n):
            if _empty_hemi[dir_idx]:
                continue
            q0u_opt, hess_q0u = _parabolic_min_from_gpu(
                q0_chi2_list[dir_idx * 2], q0_grid_np)
            q0d_opt, hess_q0d = _parabolic_min_from_gpu(
                q0_chi2_list[dir_idx * 2 + 1], q0_grid_np)
            results_q0_u[dir_idx] = q0u_opt
            results_q0_d[dir_idx] = q0d_opt
            results_q0_u_err[dir_idx] = np.sqrt(2.0 / hess_q0u) if hess_q0u > 0 else 1.0
            results_q0_d_err[dir_idx] = np.sqrt(2.0 / hess_q0d) if hess_q0d > 0 else 1.0
    else:
        for dir_idx in range(n):
            if _empty_hemi[dir_idx]:
                continue
            base = dir_idx * 4
            h0u_opt, hess_h0u = _parabolic_min_from_gpu(
                chi2_list[base + 0], h0_grid_np)
            h0d_opt, hess_h0d = _parabolic_min_from_gpu(
                chi2_list[base + 1], h0_grid_np)
            results_h0_u[dir_idx] = h0u_opt
            results_h0_d[dir_idx] = h0d_opt
            results_h0_u_err[dir_idx] = np.sqrt(2.0 / hess_h0u) if hess_h0u > 0 else 1.0
            results_h0_d_err[dir_idx] = np.sqrt(2.0 / hess_h0d) if hess_h0d > 0 else 1.0

            q0u_opt, hess_q0u = _parabolic_min_from_gpu(
                chi2_list[base + 2], q0_grid_np)
            q0d_opt, hess_q0d = _parabolic_min_from_gpu(
                chi2_list[base + 3], q0_grid_np)
            results_q0_u[dir_idx] = q0u_opt
            results_q0_d[dir_idx] = q0d_opt
            results_q0_u_err[dir_idx] = np.sqrt(2.0 / hess_q0u) if hess_q0u > 0 else 1.0
            results_q0_d_err[dir_idx] = np.sqrt(2.0 / hess_q0d) if hess_q0d > 0 else 1.0

    results_h0 = (results_h0_u, results_h0_d, results_h0_u_err, results_h0_d_err)
    results_q0 = (results_q0_u, results_q0_d, results_q0_u_err, results_q0_d_err)

    if save is not None:
        header_map = (
            f"Data for Hubble and q0 maps:\n"
            f"{pts} points, q0f={q0f}, h0f={h0f}, zup={zup}, zdown={zdown}\n\n"
            f"h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err"
        )
        filename_map = (
            f"compilations/[NEW][MAP][SH0ES_CALIB]"
            f"(method={method})"
            f"(pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt"
        )
        save_data_map = np.column_stack([
            results_h0_u, results_h0_u_err, results_h0_d, results_h0_d_err,
            results_q0_u, results_q0_u_err, results_q0_d, results_q0_d_err,
        ])
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


def _exec_map_numba_woodbury(
    healpix_dirs: np.ndarray,
    datos: tuple,
    save: str = None,
    n_grid: int = 150,
    use_woodbury_chi2: bool = True,
) -> tuple:
    """Woodbury-based h₀ (CPU) + GPU grid search for q₀.

    Uses the Woodbury identity to compute analytical h₀ for each hemisphere
    without factorising the full n_H×n_H covariance matrix.

    Two q₀ χ² strategies:
    - use_woodbury_chi2=True (default): Woodbury χ² — no batched Cholesky,
      uses the Woodbury identity for each hemisphere's q₀ grid evaluation.
    - use_woodbury_chi2=False: Batched Cholesky + Cholesky χ² — the original
      GPU grid search path (same as method='grid' uses for q₀).

    Sequential k×k updates across adjacent HEALPix directions (RING order)
    reduce the per-direction cost of building the cap matrix M.
    Periodic full recomputation prevents numerical drift.
    """
    if not HAVE_CUPY:
        raise ImportError("Woodbury path requires CuPy for q₀ grid search")

    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model = datos
    n = len(healpix_dirs)
    method = "woodbury" if use_woodbury_chi2 else "woodbury-cholesky"

    # ---- Phase 0: extract systematic covariance and decompose ----
    # Check for cached decomposition (reused across MC iterations)
    # Include n (number of SNe) in filename to prevent cross-run cache poisoning
    sigmuz = r1[:, 6].astype(np.float64)
    D_diag = sigmuz ** 2
    n_sne = len(sigmuz)
    cache_path_v = f"compilations/woodbury_V_{n_sne}.npy"
    cache_path_l = f"compilations/woodbury_Lambda_{n_sne}.npy"
    if os.path.exists(cache_path_v) and os.path.exists(cache_path_l):
        V = np.load(cache_path_v)
        Lambda_d = np.load(cache_path_l)
        k = V.shape[1]
        if k == 0:
            print("  [Woodbury] k=0 (cached, purely diagonal cov), falling back to GPU grid")
            return _exec_map_numba_gpu_grid(healpix_dirs, datos, "grid", save, n_grid=n_grid)
        _print_once("woodbury_cached", f"  [Woodbury] Loaded cached V,Λ (k={k})")
    else:
        V, Lambda_d, _, k = decompose_systematic_covariance(cov_numpy, stat_diag=D_diag)
        if k == 0:
            print("  [Woodbury] k=0 (purely diagonal cov), falling back to GPU grid")
            return _exec_map_numba_gpu_grid(healpix_dirs, datos, "grid", save, n_grid=n_grid)
        np.save(cache_path_v, V)
        np.save(cache_path_l, Lambda_d)

    if k > MAX_WOODBURY_RANK:
        print(f"  [Woodbury] k={k} > MAX={MAX_WOODBURY_RANK}, falling back to GPU grid")
        return _exec_map_numba_gpu_grid(healpix_dirs, datos, "grid", save, n_grid=n_grid)

    _print_once("woodbury_k", f"  [Woodbury] k={k}, {n} directions, x{k} modes")

    D_inv = np.where(D_diag > 0, 1.0 / D_diag, 0.0)
    L_inv = np.where(Lambda_d > 0, 1.0 / Lambda_d, 0.0)

    # ---- Hemisphere split ----
    direction_indices = []
    for h in healpix_dirs:
        dot_products = np.dot(v1, h)
        mask_up = dot_products >= 0
        upi = np.where(mask_up)[0]
        downi = np.where(~mask_up)[0]
        direction_indices.append((upi, downi))

    # ---- Observed mu and redshift on GPU ----
    mu_obs = np.where(hostyn == 1, r1[:, 7], r1[:, 5]).astype(np.float64)
    mu_obs_gpu = cp.array(mu_obs)
    z_gpu = cp.array(r1[:, 2].astype(np.float64))

    # ---- q₀ grid ----
    q0_grid_np = np.linspace(-1.5, 0.5, n_grid, dtype=np.float64)

    # ---- b = mu_obs - A(z, q0f) for analytical h₀ ----
    y_gpu = z_gpu / (z_gpu + 1.0)
    inner = _inner_gpu(y_gpu, cp.float64(q0f), model)
    A_gpu = 5.0 * cp.log10(cp.float64(2997.92458) * inner) + 25.0
    b_gpu = mu_obs_gpu - A_gpu
    b_cpu = cp.asnumpy(b_gpu)

    # ---- q₀ model grid on GPU ----
    z_2d = z_gpu[None, :]
    q0_2d = cp.array(q0_grid_np)[:, None]
    mu_q0 = _mu_grid_gpu(z_2d, cp.float64(h0f), q0_2d, model)
    resid_q0 = mu_obs_gpu[None, :] - mu_q0

    # ---- Pre-transfer V, D_inv to GPU (avoid per-direction cp.asarray in loop) ----
    V_gpu = cp.asarray(V)
    D_inv_gpu = cp.asarray(D_inv)

    # ---- Batched Cholesky factors (only when using legacy chi² path) ----
    if not use_woodbury_chi2:
        print(f"  [Woodbury] Using batched Cholesky for q₀ χ² ({n} directions)...")
        factors = gpu_batch_cholesky_hemispheres(cov_numpy, direction_indices)
    else:
        factors = None

    # ---- Result arrays ----
    results_h0_u = np.empty(n)
    results_h0_d = np.empty(n)
    results_q0_u = np.empty(n)
    results_q0_d = np.empty(n)
    results_h0_u_err = np.empty(n)
    results_h0_d_err = np.empty(n)
    results_q0_u_err = np.empty(n)
    results_q0_d_err = np.empty(n)

    _empty_hemi = np.zeros(n, dtype=bool)
    q0_chi2_list = [None] * (n * 2)

    # ---- Loop over HEALPix directions with sequential updates ----
    for dir_idx in range(n):
        upi, downi = direction_indices[dir_idx]

        if len(upi) == 0 or len(downi) == 0:
            _empty_hemi[dir_idx] = True
            results_h0_u[dir_idx] = h0f
            results_h0_u_err[dir_idx] = 1.0
            results_h0_d[dir_idx] = h0f
            results_h0_d_err[dir_idx] = 1.0
            results_q0_u[dir_idx] = q0f
            results_q0_u_err[dir_idx] = 1.0
            results_q0_d[dir_idx] = q0f
            results_q0_d_err[dir_idx] = 1.0
            continue

        b_up = b_cpu[upi]
        b_down = b_cpu[downi]

        # Full-rebuild Woodbury: build M and Cholesky factor for each hemisphere
        # This is faster than sequential updates for small k (k=15) because
        # _build_woodbury_M uses vectorized BLAS operations (0.06ms/dir).
        _, L_up = _build_woodbury_M(V, L_inv, D_inv, upi)
        _, L_down = _build_woodbury_M(V, L_inv, D_inv, downi)
        if not _check_condition(L_up) or not _check_condition(L_down):
            print(f"  [Woodbury] Warning: M ill-conditioned at direction {dir_idx}, "
                  f"skipping (filling with fiducial values)")
            _empty_hemi[dir_idx] = True
            results_h0_u[dir_idx] = h0f
            results_h0_u_err[dir_idx] = 1.0
            results_h0_d[dir_idx] = h0f
            results_h0_d_err[dir_idx] = 1.0
            results_q0_u[dir_idx] = q0f
            results_q0_u_err[dir_idx] = 1.0
            results_q0_d[dir_idx] = q0f
            results_q0_d_err[dir_idx] = 1.0
            continue

        # h₀ via Woodbury + analytical formula
        h0_up, h0_up_err = _woodbury_h0_solve(V, D_inv, upi, b_up, L_up)
        h0_down, h0_down_err = _woodbury_h0_solve(V, D_inv, downi, b_down, L_down)

        results_h0_u[dir_idx] = h0_up
        results_h0_u_err[dir_idx] = h0_up_err
        results_h0_d[dir_idx] = h0_down
        results_h0_d_err[dir_idx] = h0_down_err

        # q₀ χ²: Woodbury (new) or Cholesky (legacy) path
        if use_woodbury_chi2:
            # V_gpu, D_inv_gpu, resid_q0 are already on GPU (pre-transferred)
            # Only L_up/L_down (k×k = 15×15) need per-direction transfer
            q0_chi2_list[dir_idx * 2] = _chi2_woodbury_gpu(
                resid_q0[:, upi], D_inv_gpu[upi], V_gpu[upi, :], cp.asarray(L_up))
            q0_chi2_list[dir_idx * 2 + 1] = _chi2_woodbury_gpu(
                resid_q0[:, downi], D_inv_gpu[downi], V_gpu[downi, :], cp.asarray(L_down))
        else:
            f_up, f_down = factors[dir_idx]
            q0_chi2_list[dir_idx * 2] = _chi2_cholesky_gpu(resid_q0[:, upi], f_up)
            q0_chi2_list[dir_idx * 2 + 1] = _chi2_cholesky_gpu(resid_q0[:, downi], f_down)

    # ---- Phase B: GPU sync and parabolic interpolation for q₀ ----
    cp.cuda.Stream.null.synchronize()

    for dir_idx in range(n):
        if _empty_hemi[dir_idx]:
            continue
        q0u_opt, hess_q0u = _parabolic_min_from_gpu(
            q0_chi2_list[dir_idx * 2], q0_grid_np)
        q0d_opt, hess_q0d = _parabolic_min_from_gpu(
            q0_chi2_list[dir_idx * 2 + 1], q0_grid_np)
        results_q0_u[dir_idx] = q0u_opt
        results_q0_d[dir_idx] = q0d_opt
        results_q0_u_err[dir_idx] = np.sqrt(2.0 / hess_q0u) if hess_q0u > 0 else 1.0
        results_q0_d_err[dir_idx] = np.sqrt(2.0 / hess_q0d) if hess_q0d > 0 else 1.0

    results_h0 = (results_h0_u, results_h0_d, results_h0_u_err, results_h0_d_err)
    results_q0 = (results_q0_u, results_q0_d, results_q0_u_err, results_q0_d_err)

    if save is not None:
        header_map = (
            f"Data for Hubble and q0 maps:\n"
            f"{pts} points, q0f={q0f}, h0f={h0f}, zup={zup}, zdown={zdown}\n\n"
            f"h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err"
        )
        filename_map = (
            f"compilations/[NEW][MAP][SH0ES_CALIB]"
            f"(method={method})"
            f"(pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt"
        )
        save_data_map = np.column_stack([
            results_h0_u, results_h0_u_err, results_h0_d, results_h0_d_err,
            results_q0_u, results_q0_u_err, results_q0_d, results_q0_d_err,
        ])
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


def _exec_map_numba_gpu(healpix_dirs, datos, save, method):
    """GPU-accelerated golden section fit.

    Note: method parameter is accepted for interface compatibility but this
    function always uses golden section (the grid path is in
    _exec_map_numba_gpu_grid). If 'scipy' is explicitly requested, warn.
    """
    if method == "scipy":
        print("  [GPU] Warning: scipy method not available on GPU, using golden section.")
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model = datos
    n = len(healpix_dirs)

    direction_indices = []
    for h in healpix_dirs:
        dot_products = np.dot(v1, h)
        mask_up = dot_products >= 0
        upi = np.where(mask_up)[0]
        downi = np.where(~mask_up)[0]
        direction_indices.append((upi, downi))

    all_inverses = gpu_batch_invert_hemispheres(cov_numpy, direction_indices)

    results_h0_u = np.empty(n)
    results_h0_d = np.empty(n)
    results_q0_u = np.empty(n)
    results_q0_d = np.empty(n)
    results_h0_u_err = np.empty(n)
    results_h0_d_err = np.empty(n)
    results_q0_u_err = np.empty(n)
    results_q0_d_err = np.empty(n)

    for idx, h in enumerate(healpix_dirs):
        upi, downi = direction_indices[idx]
        inv_cov_up, inv_cov_down = all_inverses[idx]

        up = r1[upi]
        down = r1[downi]

        z_up = up[:, 2].astype(np.float64)
        z_down = down[:, 2].astype(np.float64)
        mu_sh0es_up = up[:, 5].astype(np.float64)
        muceph_up = up[:, 7].astype(np.float64)
        mu_sh0es_down = down[:, 5].astype(np.float64)
        muceph_down = down[:, 7].astype(np.float64)
        hostyn_up = hostyn[upi].astype(np.int64)
        hostyn_down = hostyn[downi].astype(np.int64)

        h0u, h0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, q0f, True, 0.3, 1.5, model=model)
        h0d, h0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, q0f, True, 0.3, 1.5, model=model)
        q0u, q0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, False, -1.5, 0.5, model=model)
        q0d, q0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, False, -1.5, 0.5, model=model)

        results_h0_u[idx] = h0u
        results_h0_d[idx] = h0d
        results_q0_u[idx] = q0u
        results_q0_u_err[idx] = q0u_err
        results_h0_u_err[idx] = h0u_err
        results_h0_d_err[idx] = h0d_err
        results_q0_d[idx] = q0d
        results_q0_d_err[idx] = q0d_err

    results_h0 = (results_h0_u, results_h0_d, results_h0_u_err, results_h0_d_err)
    results_q0 = (results_q0_u, results_q0_d, results_q0_u_err, results_q0_d_err)

    if save is not None:
        header_map = (
            f"Data for Hubble and q0 maps:\n"
            f"{pts} points, q0f={q0f}, h0f={h0f}, zup={zup}, zdown={zdown}\n\n"
            f"h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err"
        )
        filename_map = (
            f"compilations/[NEW][MAP][SH0ES_CALIB]"
            f"(method={method})"
            f"(pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt"
        )
        save_data_map = np.column_stack([
            results_h0_u, results_h0_u_err, results_h0_d, results_h0_d_err,
            results_q0_u, results_q0_u_err, results_q0_d, results_q0_d_err,
        ])
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


# ---- Numba-accelerated public functions ----


def multi_hem_map_numba(healpix_vec: np.ndarray, datos: tuple, method: str = "golden"):
    """Fit h0/q0 for a HEALPix direction.

    Args:
        method: 'golden' (fast 1D golden section), 'grid' (GPU grid search, falls
                back to golden on CPU), 'woodbury', 'woodbury-cholesky' (same fallback),
                or 'scipy' (generic scipy, any-D).

    """
    r1 = datos[0]
    v1 = datos[1]
    hostyn = datos[2]
    cov_numpy = datos[9]
    q0f = datos[5]
    h0f = datos[4]
    model = datos[10]

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

    inv_cov_up = np.linalg.inv(cov_numpy[np.ix_(upi, upi)])
    inv_cov_down = np.linalg.inv(cov_numpy[np.ix_(downi, downi)])

    if method in ("golden", "grid", "woodbury", "woodbury-cholesky"):
        if method in ("grid", "woodbury", "woodbury-cholesky"):
            if method not in _WARNED_GPU_FALLBACK:
                _WARNED_GPU_FALLBACK.add(method)
                print(f"  [CPU] Warning: method='{method}' requires GPU, falling back to golden section.")
        h0u, h0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, q0f, True, 0.3, 1.5, model=model)
        h0d, h0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, q0f, True, 0.3, 1.5, model=model)
        q0u, q0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, False, -1.5, 0.5, model=model)
        q0d, q0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, False, -1.5, 0.5, model=model)
    elif method == "scipy":
        h0u, _, h0u_err, _ = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (True, False), model)
        h0d, _, h0d_err, _ = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (True, False), model)
        _, q0u, _, q0u_err = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (False, True), model)
        _, q0d, _, q0d_err = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (False, True), model)
    else:
        raise ValueError(f"Unknown method: {method}")

    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_numba(healpix_dirs: np.ndarray, datos: tuple, save=None, pool: Pool = None, n_workers=1, method: str = "golden", use_cholesky: bool = True, analytical_h0: bool = True):
    """Run multi_hem_map_numba across all directions.

    Args:
        method: 'golden' (fast numba golden section),
                'grid' (GPU grid search, requires CuPy),
                'woodbury' (CPU Woodbury for h₀ + Woodbury χ² q₀),
                'woodbury-cholesky' (CPU Woodbury for h₀ + Cholesky χ² q₀),
                'scipy' (generic scipy fallback).
        use_cholesky: If True (default), use batched Cholesky for the GPU
                      grid search. If False, use batched inverse.
        analytical_h0: If True (default), exact closed-form h0 solution.
                       Only used when method='grid'.

    """
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model = datos
    n = len(healpix_dirs)

    if HAVE_CUPY and n_workers <= 1 and pool is None:
        if method == "woodbury":
            _print_once("entry_woodbury", f"  [Woodbury] Processing {n} directions...")
            return _exec_map_numba_woodbury(healpix_dirs, datos, save, use_woodbury_chi2=True)
        if method == "woodbury-cholesky":
            _print_once("entry_woodbury_cholesky", f"  [Woodbury+Cholesky] Processing {n} directions...")
            return _exec_map_numba_woodbury(healpix_dirs, datos, save, use_woodbury_chi2=False)
        if method == "grid":
            chol_label = " Cholesky" if use_cholesky else ""
            ana_label = "+analytical h0" if analytical_h0 else ""
            _print_once("entry_grid", f"  [GPU{chol_label} Grid{ana_label}] Processing {n} directions...")
            return _exec_map_numba_gpu_grid(healpix_dirs, datos, method, save, use_cholesky=use_cholesky, analytical_h0=analytical_h0)
        _print_once("entry_gpu", f"  [GPU] Processing {n} directions (method={method})...")
        return _exec_map_numba_gpu(healpix_dirs, datos, save, method)

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
            f"Data for Hubble and q0 maps:\n"
            f"{pts} points, q0f={q0f}, h0f={h0f}, zup={zup}, zdown={zdown}\n\n"
            f"h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err"
        )
        filename_map = (
            f"compilations/[NEW][MAP][SH0ES_CALIB]"
            f"(method={method})"
            f"(pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt"
        )
        save_data_map = np.column_stack([h0u, h0u_err, h0d, h0d_err, q0u, q0u_err, q0d, q0d_err])
        np.savetxt(filename_map, save_data_map, header=header_map)

    return results_h0, results_q0


def multi_hem_map_fixed_numba(healpix_vec: np.ndarray, dir_idx: int, datos: tuple, precomputed: dict, method: str = "golden"):
    """Precomputed version for LCDM.

    Args:
        method: 'golden', 'grid', 'woodbury', 'woodbury-cholesky' (GPU methods
                fall back to golden section on CPU), or 'scipy'.

    """
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, _, _ = datos
    model = datos[10]
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

    if method in ("golden", "grid", "woodbury", "woodbury-cholesky"):
        if method in ("grid", "woodbury", "woodbury-cholesky"):
            if method not in _WARNED_GPU_FALLBACK:
                _WARNED_GPU_FALLBACK.add(method)
                print(f"  [CPU] Warning: method='{method}' requires GPU, falling back to golden section.")
        h0u, h0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, q0f, True, 0.3, 1.5, model=model)
        h0d, h0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, q0f, True, 0.3, 1.5, model=model)
        q0u, q0u_err = _golden_fit(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, False, -1.5, 0.5, model=model)
        q0d, q0d_err = _golden_fit(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, False, -1.5, 0.5, model=model)
    elif method == "scipy":
        h0u, _, h0u_err, _ = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (True, False), model)
        h0d, _, h0d_err, _ = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (True, False), model)
        _, q0u, _, q0u_err = _fit_hemisphere_scipy(z_up, muceph_up, mu_sh0es_up, hostyn_up, inv_cov_up, h0f, q0f, (False, True), model)
        _, q0d, _, q0d_err = _fit_hemisphere_scipy(z_down, muceph_down, mu_sh0es_down, hostyn_down, inv_cov_down, h0f, q0f, (False, True), model)
    else:
        raise ValueError(f"Unknown method: {method}")

    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_fixed_numba(healpix_dirs: np.ndarray, datos: tuple, precomputed: dict, pool: Pool = None, n_workers=1, method: str = "golden"):
    """Run multi_hem_map_fixed_numba across all directions (for LCDM).

    GPU methods (woodbury, woodbury-cholesky, grid) route directly to the
    GPU functions — the precomputed data is only used for CPU fallbacks.
    """
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model = datos
    n = len(healpix_dirs)

    if HAVE_CUPY and n_workers <= 1 and pool is None:
        if method == "woodbury":
            _print_once("lcdm_woodbury", f"  [Woodbury] Processing {n} LCDM directions...")
            return _exec_map_numba_woodbury(healpix_dirs, datos, use_woodbury_chi2=True)
        if method == "woodbury-cholesky":
            _print_once("lcdm_woodbury_cholesky", f"  [Woodbury+Cholesky] Processing {n} LCDM directions...")
            return _exec_map_numba_woodbury(healpix_dirs, datos, use_woodbury_chi2=False)
        if method == "grid":
            _print_once("lcdm_grid", f"  [GPU Grid] Processing {n} LCDM directions...")
            return _exec_map_numba_gpu_grid(healpix_dirs, datos, method)
        _print_once("lcdm_gpu", f"  [GPU] Processing {n} LCDM directions (method={method})...")
        return _exec_map_numba_gpu(healpix_dirs, datos, save=None, method=method)

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
