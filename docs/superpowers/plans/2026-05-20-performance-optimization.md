# Performance Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce total pipeline runtime by 40-60% through targeted optimizations of the hemispheric comparison inner loops, with zero changes to numerical results.

**Architecture:** The dominant cost (~99% of runtime) is the hemispheric comparison (`exec_map`), which runs 1001 times per pipeline (1 real data + 500 ISO + 500 LCDM). Each call processes 1536 HEALPix directions via `multiprocessing.Pool.starmap`. Per direction, it runs `hem_h0` + `hem_q0`, each computing: cos-angle mask → subset cov matrix → invert → L-BFGS-B optimization (mu model + quadratic form). The key insight: for LCDM simulations (50% of total work), hemisphere membership never changes between iterations — only `r1[:,5]` does. All mask, indexing, and matrix inversion work is identical across all 500 iterations.

**Tech Stack:** Python 3.10+, numpy, scipy, multiprocessing

**Plan location:** `docs/superpowers/plans/2026-05-20-performance-optimization.md`

---
## Profiling Data (Baseline)

Measured on real data (630 SNe, nside=16, 1536 directions):

| Operation | Time | Notes |
|---|---|---|
| Data loading | 0.2s | Negligible |
| `multi_hem_map` per direction (sequential) | ~0.14s | h0 + q0 combined |
| 32 directions parallel (16 cores) | ~5s | batch overhead |
| `exec_map` full (1536 dirs, 16 cores) | ~240s | estimate |
| ISO loop (500 iterations) | ~33h | 500 × 240s |
| LCDM loop (500 iterations) | ~33h | 500 × 240s |
| **Total pipeline** | **~66h** | baseline |

Per-direction breakdown:
- Cov matrix inversion (×4 per direction): ~42%
- L-BFGS-B optimization (×4 per direction): ~56%
- Other: ~2%

---
## Task 1: Vectorize `mu()` to accept arrays

**Files:**
- Modify: `src/cosmographic_analysis/cosmology.py`

- [ ] **Step 1: Change the `mu()` call in `hemispheric_comparison.py` from list comprehension to vectorized call**

In `hem_h0()`, find:
```python
mu_model_up = np.array([mu(zi, h0, q0f) for zi in z_up])
mu_model_down = np.array([mu(zi, h0, q0f) for zi in z_down])
```
Replace with:
```python
mu_model_up = mu(z_up, h0, q0f)
mu_model_down = mu(z_down, h0, q0f)
```

In `hem_q0()`, find:
```python
mu_model_up = np.array([mu(zi, h0f, q0) for zi in z_up])
mu_model_down = np.array([mu(zi, h0f, q0) for zi in z_down])
```
Replace with:
```python
mu_model_up = mu(z_up, h0f, q0)
mu_model_down = mu(z_down, h0f, q0)
```

- [ ] **Step 2: Verify results are identical**

Run: `python -c "
from cosmographic_analysis.cosmology import mu
import numpy as np
z = np.array([0.01, 0.05, 0.1])
old = np.array([mu(zi, 0.73, -0.57) for zi in z])
new = mu(z, 0.73, -0.57)
assert np.allclose(old, new), 'Results differ!'
print(f'Speedup: {t_seq/t_vec:.0f}x')
"`

Expected: ~100x speedup on mu computation (minor in isolation, compounds across optimizer iterations).

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "perf: vectorize mu() calls in hemispheric comparison"
```

---
## Task 2: Precompute hemisphere indices and cov inversions for LCDM

**Files:**
- Create: `scripts/run_pipeline.py` (modify existing)
- Create: `src/cosmographic_analysis/hemispheric_comparison.py` (add new function `precompute_hemisphere_data`)

**Context:** In the LCDM loop, positions (v1) never change. Only the distance modulus column (r1[:,5]) changes each iteration. This means hemisphere membership (which SNe are "up" vs "down" for each HEALPix direction) is identical across all 500 iterations. Currently, `hem_h0` and `hem_q0` recompute the dot product mask, subset the covariance matrix, and invert it on every call — 500× more work than necessary.

**Implementation plan:** Add a `precompute_hemisphere_data()` function that processes all 1536 directions once, computing and storing:
- For each direction: up/down indices
- For each hemisphere: inverse covariance matrix

Then modify `hem_h0`/`hem_q0` to accept precomputed data, skipping the mask + inversion steps.

- [ ] **Step 1: Add `precompute_hemisphere_data()` to hemispheric_comparison.py**

Write a new function:

```python
def precompute_hemisphere_data(healpix_dirs: np.ndarray, datos: Tuple) -> dict:
    """Precompute hemisphere indices and cov matrix inversions for all directions.

    Since LCDM simulations keep positions fixed, hemisphere membership
    is identical across all iterations. This function computes the
    expensive mask+subset+invert operations once.

    Returns a dict keyed by direction index containing:
      up_indices, down_indices,
      inv_cov_up, inv_cov_down
    """
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos

    precomputed = {}
    for idx, healpix_dir in enumerate(tqdm(healpix_dirs, desc="Precomputing hemisphere data")):
        dot_products = np.dot(v1, healpix_dir)
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
```

- [ ] **Step 2: Add LCDM-specific hem functions**

Add `hem_h0_fixed()` and `hem_q0_fixed()` that accept precomputed data:

```python
def hem_h0_fixed(healpix_dir: np.ndarray, datos: Tuple, precomputed: dict, dir_idx: int) -> Tuple[float, float, float, float]:
    """Same as hem_h0 but uses precomputed indices and inversions."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    q0f = datos[5]

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


def hem_q0_fixed(healpix_dir: np.ndarray, datos: Tuple, precomputed: dict, dir_idx: int) -> Tuple[float, float, float, float]:
    """Same as hem_q0 but uses precomputed indices and inversions."""
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    h0f = datos[4]

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
```

- [ ] **Step 3: Add `exec_map_fixed()` for LCDM**

Add a new function that uses precomputed data internally:

```python
def multi_hem_map_fixed(healpix_vec_and_idx: Tuple, datos: Tuple, precomputed: dict):
    healpix_vec, dir_idx = healpix_vec_and_idx
    h0u, h0d, h0u_err, h0d_err = hem_h0_fixed(healpix_vec, datos, precomputed, dir_idx)
    q0u, q0d, q0u_err, q0d_err = hem_q0_fixed(healpix_vec, datos, precomputed, dir_idx)
    return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err


def exec_map_fixed(healpix_dirs: np.ndarray, datos: Tuple, precomputed: dict):
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    args_list = [(healpix_dir, idx) for idx, healpix_dir in enumerate(healpix_dirs)]

    with Pool() as pool:
        results_map = list(tqdm(
            pool.starmap(multi_hem_map_fixed, [(a, datos, precomputed) for a in args_list]),
            total=len(healpix_dirs),
        ))

    h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err = zip(*results_map)
    return (h0u, h0d, h0u_err, h0d_err), (q0u, q0d, q0u_err, q0d_err)
```

- [ ] **Step 4: Update run_pipeline.py LCDM loop to use precomputed data**

In `scripts/run_pipeline.py`, before the LCDM loop, add:

```python
# Precompute hemisphere data for LCDM (positions don't change)
from cosmographic_analysis.hemispheric_comparison import (
    precompute_hemisphere_data, exec_map_fixed
)
print("  Precomputing hemisphere data for LCDM...")
lcdm_precomputed = precompute_hemisphere_data(healpix_dirs, datos)
```

Then replace the LCDM iteration loop — instead of calling `exec_map()`, call `exec_map_fixed(healpix_dirs, tuple(datos_lcdm), lcdm_precomputed)`.

- [ ] **Step 5: Verify results match original**

Pick a single LCDM iteration and compare `exec_map()` output vs `exec_map_fixed()` output for the same data. Verify all h0u, h0d, q0u, q0d match within tolerance.

- [ ] **Step 6: Benchmark speedup**

Time a single LCDM iteration with original code vs optimized code. Expected: ~2-3x faster per iteration (saves 42% inversion cost + mask computation).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "perf: precompute hemisphere indices and covariance inversions for LCDM"
```

---
## Task 3: Reuse multiprocessing Pool across iterations

**Files:**
- Modify: `src/cosmographic_analysis/hemispheric_comparison.py`
- Modify: `scripts/run_pipeline.py`

**Context:** Both `exec_map()` and `exec_map_fixed()` create a new `Pool()` on every call. For 1000+ iterations, this adds overhead and prevents worker warm-up.

**Simple approach:** Move Pool creation outside the loop. Create it once, pass it in.

- [ ] **Step 1: Modify `exec_map` to accept optional Pool**

```python
def exec_map(healpix_dirs: np.ndarray, datos: Tuple, save=None, pool: Optional[Pool] = None):
    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    args_list = [(healpix_dir, datos) for healpix_dir in healpix_dirs]

    def _execute(pool):
        return list(tqdm(
            pool.starmap(multi_hem_map, args_list),
            total=len(healpix_dirs),
        ))

    if pool is not None:
        results_map = _execute(pool)
    else:
        with Pool() as new_pool:
            results_map = _execute(new_pool)

    # ... rest unchanged
```

Same pattern for `exec_map_fixed`.

- [ ] **Step 2: Update pipeline to reuse Pool**

In `scripts/run_pipeline.py`, before any loops:

```python
from multiprocessing import Pool
pool = Pool()
```

Then pass `pool=pool` to all `exec_map()` and `exec_map_fixed()` calls. After all loops finish, `pool.close()`.

- [ ] **Step 3: Benchmark**

Time 5 sequential `exec_map` calls with new Pool each time vs reused Pool.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "perf: reuse multiprocessing Pool across iterations"
```

---
## Task 4: Benchmark and verify

**Files:** No source changes — benchmarking only.

- [ ] **Step 1: Time each optimization independently**

Run timing for:
1. Vectorized `mu`: 1 real `exec_map` call — should be ~5-10% faster
2. Precomputed LCDM: 1 LCDM-iteration `exec_map_fixed` vs original — should be ~40-50% faster
3. Pool reuse: 10 sequential iterations — should be ~5% faster

- [ ] **Step 2: Estimate total pipeline time**

Scale benchmarks to full 500+500+1 iterations. Document expected improvement.

- [ ] **Step 3: Verify numerical equivalence**

Run a single LCDM iteration with both old and new code. Compare h0u, h0d, q0u, q0d arrays element-by-element. Maximum difference should be < 1e-10 (floating point).
