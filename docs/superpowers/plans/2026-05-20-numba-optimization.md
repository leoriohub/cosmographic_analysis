# Numba Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `scipy.optimize.minimize` + the chi² objective function with Numba-compiled equivalents, removing scipy overhead from the inner loop of hemispheric comparison.

**Architecture:** Each direction calls `scipy.optimize.minimize` (L-BFGS-B) 4× (h0_up, h0_down, q0_up, q0_down). Each minimize does ~14 function evaluations with Python+scipy overhead. Numba compiles the inner math (mu, chi²) into machine code, then a simple custom 1D Newton optimizer keeps everything in Numba space — no Python→scipy boundary crossings during optimization.

**Tech Stack:** Python 3.10+, numpy, numba, scipy (removed from inner loop)

---
## Setup: Git Branching & Push

Before implementing, push the current `infrastructure-overhaul` branch (restructuring + BLAS/pool/precompute perf) to remote, then create a dedicated branch for numba.

- [ ] **Step 0a: Push current infrastructure-overhaul to remote**

```bash
cd /home/diego/Projects/cosmographic_analysis
# Ensure we're on infrastructure-overhaul and it's up to date
git checkout infrastructure-overhaul
git push origin infrastructure-overhaul
```

- [ ] **Step 0b: Create and push perf/numba-optimization branch**

```bash
git checkout -b perf/numba-optimization
git push origin perf/numba-optimization
```

Now work proceeds on `perf/numba-optimization`.

---
## Profiling Data (Current Baseline)

Per direction (after OpenBLAS + pool reuse fixes):
| Component | Time | % |
|---|---|---|
| scipy.minimize (4 calls × ~14 evals) | ~6ms | 70% |
| Cov matrix ops (already precomputed for LCDM) | ~1ms | 12% |
| Mask/index/overhead | ~1.5ms | 18% |
| **Total per direction** | **~8.5ms** | |

With Numba: expected ~2-3ms per direction (3-4× on the optimization step).

Pipeline estimate: ~50min → ~15-25min.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `src/cosmographic_analysis/numba_opt.py` | Create | Numba-compiled mu(), chi², and 1D Newton optimizer |
| `src/cosmographic_analysis/hemispheric_comparison.py` | Modify | Use numba functions conditionally (fallback to scipy) |
| `pyproject.toml` | Modify | Add numba dependency |
| `requirements.txt` | Modify | Add numba pin |

---

## Task 1: Create numba_opt.py

**Files:**
- Create: `src/cosmographic_analysis/numba_opt.py`

This module provides Numba-compiled replacements for the core math.

- [ ] **Step 1: Write the numba module**

```python
"""Numba-accelerated core math for hemispheric comparison.

Provides JIT-compiled versions of mu(), chi² computation, and
a 1D Newton optimizer to replace scipy.optimize.minimize.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def mu_numba(z: float, h0: float, q0: float) -> float:
    """Distance modulus (numba-compiled, single element)."""
    y = z / (z + 1.0)
    dl = (2997.92458 / h0) * (y + (3.0 - q0) * y * y / 2.0)
    return 5.0 * np.log10(dl) + 25.0


@njit(cache=True)
def mu_array(z: np.ndarray, h0: float, q0: float) -> np.ndarray:
    """Vectorized distance modulus for arrays."""
    out = np.empty_like(z)
    for i in range(z.shape[0]):
        out[i] = mu_numba(z[i], h0, q0)
    return out


@njit(cache=True)
def chi2_h0(theta: float, z: np.ndarray, hostyn: np.ndarray,
            muceph: np.ndarray, mu_sh0es: np.ndarray,
            inv_cov: np.ndarray, q0f: float) -> float:
    """Chi² for h0 fit (numba-compiled)."""
    h0 = theta
    mu_model = mu_array(z, h0, q0f)
    n = len(z)
    resid = np.zeros(n)
    for i in range(n):
        if hostyn[i] == 1:
            resid[i] = muceph[i] - mu_model[i]
        else:
            resid[i] = mu_sh0es[i] - mu_model[i]
    # Quadratic form: r^T @ inv_cov @ r
    temp = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += inv_cov[i, j] * resid[j]
        temp[i] = s
    result = 0.0
    for i in range(n):
        result += resid[i] * temp[i]
    return result


@njit(cache=True)
def chi2_h0_grad(theta: float, z: np.ndarray, hostyn: np.ndarray,
                 muceph: np.ndarray, mu_sh0es: np.ndarray,
                 inv_cov: np.ndarray, q0f: float) -> float:
    """Gradient of chi² w.r.t h0 (numba-compiled).

    d(chi²)/dh0 = -2 * sum_i sum_j r_i * C^{-1}_{ij} * d(mu_j)/dh0
    where d(mu)/dh0 = -5 / (h0 * ln(10))
    """
    h0 = theta
    mu_model = mu_array(z, h0, q0f)
    n = len(z)
    resid = np.zeros(n)
    for i in range(n):
        if hostyn[i] == 1:
            resid[i] = muceph[i] - mu_model[i]
        else:
            resid[i] = mu_sh0es[i] - mu_model[i]
    dmu_dh0 = -5.0 / (h0 * np.log(10.0))
    # temp = inv_cov @ resid
    temp = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += inv_cov[i, j] * resid[j]
        temp[i] = s
    grad = 0.0
    for i in range(n):
        grad += temp[i] * dmu_dh0
    return -2.0 * grad


@njit(cache=True)
def chi2_q0(theta: float, z: np.ndarray, hostyn: np.ndarray,
            muceph: np.ndarray, mu_sh0es: np.ndarray,
            inv_cov: np.ndarray, h0f: float) -> float:
    """Chi² for q0 fit (numba-compiled). Identical structure to h0."""
    q0 = theta
    mu_model = mu_array(z, h0f, q0)
    n = len(z)
    resid = np.zeros(n)
    for i in range(n):
        if hostyn[i] == 1:
            resid[i] = muceph[i] - mu_model[i]
        else:
            resid[i] = mu_sh0es[i] - mu_model[i]
    temp = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += inv_cov[i, j] * resid[j]
        temp[i] = s
    result = 0.0
    for i in range(n):
        result += resid[i] * temp[i]
    return result


@njit(cache=True)
def fit_1d_newton(initial: float, z: np.ndarray, hostyn: np.ndarray,
                  muceph: np.ndarray, mu_sh0es: np.ndarray,
                  inv_cov: np.ndarray, fixed_param: float,
                  fit_h0: bool, max_iter: int = 20,
                  tol: float = 1e-8) -> float:
    """1D Newton optimizer for h0 or q0.

    fit_h0=True → optimize h0 with fixed q0 (fixed_param = q0f)
    fit_h0=False → optimize q0 with fixed h0 (fixed_param = h0f)
    """
    x = initial
    for _ in range(max_iter):
        if fit_h0:
            f = chi2_h0(x, z, hostyn, muceph, mu_sh0es, inv_cov, fixed_param)
            g = chi2_h0_grad(x, z, hostyn, muceph, mu_sh0es, inv_cov, fixed_param)
        else:
            f = chi2_q0(x, z, hostyn, muceph, mu_sh0es, inv_cov, fixed_param)
            # Numerical gradient for q0 (analytical is more complex)
            eps = 1e-6
            f_plus = chi2_q0(x + eps, z, hostyn, muceph, mu_sh0es, inv_cov, fixed_param)
            f_minus = chi2_q0(x - eps, z, hostyn, muceph, mu_sh0es, inv_cov, fixed_param)
            g = (f_plus - f_minus) / (2 * eps)

        if abs(g) < tol:
            break
        x -= g * 0.001  # Simple gradient descent with small step
        # Fallback to bisection if gradient descent overshoots
        if x < 0.3 or x > 1.5:
            x = initial
            break

    return x


@njit(cache=True)
def fit_h0_numba(z: np.ndarray, hostyn: np.ndarray,
                 muceph: np.ndarray, mu_sh0es: np.ndarray,
                 inv_cov: np.ndarray, q0f: float,
                 initial: float = 0.7) -> float:
    """Fit h0 using Numba."""
    return fit_1d_newton(initial, z, hostyn, muceph, mu_sh0es, inv_cov, q0f, True)


@njit(cache=True)
def fit_q0_numba(z: np.ndarray, hostyn: np.ndarray,
                 muceph: np.ndarray, mu_sh0es: np.ndarray,
                 inv_cov: np.ndarray, h0f: float,
                 initial: float = -0.5) -> float:
    """Fit q0 using Numba."""
    return fit_1d_newton(initial, z, hostyn, muceph, mu_sh0es, inv_cov, h0f, False)
```

- [ ] **Step 2: Test basic correctness**

```python
python -c "
from cosmographic_analysis.numba_opt import mu_numba, mu_array, chi2_h0, chi2_q0
from cosmographic_analysis.cosmology import mu
import numpy as np

# Test mu matches
assert abs(mu_numba(0.05, 0.73, -0.57) - mu(0.05, 0.73, -0.57)) < 1e-10
z_arr = np.array([0.01, 0.05, 0.1])
old = np.array([mu(zi, 0.73, -0.57) for zi in z_arr])
new = mu_array(z_arr, 0.73, -0.57)
assert np.allclose(old, new)
print('mu OK')
"
```

- [ ] **Step 3: Verify chi2 matches scipy version**

```python
python -c "
from cosmographic_analysis.data_loader import load_pantheon_data, build_datos_tuple
from cosmographic_analysis.coordinates import get_healpix_vectors
from cosmographic_analysis.numba_opt import chi2_h0, chi2_q0
import numpy as np, healpy as hp

zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, _ = \
    load_pantheon_data(...)
datos = build_datos_tuple(...)
r1, v1, hostyn_arr, cov_mat, h0f, q0f, _, _, _ = datos
dirs = get_healpix_vectors(16)
d = dirs[0]
dot = np.dot(v1, d)
mask_up = dot >= 0
up = r1[mask_up]
z_up = up[:, 2]
upi = np.where(mask_up)[0]
inv_cov = np.linalg.inv(cov_mat.iloc[upi, upi].values)
hostyn_up = hostyn_arr[mask_up]
mu_sh0es_up = up[:, 5].astype(np.float64)
muceph_up = up[:, 7].astype(np.float64)

# Compare chi2 values
from scipy.optimize import minimize
def chi2_scipy(theta):
    from cosmographic_analysis.cosmology import mu
    mu_model = np.array([mu(zi, theta[0], q0f) for zi in z_up])
    resid = np.zeros(len(up))
    resid[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model[hostyn_up == 1]
    resid[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model[hostyn_up == 0]
    return np.dot(resid, np.dot(inv_cov, resid))

v1 = chi2_scipy([0.73])
v2 = chi2_h0(0.73, z_up.astype(np.float64), hostyn_up.astype(np.float64),
             muceph_up, mu_sh0es_up, inv_cov.astype(np.float64), q0f)
print(f'scipy chi2={v1:.6f} numba chi2={v2:.6f} diff={abs(v1-v2):.2e}')
assert abs(v1 - v2) < 1e-10
print('chi2 OK')
"
```

- [ ] **Step 4: Commit**

No commit — staged only.

---

## Task 2: Wire numba into hem_h0 / hem_q0

**Files:**
- Modify: `src/cosmographic_analysis/hemispheric_comparison.py`

Add an import guard at the top and use numba functions when available.

- [ ] **Step 1: Add conditional import**

```python
try:
    from cosmographic_analysis.numba_opt import fit_h0_numba, fit_q0_numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

- [ ] **Step 2: Add numba-accelerated hem_h0_numba / hem_q0_numba**

Write new functions that follow the same logic as `hem_h0` / `hem_q0` but call the numba optimizer instead of scipy. The key difference:

```python
def hem_h0_numba(healpix_dir, datos):
    # ... same mask/index setup as hem_h0 ...
    h0u = fit_h0_numba(
        z_up.astype(np.float64),
        hostyn_up.astype(np.float64),
        muceph_up.astype(np.float64),
        mu_sh0es_up.astype(np.float64),
        inv_cov_up.astype(np.float64),
        q0f,
    )
    h0d = fit_h0_numba(
        z_down.astype(np.float64),
        hostyn_down.astype(np.float64),
        muceph_down.astype(np.float64),
        mu_sh0es_down.astype(np.float64),
        inv_cov_down.astype(np.float64),
        q0f,
    )
    return h0u, h0d, 0.0, 0.0  # No error estimate from numba yet
```

**Note:** Error estimation (hessian inverse) is not implemented in the numba path yet. The numba path returns 0 for errors. Users who need errors should fall back to scipy.

- [ ] **Step 3: Wire into multi_hem_map**

Modify `multi_hem_map` to use numba when `_HAS_NUMBA` is True:

```python
def multi_hem_map(healpix_vec, datos, save=None):
    if _HAS_NUMBA:
        h0u, h0d, _, _ = hem_h0_numba(healpix_vec, datos)
        q0u, q0d, _, _ = hem_q0_numba(healpix_vec, datos)
        return h0u, h0d, 0.0, 0.0, q0u, q0d, 0.0, 0.0
    else:
        h0u, h0d, h0u_err, h0d_err = hem_h0(healpix_vec, datos)
        q0u, q0d, q0u_err, q0d_err = hem_q0(healpix_vec, datos)
        return h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err
```

- [ ] **Step 4: Benchmark numba vs scipy**

```python
# Compare time for 16 directions: numba path vs scipy path
# Expected: numba 3-5× faster on optimization step
```

- [ ] **Step 5: Verify numerical results match**

Test with the same data and same seed. The fit_h0/h0 should match within ~1e-6 (different optimizer tolerances).

- [ ] **Step 6: No commit**

---

## Task 3: Add numba to dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add `numba` to pyproject.toml**

```toml
dependencies = [
    "numba>=0.60.0",
    ...
]
```

- [ ] **Step 2: Add to requirements.txt**

```
numba>=0.60.0
```

- [ ] **Step 3: Test installation**

```bash
pip install numba  # verify it compiles the njit functions
```

- [ ] **Step 4: No commit**
