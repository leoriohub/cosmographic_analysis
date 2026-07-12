"""Covariance decomposition and Woodbury-matrix analytical h₀ fit.

Pantheon+ covariance structure:
    C      = D + C_sys        (n × n)
    D      = diag(σ_i²)       (statistical uncertainty)
    C_sys  = V · Λ · V^T      (systematic, low-rank eigendecomposed)

For a hemisphere H (|H| = n_H):
    C_HH⁻¹ = D_HH⁻¹ - D_HH⁻¹·V_H · (Λ⁻¹ + V_H^T·D_HH⁻¹·V_H)⁻¹ · V_H^T·D_HH⁻¹

The k×k "cap matrix" M = Λ⁻¹ + V_H^T·D_HH⁻¹·V_H is built and factored once,
then solved for y₁ = V_H^T·D_HH⁻¹·1 and y_b = V_H^T·D_HH⁻¹·b.

Analytical h₀:
    α* = -(1ᵀ·C⁻¹·b) / (1ᵀ·C⁻¹·1)
    h₀* = 10^(α*/5)
"""

from typing import Optional

import numpy as np
from scipy import linalg

# Maximum rank for which Woodbury is worthwhile.
# If effective rank > MAX_WOODBURY_RANK, fall back to full Cholesky.
MAX_WOODBURY_RANK = 100


# ---------------------------------------------------------------------------
# Phase 0: Systematic covariance decomposition
# ---------------------------------------------------------------------------


def decompose_systematic_covariance(
    cov_numpy: np.ndarray,
    stat_diag: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract V, Λ from the systematic part of the covariance matrix.

    Pantheon+ covariance: C = D_stat + C_sys, where D_stat = diag(σ_i²)
    is the statistical-only uncertainty (from sigmuz column), and C_sys
    is the systematic covariance built from nuisance parameter shifts.

    C_sys is low-rank (≈ 20-50) because it is constructed as a sum of
    rank-1 matrices from systematic parameters.

    Parameters
    ----------
    cov_numpy : (n, n) ndarray
        Full Pantheon+ covariance matrix (stat + sys).
    stat_diag : (n,) ndarray or None
        Statistical-only diagonal σ_i² (sigmuz² from data). If None,
        falls back to diag(cov_numpy), which mixes stat and sys variance.

    Returns
    -------
    V : (n, k) ndarray
        Eigenvectors of C_sys, columns sorted by eigenvalue descending.
    Lambda_diag : (k,) ndarray
        Positive eigenvalues of C_sys, descending.
    D_diag : (n,) ndarray
        Statistical-only diagonal (σ_i²) used for Woodbury.
    k : int
        Effective rank (number of eigenvalues kept, |k| < MAX_WOODBURY_RANK).
    """
    n = cov_numpy.shape[0]
    if stat_diag is not None:
        D_diag = np.asarray(stat_diag, dtype=np.float64).copy()
    else:
        D_diag = np.diag(cov_numpy).copy()

    # C_sys = C - D_stat
    C_sys = cov_numpy - np.diag(D_diag)

    # Eigendecompose the symmetric systematic matrix
    eigenvalues, eigenvectors = linalg.eigh(C_sys)

    # Positive eigenvalues above machine precision noise floor
    eps = max(np.finfo(np.float64).eps * n, 1e-15 * np.max(eigenvalues))
    pos_mask = eigenvalues > eps
    pos_vals = eigenvalues[pos_mask]
    pos_vecs = eigenvectors[:, pos_mask]

    if len(pos_vals) == 0:
        return np.empty((n, 0)), np.empty(0), D_diag, 0

    # Sort descending
    sort_idx = np.argsort(pos_vals)[::-1]
    pos_vals = pos_vals[sort_idx]
    pos_vecs = pos_vecs[:, sort_idx]

    # Clamp to manageable rank
    k = min(len(pos_vals), MAX_WOODBURY_RANK)

    return pos_vecs[:, :k], pos_vals[:k], D_diag, k


# ---------------------------------------------------------------------------
# Phase 1: Woodbury analytical h₀
# ---------------------------------------------------------------------------


# Condition number threshold for Cholesky factor diagonal ratio.
# If max(diag(L)) / min(diag(L)) > CONDITION_THRESHOLD, trigger full rebuild.
CONDITION_THRESHOLD = 1e10


def _check_condition(L: np.ndarray) -> bool:
    """Check if Cholesky factor condition is acceptable.

    Uses the ratio of extreme diagonal elements as a cheap condition
    estimate. Returns False (unhealthy) if max/min > CONDITION_THRESHOLD.
    """
    d = np.diag(L)
    ratio = np.max(d) / np.maximum(np.min(d), 1e-300)
    return ratio < CONDITION_THRESHOLD


def _build_woodbury_M(
    V: np.ndarray,
    L_inv: np.ndarray,
    D_inv: np.ndarray,
    H_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the k×k cap matrix M and its lower Cholesky factor.

    M = Λ⁻¹ + V_H^T·D_HH⁻¹·V_H
      = diag(L_inv) + Σ_i D_inv[H[i]] · V[H[i], :] · V[H[i], :]^T

    Parameters
    ----------
    V : (n, k) ndarray
        Eigenvectors of C_sys.
    L_inv : (k,) ndarray
        1 / λ_i  (inverse eigenvalues).
    D_inv : (n,) ndarray
        1 / σ_i²  (inverse statistical variances).
    H_indices : (n_H,) ndarray
        Indices of SNe in this hemisphere.

    Returns
    -------
    M : (k, k) ndarray
        The cap matrix (symmetric positive definite).
    L_M : (k, k) ndarray
        Lower-triangular Cholesky factor of M.
    """
    k = V.shape[1]
    M = np.diag(L_inv)

    V_H = V[H_indices, :]
    w_H = D_inv[H_indices]
    M += V_H.T @ (w_H[:, np.newaxis] * V_H)
    L_M = np.linalg.cholesky(M)
    return M, L_M


def _woodbury_h0_fit(
    V: np.ndarray,
    L_inv: np.ndarray,
    D_inv: np.ndarray,
    H_indices: np.ndarray,
    b_H: np.ndarray,
) -> tuple[float, float]:
    """Analytical h₀ fit via Woodbury identity.

    Returns (h0_opt, h0_err).
    """
    # Step 1: Weighted vectors
    w_1 = D_inv[H_indices]                 # D_HH⁻¹·1
    w_b = D_inv[H_indices] * b_H           # D_HH⁻¹·b_H

    # Step 2: Project to k-space
    V_H = V[H_indices, :]                  # (n_H, k)
    y_1 = V_H.T @ w_1                      # (k,)
    y_b = V_H.T @ w_b                      # (k,)

    # Step 3: Build and factor M
    _, L_M = _build_woodbury_M(V, L_inv, D_inv, H_indices)

    # Step 4: Solve M·v = y via Cholesky
    v_1 = linalg.cho_solve((L_M, True), y_1)   # True = lower
    v_b = linalg.cho_solve((L_M, True), y_b)

    # Step 5: Dot products with Woodbury correction
    dot_1_1 = np.sum(w_1)                        # 1^T·D_HH⁻¹·1
    dot_1_b = np.sum(w_b)                        # 1^T·D_HH⁻¹·b_H
    corr_1 = np.dot(y_1, v_1)                    # y_1^T·M⁻¹·y_1
    corr_b = np.dot(y_1, v_b)                    # y_1^T·M⁻¹·y_b (note: y_1, not y_b!)

    num_1 = dot_1_1 - corr_1                     # 1^T·C_HH⁻¹·1
    num_b = dot_1_b - corr_b                     # 1^T·C_HH⁻¹·b_H

    # Step 6: Closed-form minimum
    if num_1 <= 0.0:
        return 0.0, 1.0

    alpha = -num_b / num_1
    h0_opt = 10.0 ** (alpha / 5.0)
    h0_err = h0_opt * np.log(10.0) / 5.0 / np.sqrt(num_1)

    return h0_opt, h0_err


def _woodbury_h0_solve(
    V: np.ndarray,
    D_inv: np.ndarray,
    H_indices: np.ndarray,
    b_H: np.ndarray,
    L_M: np.ndarray,
) -> tuple[float, float]:
    """Analytical h₀ fit using a **pre-built** Cholesky factor L_M.

    Use this in the sequential-update path where L_M has already been
    updated/downdated from the previous HEALPix direction.

    Parameters
    ----------
    L_M : (k, k) ndarray
        Lower-triangular Cholesky factor of M (pre-built or updated).

    Returns (h0_opt, h0_err).
    """
    w_1 = D_inv[H_indices]
    w_b = D_inv[H_indices] * b_H

    V_H = V[H_indices, :]
    y_1 = V_H.T @ w_1
    y_b = V_H.T @ w_b

    v_1 = linalg.cho_solve((L_M, True), y_1)
    v_b = linalg.cho_solve((L_M, True), y_b)

    dot_1_1 = np.sum(w_1)
    dot_1_b = np.sum(w_b)
    corr_1 = np.dot(y_1, v_1)
    corr_b = np.dot(y_1, v_b)

    num_1 = dot_1_1 - corr_1
    num_b = dot_1_b - corr_b

    if num_1 <= 0.0:
        return 0.0, 1.0

    alpha = -num_b / num_1
    h0_opt = 10.0 ** (alpha / 5.0)
    h0_err = h0_opt * np.log(10.0) / 5.0 / np.sqrt(num_1)

    return h0_opt, h0_err


# ---------------------------------------------------------------------------
# Phase 2: Cholesky rank-1 update / downdate (LINPACK dchud / dchdd)
# ---------------------------------------------------------------------------


def _chol_rank1_update(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank-1 update of lower Cholesky factor.

    Given L·L^T = M, compute L_new such that L_new·L_new^T = M + x·x^T.

    Implements the LINPACK dchud algorithm (O(k²), Givens rotations).

    Parameters
    ----------
    L : (k, k) ndarray
        Lower-triangular Cholesky factor.
    x : (k,) ndarray
        Vector for rank-1 update.

    Returns
    -------
    L_new : (k, k) ndarray
        Updated lower-triangular Cholesky factor.
    """
    L_new = L.copy()
    k = L.shape[0]
    x_local = x.copy().astype(np.float64)

    for j in range(k):
        # Givens rotation: zero out x_local[j] by rotating with L[j,j]
        a = L_new[j, j]
        b = x_local[j]
        r = np.sqrt(a * a + b * b)
        if r == 0.0:
            continue
        c = a / r   # cos
        s = b / r   # sin
        L_new[j, j] = r

        for i in range(j + 1, k):
            t = c * L_new[i, j] + s * x_local[i]
            x_local[i] = -s * L_new[i, j] + c * x_local[i]
            L_new[i, j] = t

    return L_new


def _chol_rank1_downdate(L: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
    """Rank-1 downdate of lower Cholesky factor.

    Given L·L^T = M, compute L_new such that L_new·L_new^T = M - x·x^T.

    Implements the LINPACK dchdd algorithm (O(k²), hyperbolic Givens rotations).

    Parameters
    ----------
    L : (k, k) ndarray
        Lower-triangular Cholesky factor.
    x : (k,) ndarray
        Vector for rank-1 downdate.

    Returns
    -------
    L_new : (k, k) ndarray
        Updated lower-triangular Cholesky factor (may be invalid if info > 0).
    info : int
        0 on success, 1 if the downdate would make M non-positive-definite.
    """
    L_new = L.copy()
    k = L.shape[0]
    x_local = x.copy().astype(np.float64)

    for j in range(k):
        a = L_new[j, j]
        b = x_local[j]
        r_sq = a * a - b * b

        if r_sq <= 0.0:
            return L_new, 1

        r = np.sqrt(r_sq)
        c = a / r
        s = b / r
        L_new[j, j] = r

        for i in range(j + 1, k):
            t = c * L_new[i, j] - s * x_local[i]
            x_local[i] = -s * L_new[i, j] + c * x_local[i]
            L_new[i, j] = t

    return L_new, 0


def _update_woodbury_M(
    L_M: np.ndarray,
    V_j: np.ndarray,
    sigma_j: float,
    op: str = "add",
) -> tuple[np.ndarray, int]:
    """Rank-1 update or downdate of the Woodbury cap matrix's Cholesky factor.

    When adding SN j to the hemisphere:
        M_new = M_old + (1/σ_j²)·V_j,:^T·V_j,:
    This is a rank-1 update with x = (1/σ_j)·V_j,:.

    When removing:
        M_new = M_old - (1/σ_j²)·V_j,:^T·V_j,:
    This is a rank-1 downdate with x = (1/σ_j)·V_j,:.

    Parameters
    ----------
    L_M : (k, k) ndarray
        Current lower Cholesky factor of M.
    V_j : (k,) ndarray
        Eigenvector row for SN j  (V[j, :]).
    sigma_j : float
        Statistical uncertainty of SN j.
    op : {"add", "remove"}
        Whether to add or remove the SN from the hemisphere.

    Returns
    -------
    L_M_new : (k, k) ndarray
        Updated lower Cholesky factor.
    info : int
        0 on success, 1 on downdate failure (non-PD matrix).
        Always 0 for 'add'.
    """
    x = V_j / sigma_j  # (k,) — the weighted eigenvector

    if op == "add":
        return _chol_rank1_update(L_M, x), 0
    elif op == "remove":
        return _chol_rank1_downdate(L_M, x)
    else:
        raise ValueError(f"op must be 'add' or 'remove', got '{op}'")


# ---------------------------------------------------------------------------
# Helper: Full Cholesky-based analytical h₀ (for verification)
# ---------------------------------------------------------------------------


def _direct_cholesky_h0_fit(
    cov_numpy: np.ndarray,
    H_indices: np.ndarray,
    b_H: np.ndarray,
) -> tuple[float, float]:
    """Analytical h₀ via full Cholesky (for verification only).

    Computes C_HH = cov_numpy[np.ix_(H, H)], factors C_HH = L·L^T,
    solves L·z₁ = 1, L·z₂ = b, and returns (h0_opt, h0_err).
    """
    C_HH = cov_numpy[np.ix_(H_indices, H_indices)]
    L = np.linalg.cholesky(C_HH)

    ones = np.ones(len(H_indices))
    z_1 = linalg.cho_solve((L, True), ones)
    z_2 = linalg.cho_solve((L, True), b_H)

    num_1 = np.dot(ones, z_1)
    num_b = np.dot(ones, z_2)

    if num_1 <= 0.0:
        return 0.0, 1.0

    alpha = -num_b / num_1
    h0_opt = 10.0 ** (alpha / 5.0)
    h0_err = h0_opt * np.log(10.0) / 5.0 / np.sqrt(num_1)

    return h0_opt, h0_err
