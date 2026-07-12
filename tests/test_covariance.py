"""Tests for covariance decomposition and Woodbury analytical h₀.

Follows physics-testing patterns:
  1. Known analytic values — direct inversion comparison
  2. Invariants — Cholesky update preserves M + xx^T
  3. Roundtrips — add+remove recovers original
  4. Edge cases — empty hemisphere, single SN, failed downdate
  5. Regression locks — shape contracts, rank detection
"""

import numpy as np
import pytest
from scipy import linalg

try:
    import cupy as cp
    import cupyx.scipy.linalg
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

from cosmographic_analysis.covariance import (
    _chol_rank1_update,
    _chol_rank1_downdate,
    _update_woodbury_M,
    _build_woodbury_M,
    _woodbury_h0_fit,
    _woodbury_h0_solve,
    _direct_cholesky_h0_fit,
    decompose_systematic_covariance,
    MAX_WOODBURY_RANK,
)

if HAVE_CUPY:
    from cosmographic_analysis.hemispheric_comparison import (
        _chi2_cholesky_gpu,
        _chi2_woodbury_gpu,
    )

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def spd_matrix_5(rng):
    """A well-conditioned 5×5 SPD matrix."""
    M = rng.standard_normal((5, 5))
    M = M.T @ M + np.eye(5) * 0.5
    return M


@pytest.fixture
def chol_factor_5(spd_matrix_5):
    return np.linalg.cholesky(spd_matrix_5)


@pytest.fixture
def realistic_cov_data(rng):
    """Build a realistic n=200 covariance with known rank-15 C_sys."""
    n, k_true = 200, 15
    Lambda_true = np.sort(np.exp(-np.arange(k_true) * 0.5) * 10.0)[::-1]
    # Orthonormal basis for C_sys
    U, _ = np.linalg.qr(rng.standard_normal((n, k_true)))
    C_sys = U @ np.diag(Lambda_true) @ U.T
    # Statistical diagonal
    D_stat = np.abs(rng.standard_normal(n)) + 0.01
    cov = C_sys + np.diag(D_stat)
    return cov, D_stat, C_sys, Lambda_true, k_true, U


# ===================================================================
# Cholesky rank-1 update / downdate
# ===================================================================


class TestCholRank1Update:
    def test_update_preserves_m_plus_xxT(self, spd_matrix_5, chol_factor_5, rng):
        """Invariant: L_new·L_new^T = M + x·x^T."""
        x = rng.standard_normal(5)
        L_new = _chol_rank1_update(chol_factor_5, x)
        M_new = L_new @ L_new.T
        expected = spd_matrix_5 + np.outer(x, x)
        assert np.allclose(M_new, expected, atol=1e-12)

    def test_update_add_remove_roundtrip(self, chol_factor_5, rng):
        """Roundtrip: add x then remove x recovers original M."""
        x = rng.standard_normal(5)
        L_updated = _chol_rank1_update(chol_factor_5, x)
        L_restored, info = _chol_rank1_downdate(L_updated, x)
        assert info == 0
        original = chol_factor_5 @ chol_factor_5.T
        restored = L_restored @ L_restored.T
        assert np.allclose(original, restored, atol=1e-12)

    def test_downdate_failure_detected(self, rng):
        """Edge: downdate of M - x·x^T where x·x^T > M detects non-PD."""
        M = np.eye(3) * 1.0
        L = np.linalg.cholesky(M)
        x_big = np.array([10.0, 0.0, 0.0])
        _, info = _chol_rank1_downdate(L, x_big)
        assert info == 1, "Should detect non-PD matrix"

    def test_multiple_updates_match_batch(self, rng):
        """20 sequential updates give same result as batch rebuild."""
        k = 10
        M0 = np.eye(k) * 2.0
        L0 = np.linalg.cholesky(M0)
        xs = rng.standard_normal((20, k))

        L_seq = L0.copy()
        for x_i in xs:
            L_seq = _chol_rank1_update(L_seq, x_i)

        M_batch = M0.copy()
        for x_i in xs:
            M_batch += np.outer(x_i, x_i)
        L_batch = np.linalg.cholesky(M_batch)

        assert np.allclose(L_seq @ L_seq.T, M_batch, atol=1e-12)

    def test_full_update_downdate_cycle(self, rng):
        """Full 20x add then 20x remove returns to original."""
        k = 10
        M0 = np.eye(k) * 2.0
        L0 = np.linalg.cholesky(M0)
        xs = rng.standard_normal((20, k))

        L = L0.copy()
        for x_i in xs:
            L = _chol_rank1_update(L, x_i)
        for x_i in xs:
            L, info = _chol_rank1_downdate(L, x_i)
            assert info == 0

        assert np.allclose(L @ L.T, M0, atol=1e-12)

    def test_update_with_single_element(self):
        """Edge: k=1 Cholesky update works correctly."""
        L = np.array([[2.0]])
        x = np.array([3.0])
        L_new = _chol_rank1_update(L, x)
        assert np.allclose(L_new @ L_new.T, np.array([[13.0]]), atol=1e-12)

    def test_update_woodbury_matches_direct(self, rng):
        """_update_woodbury_M with sigma_j matches direct L change."""
        k = 5
        M = np.eye(k) * 3.0
        L = np.linalg.cholesky(M)
        V_j = rng.standard_normal(k)
        sigma_j = 1.5

        L_add, info = _update_woodbury_M(L, V_j, sigma_j, op="add")
        assert info == 0
        x = V_j / sigma_j
        expected = L @ L.T + np.outer(x, x)
        assert np.allclose(L_add @ L_add.T, expected, atol=1e-12)


# ===================================================================
# Woodbury analytical h₀
# ===================================================================


class TestWoodburyHFit:
    def test_matches_direct_cholesky(self, realistic_cov_data, rng):
        """Known analytic value: Woodbury = direct Cholesky to 1e-12."""
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, k = decompose_systematic_covariance(cov, stat_diag=D_stat)
        assert k == k_true

        # Test 50 random hemispheres
        n = cov.shape[0]
        max_rel = 0.0
        for _ in range(50):
            H_idx = np.sort(rng.choice(n, 80, replace=False))
            b_H = rng.standard_normal(80)
            h0_w, _ = _woodbury_h0_fit(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx, b_H)
            h0_d, _ = _direct_cholesky_h0_fit(cov, H_idx, b_H)
            rel = abs(h0_w - h0_d) / max(abs(h0_d), 1e-10)
            max_rel = max(max_rel, rel)
        assert max_rel < 1e-12, f"Max rel diff: {max_rel}"

    def test_woodbury_solve_matches_fit(self, realistic_cov_data, rng):
        """_woodbury_h0_solve with pre-built L_M = _woodbury_h0_fit."""
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)

        n = cov.shape[0]
        H_idx = np.sort(rng.choice(n, 80, replace=False))
        b_H = rng.standard_normal(80)

        h0_fit, _ = _woodbury_h0_fit(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx, b_H)
        _, L_M = _build_woodbury_M(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx)
        h0_solve, _ = _woodbury_h0_solve(V, 1.0 / D_diag, H_idx, b_H, L_M)

        assert abs(h0_fit - h0_solve) < 1e-14

    def test_single_sn_hemisphere(self, realistic_cov_data, rng):
        """Edge: hemisphere with a single SN still produces valid result."""
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)

        H_idx = np.array([0])
        b_H = np.array([0.5])

        h0, h0_err = _woodbury_h0_fit(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx, b_H)
        assert np.isfinite(h0)
        assert h0_err > 0
        assert h0 > 0

    def test_error_is_positive_finite(self, realistic_cov_data, rng):
        """Invariant: h0 error is always positive and finite."""
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)

        n = cov.shape[0]
        for _ in range(20):
            H_idx = np.sort(rng.choice(n, 80, replace=False))
            b_H = rng.standard_normal(80)
            _, h0_err = _woodbury_h0_fit(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx, b_H)
            assert np.isfinite(h0_err) and h0_err > 0

    def test_h0_equals_input_when_b_H_is_zero(self, realistic_cov_data, rng):
        """Invariant: b_H=0 → h0 = 10^(0/5) = 1.0."""
        cov, D_stat, _, _, _, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)

        n = cov.shape[0]
        H_idx = np.sort(rng.choice(n, 80, replace=False))
        b_H = np.zeros(80)

        h0, _ = _woodbury_h0_fit(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx, b_H)
        assert h0 == pytest.approx(1.0, abs=1e-14)


# ===================================================================
# Covariance decomposition
# ===================================================================


class TestDecomposeSystematicCovariance:
    def test_recovers_known_rank_and_eigenvalues(self, realistic_cov_data):
        """Rank and eigenvalues recovered correctly with known C_sys."""
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, k = decompose_systematic_covariance(cov, stat_diag=D_stat)
        assert k == k_true
        assert np.allclose(Lambda_d, Lambda_true, atol=1e-10)
        assert D_diag == pytest.approx(D_stat)

    def test_shapes_are_correct(self, realistic_cov_data):
        """Shape contract: V (n, k), Lambda (k,), D_diag (n,)."""
        cov, D_stat, _, _, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, k = decompose_systematic_covariance(cov, stat_diag=D_stat)
        n = cov.shape[0]
        assert V.shape == (n, k)
        assert Lambda_d.shape == (k,)
        assert D_diag.shape == (n,)

    def test_max_woodbury_rank_gate(self, rng):
        """If effective rank > MAX_WOODBURY_RANK, still capped."""
        n = 50
        k_large = 30  # would truncate at MAX_WOODBURY_RANK
        Lambda = np.sort(np.exp(-np.arange(k_large)) * 100.0)[::-1]
        U, _ = np.linalg.qr(rng.standard_normal((n, k_large)))
        C_sys = U @ np.diag(Lambda) @ U.T
        D_stat = np.abs(rng.standard_normal(n)) + 0.01
        cov = C_sys + np.diag(D_stat)

        V, Lambda_d, _, k = decompose_systematic_covariance(cov, stat_diag=D_stat)
        effective_rank = np.sum(Lambda > 1e-15 * Lambda[0])
        k_expected = min(effective_rank, MAX_WOODBURY_RANK)
        assert k == k_expected

    def test_degenerate_no_positive_eigenvalues(self):
        """Edge: C_sys with no positive eigenvalues returns empty."""
        cov = np.eye(10) * 0.5  # purely diagonal
        V, Lambda_d, _, k = decompose_systematic_covariance(cov, stat_diag=np.ones(10) * 0.5)
        assert k == 0
        assert V.shape == (10, 0)
        assert len(Lambda_d) == 0

    def test_stat_diag_none_fallback(self, realistic_cov_data):
        """Fallback: stat_diag=None uses diag(cov) — less accurate but works."""
        cov, D_stat, _, _, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, k = decompose_systematic_covariance(cov, stat_diag=None)
        # Without stat_diag, C_sys = cov - diag(diag(cov)) = C_sys - diag(diag(C_sys))
        # This loses rank information, so k will be different
        assert D_diag is not None
        assert V.shape[0] == cov.shape[0]


class TestSequentialUpdate:
    """Integration tests for sequential M updates across HEALPix-like directions."""

    def test_sequential_updates_match_rebuild(self, realistic_cov_data, rng):
        """Build M for direction 0, then update for directions 1..N, compare against full rebuild."""
        cov, D_stat, _, _, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n_total = cov.shape[0]

        # Simulate 50 hemispheres with small membership changes (like HEALPix RING order)
        n_hemi = 120
        hemispheres = []
        base_idx = np.sort(rng.choice(n_total, n_hemi, replace=False))
        hemispheres.append(base_idx)

        for _ in range(49):
            prev = hemispheres[-1]
            # Flip ~10% of SNe between hemispheres
            n_flip = max(1, n_hemi // 10)
            flip_out = rng.choice(prev, n_flip, replace=False)
            flip_in = rng.choice(np.setdiff1d(np.arange(n_total), prev), n_flip, replace=False)
            new_h = np.sort(np.setdiff1d(prev, flip_out).tolist() + flip_in.tolist())
            hemispheres.append(new_h)

        # Sequential updates
        _, L_seq = _build_woodbury_M(V, L_inv, D_inv, hemispheres[0])
        for i in range(1, len(hemispheres)):
            prev_h = hemispheres[i - 1]
            curr_h = hemispheres[i]

            # Remove leaving SNe
            for j in set(prev_h) - set(curr_h):
                sigma_j = np.sqrt(D_diag[j])
                L_seq, info = _update_woodbury_M(L_seq, V[j, :], sigma_j, op="remove")
                if info != 0:
                    _, L_seq = _build_woodbury_M(V, L_inv, D_inv, curr_h)
                    break
            else:  # no break: no downdate failure
                for j in set(curr_h) - set(prev_h):
                    sigma_j = np.sqrt(D_diag[j])
                    L_seq = _update_woodbury_M(L_seq, V[j, :], sigma_j, op="add")[0]

            # Compare against full rebuild
            _, L_rebuild = _build_woodbury_M(V, L_inv, D_inv, curr_h)
            assert np.allclose(L_seq @ L_seq.T, L_rebuild @ L_rebuild.T, atol=1e-10), \
                f"Mismatch at hemisphere {i}"

    def test_sequential_preserves_h0_accuracy(self, realistic_cov_data, rng):
        """h₀ computed via sequentially-updated L_M matches full rebuild."""
        cov, D_stat, _, _, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n_total = cov.shape[0]
        n_hemi = 120

        # Build three consecutive hemispheres with small membership changes
        h0 = np.sort(rng.choice(n_total, n_hemi, replace=False))
        flip_out_1 = rng.choice(h0, 5, replace=False)
        flip_in_1 = rng.choice(np.setdiff1d(np.arange(n_total), h0), 5, replace=False)
        h1 = np.sort(np.setdiff1d(h0, flip_out_1).tolist() + flip_in_1.tolist())

        flip_out_2 = rng.choice(h1, 12, replace=False)
        flip_in_2 = rng.choice(np.setdiff1d(np.arange(n_total), h1), 12, replace=False)
        h2 = np.sort(np.setdiff1d(h1, flip_out_2).tolist() + flip_in_2.tolist())

        b0 = rng.standard_normal(len(h0))
        b1 = rng.standard_normal(len(h1))
        b2 = rng.standard_normal(len(h2))

        # Full rebuild references
        _, L0_ref = _build_woodbury_M(V, L_inv, D_inv, h0)
        _, L1_ref = _build_woodbury_M(V, L_inv, D_inv, h1)
        _, L2_ref = _build_woodbury_M(V, L_inv, D_inv, h2)
        h0_ref, _ = _woodbury_h0_solve(V, D_inv, h1, b1, L1_ref)
        h0_ref_2, _ = _woodbury_h0_solve(V, D_inv, h2, b2, L2_ref)

        # Sequential update: h0 → h1
        L_seq = L0_ref.copy()
        for j in set(h0) - set(h1):
            s = np.sqrt(D_diag[j])
            L_seq, info = _update_woodbury_M(L_seq, V[j, :], s, op="remove")
            assert info == 0
        for j in set(h1) - set(h0):
            s = np.sqrt(D_diag[j])
            L_seq = _update_woodbury_M(L_seq, V[j, :], s, op="add")[0]

        h0_upd, _ = _woodbury_h0_solve(V, D_inv, h1, b1, L_seq)
        assert abs(h0_upd - h0_ref) < 1e-12, \
            f"Update h0→h1 mismatch: {h0_upd:.14f} vs {h0_ref:.14f}"

        # Sequential update: h1 → h2
        for j in set(h1) - set(h2):
            s = np.sqrt(D_diag[j])
            L_seq, info = _update_woodbury_M(L_seq, V[j, :], s, op="remove")
            assert info == 0
        for j in set(h2) - set(h1):
            s = np.sqrt(D_diag[j])
            L_seq = _update_woodbury_M(L_seq, V[j, :], s, op="add")[0]

        h0_upd_2, _ = _woodbury_h0_solve(V, D_inv, h2, b2, L_seq)
        assert abs(h0_upd_2 - h0_ref_2) < 1e-12, \
            f"Update h1→h2 mismatch: {h0_upd_2:.14f} vs {h0_ref_2:.14f}"


class TestBuildWoodburyM:
    def test_matches_direct_formula(self, spd_matrix_5, chol_factor_5, rng):
        """M = Λ⁻¹ + V_H^T·D_HH⁻¹·V_H matches direct computation."""
        k = 5
        n_H = 10
        D_inv = np.abs(rng.standard_normal(n_H)) + 0.1
        L_inv = np.abs(rng.standard_normal(k)) + 0.1
        V = rng.standard_normal((n_H, k))
        H_indices = np.arange(n_H)

        M, L_M = _build_woodbury_M(V, L_inv, D_inv, H_indices)

        # Direct formula
        M_direct = np.diag(L_inv) + V.T @ np.diag(D_inv) @ V
        assert np.allclose(M, M_direct, atol=1e-12)
        assert np.allclose(L_M @ L_M.T, M, atol=1e-12)

    def test_cholesky_factor_is_valid(self, realistic_cov_data, rng):
        """L_M is a valid lower Cholesky factor: lower-triangular, positive diagonal."""
        cov, D_stat, _, _, _, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(cov, stat_diag=D_stat)

        n = cov.shape[0]
        H_idx = np.sort(rng.choice(n, 80, replace=False))
        M, L_M = _build_woodbury_M(V, 1.0 / Lambda_d, 1.0 / D_diag, H_idx)

        assert np.allclose(L_M, np.tril(L_M)), "Not lower triangular"
        assert np.all(np.diag(L_M) > 0), "Diagonal not positive"
        assert np.allclose(L_M @ L_M.T, M, atol=1e-12), "L·L^T != M"


# ===================================================================
# Woodbury chi² (GPU) — matches Cholesky chi²
# ===================================================================


@pytest.mark.skipif(not HAVE_CUPY, reason="CuPy not available")
class TestChi2WoodburyGpu:
    """_chi2_woodbury_gpu must match _chi2_cholesky_gpu to machine precision."""

    def test_chi2_matches_cholesky(self, realistic_cov_data, rng):
        """Known analytic value: Woodbury χ² = Cholesky χ² to < 1e-10.

        Given: realistic n=200 covariance with rank-15 C_sys, random hemisphere.
        When: χ² is computed via both Woodbury identity and full Cholesky.
        Then: max absolute difference across n_grid grid points < 1e-10.
        """
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(
            cov, stat_diag=D_stat
        )
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n = cov.shape[0]
        n_H = 80
        n_grid = 150

        H_idx = np.sort(rng.choice(n, n_H, replace=False))
        _, L_M = _build_woodbury_M(V, L_inv, D_inv, H_idx)

        # Build full hemisphere covariance on GPU: C_HH = D_HH + V_H·Λ·V_H^T
        V_H = V[H_idx, :]
        L_M_gpu = cp.asarray(L_M)
        V_H_gpu = cp.asarray(V_H)
        D_inv_H_gpu = cp.asarray(D_inv[H_idx])

        C_HH_gpu = cp.asarray(np.diag(D_diag[H_idx]) + V_H @ np.diag(Lambda_d) @ V_H.T)
        L_full_gpu = cp.linalg.cholesky(C_HH_gpu)

        # Random residuals on GPU
        resid_grid = cp.asarray(rng.standard_normal((n_grid, n_H)))

        chi2_chol = cp.asnumpy(_chi2_cholesky_gpu(resid_grid, L_full_gpu))
        chi2_w = cp.asnumpy(
            _chi2_woodbury_gpu(resid_grid, D_inv_H_gpu, V_H_gpu, L_M_gpu)
        )

        max_diff = float(np.max(np.abs(chi2_chol - chi2_w)))
        assert max_diff < 1e-10, f"|Δχ²|_max = {max_diff:.2e} ≥ 1e-10"

    def test_chi2_matches_cholesky_multiple_hemispheres(
        self, realistic_cov_data, rng
    ):
        """Invariant: Woodbury matches Cholesky for 10 random hemispheres.

        Given: the same covariance data.
        When: χ² is computed for 10 different random hemispheres.
        Then: max difference < 1e-10 for every hemisphere.
        """
        cov, D_stat, C_sys, Lambda_true, k_true, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(
            cov, stat_diag=D_stat
        )
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n = cov.shape[0]
        n_grid = 150
        max_max_diff = 0.0

        for _ in range(10):
            n_H = rng.integers(30, 120)
            H_idx = np.sort(rng.choice(n, n_H, replace=False))

            _, L_M = _build_woodbury_M(V, L_inv, D_inv, H_idx)
            V_H = V[H_idx, :]

            L_M_gpu = cp.asarray(L_M)
            V_H_gpu = cp.asarray(V_H)
            D_inv_H_gpu = cp.asarray(D_inv[H_idx])

            C_HH_gpu = cp.asarray(
                np.diag(D_diag[H_idx]) + V_H @ np.diag(Lambda_d) @ V_H.T
            )
            L_full_gpu = cp.linalg.cholesky(C_HH_gpu)

            resid_grid = cp.asarray(rng.standard_normal((n_grid, n_H)))

            chi2_chol = cp.asnumpy(_chi2_cholesky_gpu(resid_grid, L_full_gpu))
            chi2_w = cp.asnumpy(
                _chi2_woodbury_gpu(resid_grid, D_inv_H_gpu, V_H_gpu, L_M_gpu)
            )

            diff = float(np.max(np.abs(chi2_chol - chi2_w)))
            max_max_diff = max(max_max_diff, diff)
            assert diff < 1e-10, (
                f"|Δχ²|_max = {diff:.2e} ≥ 1e-10 for n_H={n_H}"
            )

    def test_chi2_positive(self, realistic_cov_data, rng):
        """Invariant: χ² is always non-negative.

        Given: any valid residuals and Woodbury decomposition.
        When: χ² is computed.
        Then: all values are >= 0.
        """
        cov, D_stat, _, _, _, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(
            cov, stat_diag=D_stat
        )
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n = cov.shape[0]
        n_H = 80
        H_idx = np.sort(rng.choice(n, n_H, replace=False))
        _, L_M = _build_woodbury_M(V, L_inv, D_inv, H_idx)

        V_H_gpu = cp.asarray(V[H_idx, :])
        D_inv_H_gpu = cp.asarray(D_inv[H_idx])
        L_M_gpu = cp.asarray(L_M)

        resid_grid = cp.asarray(rng.standard_normal((150, n_H)))
        chi2 = cp.asnumpy(
            _chi2_woodbury_gpu(resid_grid, D_inv_H_gpu, V_H_gpu, L_M_gpu)
        )
        assert np.all(chi2 >= 0), "χ² must be non-negative"

    def test_chi2_structured_residuals_stable(self, realistic_cov_data, rng):
        """Edge: residuals aligned to V columns (max correction, stress test).

        Given: residuals structured as V_H @ coeffs (aligned with systematic
        eigenvectors, maximizing the Woodbury correction term).
        When: χ² is computed via Woodbury.
        Then: χ² matches Cholesky to < 1e-10 and remains non-negative.
        """
        cov, D_stat, _, _, _, _ = realistic_cov_data
        V, Lambda_d, D_diag, _ = decompose_systematic_covariance(
            cov, stat_diag=D_stat
        )
        D_inv = 1.0 / D_diag
        L_inv = 1.0 / Lambda_d

        n = cov.shape[0]
        k = V.shape[1]
        n_H = 80
        n_grid = 50
        H_idx = np.sort(rng.choice(n, n_H, replace=False))
        _, L_M = _build_woodbury_M(V, L_inv, D_inv, H_idx)

        V_H = V[H_idx, :]
        V_H_gpu = cp.asarray(V_H)
        D_inv_H_gpu = cp.asarray(D_inv[H_idx])
        L_M_gpu = cp.asarray(L_M)

        C_HH_gpu = cp.asarray(
            np.diag(D_diag[H_idx]) + V_H @ np.diag(Lambda_d) @ V_H.T
        )
        L_full_gpu = cp.linalg.cholesky(C_HH_gpu)

        # Residuals aligned to V columns — maximizes the correction term
        coeffs = rng.standard_normal((n_grid, k)) * 5.0
        resid_np = V_H @ coeffs.T  # (n_H, n_grid)
        resid_grid = cp.asarray(resid_np.T)  # (n_grid, n_H)

        chi2_chol = cp.asnumpy(_chi2_cholesky_gpu(resid_grid, L_full_gpu))
        chi2_w = cp.asnumpy(
            _chi2_woodbury_gpu(resid_grid, D_inv_H_gpu, V_H_gpu, L_M_gpu)
        )

        max_diff = float(np.max(np.abs(chi2_chol - chi2_w)))
        assert max_diff < 1e-10, f"|Δχ²|_max = {max_diff:.2e} (structured residuals)"
        assert np.all(chi2_w >= 0), "χ² must be non-negative (structured residuals)"
