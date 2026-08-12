"""Test that optimizer strings route to correct functions."""
import numpy as np
import pytest
from unittest.mock import patch

# A dummy direction array so exec_map_numba doesn't crash on len()
_DIRS = np.zeros((3, 3), dtype=np.float64)

# GPU routing tests require CuPy; skip gracefully when unavailable
try:
    from cosmographic_analysis.hemispheric_comparison import HAVE_CUPY
except ImportError:
    HAVE_CUPY = False

_GPU_REASON = "GPU routing tests require CuPy"


def _make_datos(n_sne=3):
    """Create a minimal valid datos tuple for multi_hem_map_numba routing tests."""
    # (r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown, cov_numpy, model)
    r1 = np.zeros((n_sne, 9), dtype=np.float64)
    v1 = np.random.randn(n_sne, 3).astype(np.float64)
    hostyn = np.zeros(n_sne, dtype=np.int64)
    cov_numpy = np.eye(n_sne, dtype=np.float64)
    return (r1, v1, hostyn, None, 0.7, -0.5, 12, 0.1, 0.01, cov_numpy, 0)


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_numba_routes_woodbury():
    """When method='woodbury', _exec_map_numba_woodbury is called."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_woodbury') as mock:
        exec_map_numba(_DIRS, (None,) * 11, method='woodbury')
        mock.assert_called_once()


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_numba_routes_woodbury_cholesky():
    """When method='woodbury-cholesky', _exec_map_numba_woodbury is called with use_woodbury_chi2=False."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_woodbury') as mock:
        exec_map_numba(_DIRS, (None,) * 11, method='woodbury-cholesky')
        mock.assert_called_once()
        _, kwargs = mock.call_args
        assert kwargs.get('use_woodbury_chi2') is False, \
            "Expected woodbury-cholesky to route with use_woodbury_chi2=False"


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_numba_routes_grid():
    """When method='grid', _exec_map_numba_gpu_grid is called."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_gpu_grid') as mock:
        exec_map_numba(_DIRS, (None,) * 11, method='grid')
        mock.assert_called_once()


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_fixed_numba_routes_woodbury():
    """When method='woodbury', exec_map_fixed_numba calls _exec_map_numba_woodbury."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_fixed_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_woodbury') as mock:
        exec_map_fixed_numba(_DIRS, (None,) * 11, {}, method='woodbury')
        mock.assert_called_once()


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_fixed_numba_routes_woodbury_cholesky():
    """When method='woodbury-cholesky', exec_map_fixed_numba routes with use_woodbury_chi2=False."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_fixed_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_woodbury') as mock:
        exec_map_fixed_numba(_DIRS, (None,) * 11, {}, method='woodbury-cholesky')
        mock.assert_called_once()
        _, kwargs = mock.call_args
        assert kwargs.get('use_woodbury_chi2') is False, \
            "Expected woodbury-cholesky to route with use_woodbury_chi2=False"


@pytest.mark.skipif(not HAVE_CUPY, reason=_GPU_REASON)
def test_exec_map_fixed_numba_routes_grid():
    """When method='grid', exec_map_fixed_numba calls _exec_map_numba_gpu_grid."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_fixed_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_gpu_grid') as mock:
        exec_map_fixed_numba(_DIRS, (None,) * 11, {}, method='grid')
        mock.assert_called_once()


def test_multi_hem_map_numba_raises_unknown_method():
    """multi_hem_map_numba raises ValueError for unknown method (CPU path)."""
    from cosmographic_analysis.hemispheric_comparison import multi_hem_map_numba
    datos = _make_datos()
    with pytest.raises(ValueError, match="Unknown method"):
        multi_hem_map_numba(
            np.array([0.0, 0.0, 1.0]),
            datos,
            method='nonexistent',
        )
    # Cleanup
    del datos


def test_multi_hem_map_numba_accepts_woodbury_cholesky():
    """multi_hem_map_numba accepts woodbury-cholesky and falls back to golden section."""
    from cosmographic_analysis.hemispheric_comparison import multi_hem_map_numba
    datos = _make_datos()
    with patch('cosmographic_analysis.hemispheric_comparison._golden_fit') as mock_golden:
        mock_golden.return_value = (0.7, 0.1)
        # Should NOT raise ValueError("Unknown method") — any other exception
        # indicates a data setup issue, not a routing problem.
        exc = None
        try:
            multi_hem_map_numba(
                np.array([0.0, 0.0, 1.0]),
                datos,
                method='woodbury-cholesky',
            )
        except ValueError as e:
            exc = e
    assert exc is None or 'Unknown method' not in str(exc), \
        f"multi_hem_map_numba rejected 'woodbury-cholesky': {exc}"
    del datos


def test_multi_hem_map_fixed_numba_accepts_woodbury_cholesky():
    """multi_hem_map_fixed_numba accepts woodbury-cholesky and falls back to golden section."""
    from cosmographic_analysis.hemispheric_comparison import multi_hem_map_fixed_numba
    datos = _make_datos()
    precomputed = {0: {"up_indices": np.array([0, 1]), "down_indices": np.array([2]),
                        "inv_cov_up": np.eye(2), "inv_cov_down": np.eye(1)}}
    with patch('cosmographic_analysis.hemispheric_comparison._golden_fit') as mock_golden:
        mock_golden.return_value = (0.7, 0.1)
        exc = None
        try:
            multi_hem_map_fixed_numba(
                np.array([0.0, 0.0, 1.0]),
                0,
                datos,
                precomputed,
                method='woodbury-cholesky',
            )
        except ValueError as e:
            exc = e
    assert exc is None or 'Unknown method' not in str(exc), \
        f"multi_hem_map_fixed_numba rejected 'woodbury-cholesky': {exc}"
    del datos


def test_scipy_fitter_runs_and_joint_fit_finite():
    """The CPU scipy optimizer path actually executes and returns finite fits.

    Regression: chi2_func shadowed the global mu_model with its local name,
    raising UnboundLocalError whenever the scipy path ran (GPU routing
    silently falls back to golden, so GPU smokes never caught it).
    """
    from cosmographic_analysis.hemispheric_comparison import (
        _fit_hemisphere_scipy,
        multi_hem_map_numba,
    )
    n = 8
    r1 = np.zeros((n, 9), dtype=np.float64)
    r1[:, 2] = np.linspace(0.02, 0.09, n)  # z
    r1[:, 5] = 40.0  # mu_sh0es
    r1[:, 7] = 40.0  # muceph
    v1 = np.random.randn(n, 3).astype(np.float64)
    hostyn = np.zeros(n, dtype=np.int64)
    cov = np.eye(n, dtype=np.float64)
    datos = (r1, v1, hostyn, None, 0.7, -0.5, 12, 0.1, 0.01, cov, 0)

    # pipeline protocol: two conditional single-parameter fits
    res = multi_hem_map_numba(np.array([0.0, 0.0, 1.0]), datos, method='scipy')
    assert len(res) == 8
    assert all(np.isfinite(x) for x in res), res

    # joint 2-parameter fit (h0 and q0 free simultaneously) also runs
    h0, q0, h0_err, q0_err = _fit_hemisphere_scipy(
        r1[:, 2], r1[:, 7], r1[:, 5], hostyn, cov, 0.7, -0.5, (True, True)
    )
    assert np.isfinite(h0) and np.isfinite(q0)
    assert np.isfinite(h0_err) and np.isfinite(q0_err)


def test_exec_map_numba_scipy_never_routes_to_gpu():
    """method='scipy' must run the real scipy fit, never GPU golden fallback."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_gpu') as mock_gpu, \
         patch('cosmographic_analysis.hemispheric_comparison.multi_hem_map_numba') as mock_multi:
        mock_multi.return_value = (0.7, 0.7, 0.1, 0.1, -0.5, -0.5, 0.2, 0.2)
        exec_map_numba(_DIRS, (None,) * 11, method='scipy')
        mock_gpu.assert_not_called()
        assert mock_multi.call_count == len(_DIRS)


def test_exec_map_fixed_numba_scipy_never_routes_to_gpu():
    """Fixed-map scipy runs the real scipy fit, never GPU golden fallback."""
    from cosmographic_analysis.hemispheric_comparison import exec_map_fixed_numba
    with patch('cosmographic_analysis.hemispheric_comparison._exec_map_numba_gpu') as mock_gpu, \
         patch('cosmographic_analysis.hemispheric_comparison.multi_hem_map_fixed_numba') as mock_multi:
        mock_multi.return_value = (0.7, 0.7, 0.1, 0.1, -0.5, -0.5, 0.2, 0.2)
        exec_map_fixed_numba(_DIRS, (None,) * 11, {}, method='scipy')
        mock_gpu.assert_not_called()
        assert mock_multi.call_count == len(_DIRS)
