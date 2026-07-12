import numpy as np
import pytest
from scipy.stats import norm

from cosmographic_analysis.statistics import (
    fit_gaussian,
    map_statistics,
    mc_statistics,
)


def test_fit_gaussian_recovers_known_mean_and_std():
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=10000)
    x_gaussian, y_gaussian = fit_gaussian(data)
    peak_idx = int(np.argmax(y_gaussian))
    assert x_gaussian[peak_idx] == pytest.approx(5.0, abs=0.1)
    idx_at_5 = int(np.argmin(np.abs(x_gaussian - 5.0)))
    expected_pdf_at_5 = norm.pdf(5.0, loc=5.0, scale=2.0)
    assert y_gaussian[idx_at_5] == pytest.approx(expected_pdf_at_5, rel=0.05)


def test_map_statistics_returns_expected_keys_and_values():
    arr = np.array([1.0, 2.0, 3.0])
    result = map_statistics(arr, arr, arr, arr)
    assert set(result.keys()) == {"h0_mean", "q0_mean", "h0_std_dev", "q0_std_dev"}
    assert result["h0_mean"] == pytest.approx(2.0)
    assert result["q0_mean"] == pytest.approx(2.0)
    assert result["h0_std_dev"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result["q0_std_dev"] == pytest.approx(np.sqrt(2.0 / 3.0))


def test_mc_statistics_data_at_median_gives_around_fifty_pct():
    rng = np.random.default_rng(0)
    mc = rng.normal(0.0, 1.0, 1000)
    p_values = mc_statistics(
        maximum_anisotropy_data=np.array([0.0, 0.0]),
        maximum_anisotropy_mc=np.array([mc, mc, mc, mc]),
    )
    assert p_values[2] == pytest.approx(50.0, abs=5.0)
