"""Truncation-error diagnostic tests.

Thresholds grounded this session on uniform-z noiseless flat LCDM
(Om=0.3, H0=70, q0_true=-0.55):
- q0 fit bias over uniform z in [0.01, 0.3]: taylor2 -0.973, pade21 +0.029
- q0 fit bias over uniform z in [0.01, 0.1]: taylor2 -0.336, pade21 +0.004
- dL relative error at z=0.3: taylor2 -10.3%, pade11 +7.8%, pade21 +0.7%
- safe zup (|q0 bias| <= 0.05, uniform-z synthetic sample): pade21 >= 0.3,
  taylor2 < 0.2
"""
import numpy as np
import pytest

from cosmographic_analysis.truncation import (
    dl_lcdm_exact,
    dl_rel_error,
    max_safe_zup,
    mu_lcdm_exact,
    q0_fit_bias,
    scan_biases,
)


def _uniform_zz(zup, n=400, zdown=0.01):
    return np.linspace(zdown, zup, n)


def test_exact_lcdm_reference():
    """Exact flat LCDM dL(0.3) with H0=70, Om=0.3."""
    dL = dl_lcdm_exact(0.3)
    assert dL == pytest.approx(1552.72, rel=1e-4)
    assert mu_lcdm_exact(0.3) == pytest.approx(5 * np.log10(dL) + 25)


def test_q0_bias_known_values():
    """Grounded bias values for uniform-z windows."""
    assert q0_fit_bias(_uniform_zz(0.1), 0) == pytest.approx(-0.336, abs=0.02)
    assert q0_fit_bias(_uniform_zz(0.3), 0) == pytest.approx(-0.973, abs=0.02)
    assert q0_fit_bias(_uniform_zz(0.1), 2) == pytest.approx(0.004, abs=0.01)
    assert q0_fit_bias(_uniform_zz(0.3), 2) == pytest.approx(0.029, abs=0.01)


def test_bias_grows_with_window():
    """Truncation bias grows monotonically with zup (both directions)."""
    zz_all = _uniform_zz(0.3, n=300)
    b_t2_01 = q0_fit_bias(_uniform_zz(0.1), 0)
    b_t2_03 = q0_fit_bias(_uniform_zz(0.3), 0)
    b_p21_01 = q0_fit_bias(_uniform_zz(0.1), 2)
    b_p21_03 = q0_fit_bias(_uniform_zz(0.3), 2)
    assert b_t2_03 < b_t2_01 < 0  # taylor2 underestimates, worse with zup
    assert b_p21_03 > b_p21_01 > 0  # pade21 overestimates, worse with zup


def test_dl_rel_error_signs_at_z03():
    """At z=0.3: taylor2 underestimates by >5%, pade21 within 2%, pade11 ~8%."""
    assert dl_rel_error(0.3, 0) < -0.05
    assert abs(dl_rel_error(0.3, 2)) < 0.02
    assert 0.05 < dl_rel_error(0.3, 1) < 0.10


def test_safe_zup_ordering():
    """With |q0 bias| <= 0.05, pade21 is safe at 0.3, taylor2 is not at 0.2."""
    zz_all = _uniform_zz(0.6, n=600)
    zup_p21, bias_p21 = max_safe_zup(zz_all, 0.01, 2, tolerance=0.05, zup_max=0.6)
    zup_t2, bias_t2 = max_safe_zup(zz_all, 0.01, 0, tolerance=0.05, zup_max=0.6)
    assert zup_p21 >= 0.3
    assert zup_t2 < 0.2
    assert abs(bias_p21) <= 0.05
    assert abs(bias_t2) <= 0.05


def test_scan_biases_structure():
    """scan_biases returns one row per model x zup with anchors and deltas."""
    zz_all = _uniform_zz(0.3, n=300)
    rows = scan_biases(zz_all, 0.01, [0.1, 0.2, 0.3], models=(0, 2))
    assert len(rows) == 6
    for r in rows:
        assert set(r) >= {"model", "zup", "n_sne", "q0_bias", "h0_bias", "dl_err"}
        if r["zup"] == 0.1:
            assert r["delta_q0"] == 0.0
    p21_01 = [r for r in rows if r["model"] == 2 and r["zup"] == 0.1][0]
    p21_03 = [r for r in rows if r["model"] == 2 and r["zup"] == 0.3][0]
    assert p21_03["delta_q0"] == pytest.approx(
        p21_03["q0_bias"] - p21_01["q0_bias"], abs=1e-9
    )
