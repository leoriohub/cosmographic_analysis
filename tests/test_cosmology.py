import numpy as np
import pytest

from cosmographic_analysis.cosmology import dl, mu


H0_REF = 70.0
Q0_REF = -0.5
Z_REF = 0.01


def test_dl_at_zero_redshift_is_zero():
    assert dl(0.0, H0_REF, Q0_REF) == 0.0


def test_dl_scales_inversely_with_h0():
    d1 = dl(Z_REF, H0_REF, Q0_REF)
    d2 = dl(Z_REF, 2.0 * H0_REF, Q0_REF)
    assert d2 == pytest.approx(d1 / 2.0)


def test_dl_is_monotonic_in_z():
    z_grid = np.linspace(0.0, 0.1, 50)
    assert np.all(np.diff(dl(z_grid, H0_REF, Q0_REF)) >= 0.0)


def test_dl_is_not_linear_in_z():
    assert not np.isclose(dl(0.02, H0_REF, Q0_REF), 2.0 * dl(0.01, H0_REF, Q0_REF), rtol=1e-3)


def test_dl_known_value_at_reference_point():
    y = Z_REF / (Z_REF + 1.0)
    expected = (2997.92458 / H0_REF) * (y + (3.0 - Q0_REF) * y * y / 2.0)
    assert dl(Z_REF, H0_REF, Q0_REF) == pytest.approx(expected, rel=1e-12)


def test_mu_at_zero_redshift_is_neg_inf():
    assert mu(0.0, H0_REF, Q0_REF) == -np.inf


def test_mu_known_value_at_reference_point():
    y = Z_REF / (Z_REF + 1.0)
    d = (2997.92458 / H0_REF) * (y + (3.0 - Q0_REF) * y * y / 2.0)
    assert mu(Z_REF, H0_REF, Q0_REF) == pytest.approx(5.0 * np.log10(d) + 25.0, rel=1e-12)


def test_mu_shifts_by_minus_5_log10_2_when_h0_doubles():
    m1 = mu(Z_REF, H0_REF, Q0_REF)
    m2 = mu(Z_REF, 2.0 * H0_REF, Q0_REF)
    assert (m2 - m1) == pytest.approx(-5.0 * np.log10(2.0), rel=1e-12)
