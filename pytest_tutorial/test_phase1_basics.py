"""Phase 1: Zero-setup testing with plain assert statements.

Tests the two simplest pure functions in the codebase:
  - cosmology.dl(z, h0, q0)   → luminosity distance
  - cosmology.mu(z, h0, q0)   → distance modulus

These are pure functions: same inputs → same outputs, no side effects.
"""

import numpy as np
from cosmographic_analysis.cosmology import dl, mu


def test_dl_at_zero_redshift():
    """At z=0, luminosity distance should be exactly 0."""
    result = dl(0.0, 0.73, -0.57)
    assert result == 0.0


def test_dl_scales_with_inverse_h0():
    """Double h0 → half the distance (for fixed z, q0)."""
    z = 0.05
    q0 = -0.574
    d1 = dl(z, 0.5, q0)
    d2 = dl(z, 1.0, q0)
    assert abs(d2 - d1 / 2.0) < 1e-10


def test_dl_positive_for_positive_z():
    """For any positive z, luminosity distance must be positive."""
    for z in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]:
        assert dl(z, 0.73, -0.57) > 0


def test_mu_monotonic_with_z():
    """Distance modulus should increase with redshift."""
    h0, q0 = 0.7304, -0.574
    z_vals = np.linspace(0.01, 0.1, 10)
    mu_vals = [mu(z, h0, q0) for z in z_vals]
    assert all(mu_vals[i] < mu_vals[i + 1] for i in range(len(mu_vals) - 1))


# --- Intentional failure to showcase pytest error reporting ---

def test_dl_known_value_INTENTIONAL_FAILURE():
    """This test FAILS on purpose so you can see pytest's diff output.

    dl(z=0.05, h0=0.73, q0=-0.574) computes to ~204.38,
    but we assert 999.0 to trigger the failure.
    """
    z, h0, q0 = 0.05, 0.73, -0.574
    actual = dl(z, h0, q0)
    expected = 999.0
    assert actual == expected, f"dl({z}, {h0}, {q0}) should be {expected}"
