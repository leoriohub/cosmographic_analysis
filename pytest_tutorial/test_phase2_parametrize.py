"""Phase 2: Parameterized tests — one function, many inputs.

Tests coordinates.DecRa2Cartesian(dec, ra), which converts
celestial (Dec, RA) in degrees to (x, y, z) unit vectors.

Without parametrize you'd write 6 nearly identical test functions.
With parametrize it's one function + a table of (dec, ra, expected_x, expected_y, expected_z).
"""

import numpy as np
import pytest
from cosmographic_analysis.coordinates import DecRa2Cartesian


@pytest.mark.parametrize("dec, ra, exp_x, exp_y, exp_z", [
    # North pole
    (90.0,    0.0,  0.0,  0.0,  1.0),
    # South pole
    (-90.0,   0.0,  0.0,  0.0, -1.0),
    # Equator, RA=0   (vernal equinox direction)
    (0.0,     0.0,  1.0,  0.0,  0.0),
    # Equator, RA=90
    (0.0,    90.0,  0.0,  1.0,  0.0),
    # Equator, RA=180
    (0.0,   180.0, -1.0,  0.0,  0.0),
    # Equator, RA=-90  (negative angle, edge case)
    (0.0,   -90.0,  0.0, -1.0,  0.0),
])
def test_DecRa2Cartesian_known_points(dec, ra, exp_x, exp_y, exp_z):
    """Check that known (dec, ra) map to the expected unit vector."""
    result = DecRa2Cartesian(np.array([dec]), np.array([ra]))
    x, y, z = result[0]
    assert np.isclose(x, exp_x, atol=1e-10)
    assert np.isclose(y, exp_y, atol=1e-10)
    assert np.isclose(z, exp_z, atol=1e-10)


@pytest.mark.parametrize("dec, ra", [
    (0.0, 0.0),
    (45.0, 30.0),
    (-30.0, 120.0),
])
def test_DecRa2Cartesian_is_unit_vector(dec, ra):
    """For any input, the output vector should have length ≈ 1."""
    result = DecRa2Cartesian(np.array([dec]), np.array([ra]))
    norm = np.linalg.norm(result[0])
    assert np.isclose(norm, 1.0, atol=1e-10)
