"""Regression tests for sky-map direction-to-pixel placement."""
import numpy as np
import healpy as hp

from cosmographic_analysis.coordinates import DecRa2Cartesian
from cosmographic_analysis.maps import dirs_to_theta_phi, generate_map


def test_dirs_to_theta_phi_cardinal_directions():
    theta, phi = dirs_to_theta_phi(
        np.array([
            [1.0, 0.0, 0.0],   # RA=0,   Dec=0
            [0.0, 1.0, 0.0],   # RA=90,  Dec=0
            [-1.0, 0.0, 0.0],  # RA=180, Dec=0
            [0.0, -1.0, 0.0],  # RA=270, Dec=0
            [0.0, 0.0, 1.0],   # north pole
            [0.0, 0.0, -1.0],  # south pole
        ])
    )
    assert np.allclose(theta, [np.pi / 2] * 4 + [0.0, np.pi])
    assert np.allclose(phi, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 0.0, 0.0])


def test_dirs_to_theta_phi_wraps_phi_to_2pi():
    # The y < 0 branch of arctan2 returns negative angles; helper wraps to [0, 2*pi).
    theta, phi = dirs_to_theta_phi(np.array([[0.0, -1.0, 0.0]]))
    assert 0.0 <= phi[0] < 2 * np.pi
    assert np.isclose(phi[0], 3 * np.pi / 2)


def test_generate_map_places_direction_at_own_pixel():
    # Regression for the longitude mirror: phi = 180° - arctan2(y,x) painted
    # each direction at the mirrored RA. The max-q0 direction at nside=8 is
    # (RA, Dec) = (67.5°, 78.28°); under the bug it landed at RA 112.5°.
    nside = 8
    v = DecRa2Cartesian(78.28, 67.5)[0]
    theta, phi = dirs_to_theta_phi(v[np.newaxis, :])
    h0map, q0map = generate_map(nside, theta, phi, [0.0], [1.0])
    pix = hp.ang2pix(nside, theta[0], phi[0])
    assert q0map[pix] == 1.0
    ra_pix = np.degrees(hp.pix2ang(nside, pix)[1]) % 360
    assert np.isclose(ra_pix, 67.5, atol=1e-6), f"painted at RA {ra_pix}, expected 67.5"
