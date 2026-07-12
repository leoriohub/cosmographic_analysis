import numpy as np
import pytest
import healpy as hp
from cosmographic_analysis.coordinates import (
    DecRa2Cartesian,
    DecRaToIndex,
    IndexToDecRa,
    get_healpix_vectors,
)


def test_north_pole_via_dec_ra_to_cartesian():
    result = DecRa2Cartesian(90.0, 37.0)
    assert np.isclose(result[0, 0], 0.0)
    assert np.isclose(result[0, 1], 0.0)
    assert np.isclose(result[0, 2], 1.0)


def test_south_pole_via_dec_ra_to_cartesian():
    result = DecRa2Cartesian(-90.0, 0.0)
    assert np.isclose(result[0, 0], 0.0)
    assert np.isclose(result[0, 1], 0.0)
    assert np.isclose(result[0, 2], -1.0)


def test_equator_ra_zero_points_east():
    result = DecRa2Cartesian(0.0, 0.0)
    assert np.allclose(result[0], [1.0, 0.0, 0.0])


def test_dec_ra_to_cartesian_handles_arrays():
    result = DecRa2Cartesian(np.array([0.0, 0.0]), np.array([0.0, 90.0]))
    assert result.shape == (2, 3)
    assert np.allclose(result[0], [1.0, 0.0, 0.0])
    assert np.allclose(result[1], [0.0, 1.0, 0.0])


def test_get_healpix_vectors_are_unit():
    nside = 4
    vectors = get_healpix_vectors(nside)
    expected_npix = int(hp.nside2npix(nside) / 2)
    assert vectors.shape == (expected_npix, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-10)


def test_index_dec_ra_roundtrip():
    nside = 8
    index = 384
    ra, dec = IndexToDecRa(nside, index)
    recovered = DecRaToIndex(nside, dec, ra)
    assert recovered == index
