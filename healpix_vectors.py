import numpy as np
import healpy as hp


def get_healpix_vectors(nside: int) -> np.ndarray:
    """
    Get the unit vectors corresponding to each pixel of a Healpix map.

    Args:
        nside (int): The NSIDE parameter of the Healpix map.

    Returns:
        numpy.ndarray: An array of unit vectors representing the pixel locations.
    """

    # Generate sequential pixel indices
    npix = int(hp.nside2npix(nside) / 2)
    pixel_indices = np.arange(npix)

    # Convert pixel indices to unit vectors
    vectors = [hp.pix2vec(nside, idx) for idx in pixel_indices]
    vectors = np.array(vectors)

    return vectors