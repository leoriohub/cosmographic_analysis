import numpy as np
import healpy as hp

def generate_map(nside: int, theta: np.ndarray, phi: np.ndarray, h0: np.ndarray, q0: np.ndarray) -> None:
    """
    Generate h0 and q0 maps based on the best fit values and the given healpix symmetry axes.

    Args:
        theta (np.ndarray): Array of theta angles values for each pixel.
        phi (np.ndarray): Array of phi angles values for each pixel.
        h0 (np.ndarray): Array of h0 values for each pixel.
        q0 (np.ndarray): Array of q0 values for each pixel.
        h0f (float): Fixed value for h0.
        q0f (float): Fixed value for q0.

    Returns:
        None
    """

    #Number of pixels for a given nside
    npix = hp.nside2npix(nside)
    #Indices of pixels
    indices = hp.ang2pix(nside, theta, phi)
    #Generate h0 and q0 maps
    h0map = np.zeros(npix, dtype=float)
    q0map = np.zeros(npix, dtype=float)
    np.add.at(h0map, indices, h0)
    np.add.at(q0map, indices, q0)

    return h0map, q0map
