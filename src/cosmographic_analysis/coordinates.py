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

    vectors = [hp.pix2vec(nside, idx) for idx in pixel_indices]
    vectors = np.array(vectors)

    return vectors


def DecRa2Cartesian(dec, ra):
    """
    Convert Declination (dec) and Right Ascension (ra) to Cartesian coordinates*.
    
    * First convert declination and right ascension to regular spherical coordinates and then into Cartesian.
    
    Args:
        dec (float): Declination angle in degrees.
        ra (float): Right Ascension angle in degrees.
        
    Returns:
        numpy.ndarray: Array containing the Cartesian coordinates [x, y, z].
    """

    dec_rad = np.radians(dec)
    ra_rad = np.radians(ra)

    theta = np.pi/2 - dec_rad
    phi = ra_rad

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    return np.column_stack([x, y, z])



def IndexToDecRa(nside, index):
    """ 
    Convert a given healpix index to its corresponding right ascension and declination.
    
    Args:
        nside (int): The NSIDE parameter of the healpix map.
        index (int): The index of the pixel in the healpix map.
        
    Returns:
        tuple: A tuple containing the right ascension and declination angles in degrees.
    """
    theta, phi = hp.pix2ang(nside, index)
    dec = np.degrees(np.pi/2 - theta)
    ra = np.degrees(phi)
    return ra, dec


def DecRaToIndex(nside, dec, ra):
    """
    Convert declination and right ascension to healpix index.
    
    Args:
        nside (int): The NSIDE parameter of the healpix map.
        dec (float): Declination angle in degrees.
        ra (float): Right Ascension angle in degrees.
        
    Returns:
        int: The healpix index corresponding to the given declination and right ascension angles.
    """
    
    index = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))
