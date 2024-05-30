import numpy as np

def dl(z: float, h0: float, q0: float) -> float:
    '''Calculate the cosmographic luminous distance parametrized for low z divergences.
    
    Args:
        z (float): The redshift value.
        h0 (float): The Hubble constant.
        q0 (float): The deceleration parameter.
    
    Returns:
        float: The luminous distance.
    '''
    y = z / (z + 1.0)
    return (2997.92458 / h0) * (y + (3.0 - q0) * np.power(y, 2) / 2.0)

def mu(z: float, h0: float, q0: float) -> float:
    '''Calculate the distance modulus prediction.
    
    Args:
        z (float): The redshift value.
        h0 (float): The Hubble constant.
        q0 (float): The deceleration parameter.
    
    Returns:
        float: The distance modulus.
    '''
    return 5.0 * np.log10(dl(z, h0, q0)) + 25.0