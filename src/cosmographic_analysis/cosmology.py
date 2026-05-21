import numpy as np
from numba import njit


@njit(cache=True)
def dl(z, h0, q0):
    y = z / (z + 1.0)
    return (2997.92458 / h0) * (y + (3.0 - q0) * y * y / 2.0)


@njit(cache=True)
def mu(z, h0, q0):
    return 5.0 * np.log10(dl(z, h0, q0)) + 25.0
