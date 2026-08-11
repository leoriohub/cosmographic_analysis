import numpy as np
from numba import njit


@njit(cache=True)
def dl(z, h0, q0):
    y = z / (z + 1.0)
    return (2997.92458 / h0) * (y + (3.0 - q0) * y * y / 2.0)


@njit(cache=True)
def mu(z, h0, q0):
    return 5.0 * np.log10(dl(z, h0, q0)) + 25.0


MODEL_TAYLOR2 = 0
MODEL_PADE11 = 1
MODEL_PADE21 = 2
MODEL_CODES = {"taylor2": MODEL_TAYLOR2, "pade11": MODEL_PADE11, "pade21": MODEL_PADE21}


@njit(cache=True)
def dl_model(z, h0, q0, model):
    y = z / (z + 1.0)
    if model == 1:
        inner = y / (1.0 - (3.0 - q0) * y / 2.0)
    elif model == 2:
        f2 = (3.0 - q0) / 2.0
        f3 = (10.0 - 5.0 * q0 + 3.0 * q0 * q0) / 6.0  # j0 fixed at 1.0 (LambdaCDM)
        inner = y * (1.0 + (f2 - f3 / f2) * y) / (1.0 - (f3 / f2) * y)
    else:
        inner = y + (3.0 - q0) * y * y / 2.0
    return (2997.92458 / h0) * inner


@njit(cache=True)
def mu_model(z, h0, q0, model):
    return 5.0 * np.log10(dl_model(z, h0, q0, model)) + 25.0
