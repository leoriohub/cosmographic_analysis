"""Pantheon+ data loading utilities.

Replicates the exact data loading logic from main.ipynb cell 4.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from cosmographic_analysis.coordinates import DecRa2Cartesian


def load_pantheon_data(
    lcparam_path: str,
    cov_path: str,
    zup: float,
    zdown: float,
) -> Tuple:
    """Load and filter Pantheon+ supernova data.

    Returns: (zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z)
    """
    lcparam = np.loadtxt(
        lcparam_path, skiprows=1, usecols=(2, 8, 9, 10, 11, 26, 27, 12, 13)
    )
    lcparam_sys = np.loadtxt(cov_path, skiprows=1)

    ind = np.where((lcparam[:, 0] < zup) & (lcparam[:, 0] > zdown))[0]

    zz = lcparam[ind, 0]
    mz = lcparam[ind, 1]
    sigmz = lcparam[ind, 2]
    muz = lcparam[ind, 3]
    sigmuz = lcparam[ind, 4]
    ra = lcparam[ind, 5]
    dec = lcparam[ind, 6]
    muceph = lcparam[ind, 7]
    hostyn = lcparam[ind, 8]

    cov_z = lcparam_sys.reshape(1701, 1701)
    cov_z = cov_z[np.ix_(ind, ind)]
    inv_cov_z = np.linalg.inv(cov_z)
    cov_mat = pd.DataFrame(cov_z, columns=range(len(zz)))

    return zz, mz, sigmz, muz, sigmuz, ra, dec, muceph, hostyn, cov_mat, inv_cov_z


def build_datos_tuple(
    ra: np.ndarray,
    dec: np.ndarray,
    zz: np.ndarray,
    mz: np.ndarray,
    sigmz: np.ndarray,
    muz: np.ndarray,
    sigmuz: np.ndarray,
    muceph: np.ndarray,
    hostyn: np.ndarray,
    cov_mat: pd.DataFrame,
    h0f: float,
    q0f: float,
    pts: int,
    zup: float,
    zdown: float,
) -> Tuple:
    """Build the 'datos' tuple used by hemispheric comparison functions.

    Replicates main.ipynb data packaging logic.
    """
    r1 = np.column_stack([ra, dec, zz, mz, sigmz, muz, sigmuz, muceph, hostyn])
    v1 = DecRa2Cartesian(dec, ra)
    datos = (r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown)
    return datos
