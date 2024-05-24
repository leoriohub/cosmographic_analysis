from typing import Tuple
import numpy as np

# import distance modulus from cosmology.py
from cosmology import mu


def obtain_dw_statistic(healpix_dir: np.ndarray, datos: Tuple) -> float:
    """
    Calculate the dw statistic for a given healpix direction.

    Args:
        healpix_dir (np.ndarray):  3D position vector for a given healpix pixel.
        datos (Tuple): Tuple containing data arrays.
   """

    v1, r1, hostyn, cov_mat, q0f, h0f = datos

    cos_angle = np.dot(v1, healpix_dir)

    up_indices = cos_angle >= 0
    down_indices = ~up_indices

    up_data = r1[up_indices]
    down_data = r1[down_indices]

    up_hostyn = hostyn[up_indices]
    down_hostyn = hostyn[down_indices]

    up_redshift = up_data[:, 2].astype(float)
    down_redshift = down_data[:, 2].astype(float)

    up_hostyn = up_hostyn.astype(bool)
    down_hostyn = down_hostyn.astype(bool)

    ############ COV MATRIX ############
    ############ TO BE IMPLEMENTED #####

    # cholesky_up = cholesky(newcovmatu)

    # cholesky_down = cholesky(newcovmatd)

    ####################################

    mu_sh0es_up = up_data[:, 5]
    muceph_up = up_data[:, 7]
    mu_sh0es_down = down_data[:, 5]
    muceph_down = down_data[:, 7]

    mu_model_up = np.array([mu(zi, h0f, q0f) for zi in up_redshift])
    mu_model_down = np.array([mu(zi, h0f, q0f) for zi in down_redshift])

    resid_up = np.zeros(len(up_data))
    resid_up[up_hostyn == 1] = mu_sh0es_up[up_hostyn == 1] - muceph_up[up_hostyn == 1]
    resid_up[up_hostyn == 0] = mu_sh0es_up[up_hostyn == 0] - mu_model_up[up_hostyn == 0]

    resid_down = np.zeros(len(down_data))
    resid_down[down_hostyn == 1] = mu_sh0es_down[down_hostyn == 1] - muceph_down[down_hostyn == 1]
    resid_down[down_hostyn == 0] = mu_sh0es_down[down_hostyn == 0] - mu_model_down[down_hostyn == 0]

    diff_up = np.diff(resid_up)
    diff_down = np.diff(resid_down)

    dw_up = np.sum(diff_up**2)/np.sum(resid_up**2)
    dw_down = np.sum(diff_down**2)/np.sum(resid_down**2)

    return dw_up, dw_down
