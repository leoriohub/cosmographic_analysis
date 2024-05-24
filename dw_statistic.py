from multiprocessing import Pool
from typing import Tuple
from scipy.optimize import minimize
from tqdm import tqdm
from linalg import cholesky
import pandas as pd
import numpy as np

# import distance modulus from cosmology.py
from cosmology import mu


def obtain_dw_statistic(healpix_dir: np.ndarray, datos: Tuple) -> float:
    
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
    newcovmatu = cov_mat.iloc[upi, upi].values
    cholesky_up = cholesky(newcovmatu)

    newcovmatd = cov_mat.iloc[downi, downi].values
    cholesky_down = cholesky(newcovmatd)
    #################

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


    return dw_statistic
