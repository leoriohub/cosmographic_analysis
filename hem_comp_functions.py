from multiprocessing import Pool
from typing import Tuple
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
import numpy as np
# import distance modulus from cosmology.py
from cosmology import mu


def hem_h0(healpix_dirs: np.ndarray, datos: Tuple, save = None) -> Tuple[float, float, float, float]:
    """
    Fits h0 values for the given healpix directions North and South hemispheres. 
    
    Args:
        healpix_dirs (np.ndarray): Array of healpix directions.
        datos (Tuple): Tuple containing data arrays (v1, r1, hostyn, cov_mat, h0f, q0f).

        options:
            filename (str, optional): Filename to save results. Defaults to None.
        
    Returns:
        Tuple[float, float, float, float]containing h0u and h0d.
    """

    r1 = datos[0]
    v1 = datos[1]
    hostyn = datos[2]
    cov_mat = datos[3]
    q0f = datos[5]
    
    dot_products = np.dot(v1, healpix_dirs)

    mask_up = dot_products >= 0
    mask_down = ~mask_up

    up = r1[mask_up]
    down = r1[mask_down]

    upi = np.where(mask_up)[0]
    downi = np.where(mask_down)[0]

    hostyn_up = hostyn[mask_up]
    hostyn_down = hostyn[mask_down]

    z_up = up[:, 2]
    z_down = down[:, 2]

    ############ COV MATRIX ############
    newcovmatu = cov_mat.iloc[upi, upi].values
    inv_newcovu = np.linalg.inv(newcovmatu)

    newcovmatd = cov_mat.iloc[downi, downi].values
    inv_newcovd = np.linalg.inv(newcovmatd)
    #################

    mu_sh0es_up = up[:, 5]
    muceph_up = up[:, 7]
    mu_sh0es_down = down[:, 5]
    muceph_down = down[:, 7]

    def chi2uh0(theta):
        """
        Calculate chi2 for up hemisphere.

        Args:
            theta (list): List containing a single element h0.

        Returns:
            float: Chi2 value.
        """
        h0 = theta[0]  # theta is a 1-element array, extract the value
        mu_model_up = np.array([mu(zi, h0, q0f) for zi in z_up])

        resid_up = np.zeros(len(up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]

        Ar = np.dot(resid_up, np.dot(inv_newcovu, resid_up))
        return Ar

    chi2umin = minimize(chi2uh0, [0.7], method='L-BFGS-B')
    h0u = chi2umin.x[0]
    h0u_err = np.sqrt(chi2umin.hess_inv([1])[0])  # error is defined as the sqrt of the diagonal elements of the inverse Hessian matrix

    def chi2dh0(theta):
        """
        Calculate chi2 for down values.

        Args:
            theta (list): List containing a single element h0.

        Returns:
            float: Chi2 value.
        """
        h0 = theta[0]  # theta is a 1-element array, extract the value
        mu_model_down = np.array([mu(zi, h0, q0f) for zi in z_down])

        resid_down = np.zeros(len(down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down ==0] - mu_model_down[hostyn_down == 0]

        Ar = np.dot(resid_down, np.dot(inv_newcovd, resid_down))
        return Ar

    chi2dmin = minimize(chi2dh0, [0.7], method='L-BFGS-B')
    h0d = chi2dmin.x[0]
    h0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])  # error is defined as the sqrt of the diagonal elements of the inverse Hessian matrix
    return h0u, h0d, h0u_err, h0d_err  #


# Hemispheric comparison implementation for q0.

def hem_q0(healpix_dirs: np.ndarray, datos: Tuple, save = None) -> Tuple[float, float, float, float]:
    """
    Fits q0 values for the given healpix directions separating data into up and down hemispheres.

    Args:
        healpix_dirs (np.ndarray): Array of healpix directions.
        datos (Tuple): Tuple containing data arrays (v1, r1, hostyn, cov_mat, h0f, q0f).

    Returns:
        Tuple[float, float,float, float]containing q0u and q0d values and their errors.
    """
    r1 = datos[0]
    v1 = datos[1]
    hostyn = datos[2]
    cov_mat = datos[3]
    h0f = datos[4]

    # Calculate dot products
    dot_products = np.dot(v1, healpix_dirs)

    # Create masks for dot products
    mask_up = dot_products >= 0

    # Split r1 values based on masks
    datos_sne_up = r1[mask_up]
    datos_sne_down = r1[~mask_up]

    # Using numpy's boolean indexing directly instead of np.where
    upi = np.where(mask_up)[0]
    downi = np.where(~mask_up)[0]

    # Split hostyn values based on masks
    hostyn_up = hostyn[mask_up]
    hostyn_down = hostyn[~mask_up]

    # Split z, muceph, and mu_sh0es values
    z_up, z_down = datos_sne_up[:, 2], datos_sne_down[:, 2]
    muceph_up, muceph_down = datos_sne_up[:, 7], datos_sne_down[:, 7]
    mu_sh0es_up, mu_sh0es_down = datos_sne_up[:, 5], datos_sne_down[:, 5]

    ############ COV MATRIX ############
    newcovmatu = cov_mat.iloc[upi, upi].values  # Using iloc for indexing
    inv_newcovu = np.linalg.inv(newcovmatu)

    newcovmatd = cov_mat.iloc[downi, downi].values  # Using iloc for indexing
    inv_newcovd = np.linalg.inv(newcovmatd)
    #################

    def chi2uq0(theta):
        """
        Calculate chi2 for up hemisphere values.

        Args:
            theta (array-like): Array containing q0 value.

        Returns:
            float: Ar value.
        """
        q0 = theta[0]  # theta is a 1-element array, extract the value
        mu_model_up = np.array([mu(zi, h0f, q0) for zi in z_up])

        resid_up = np.zeros(len(datos_sne_up))
        resid_up[hostyn_up == 1] = muceph_up[hostyn_up == 1] - mu_model_up[hostyn_up == 1]
        resid_up[hostyn_up == 0] = mu_sh0es_up[hostyn_up == 0] - mu_model_up[hostyn_up == 0]

        Ar = np.dot(resid_up, np.dot(inv_newcovu, resid_up))
        return Ar

    # Minimize chi2uq0 function to find q0u
    chi2umin = minimize(chi2uq0, [-0.5], method='L-BFGS-B')
    q0u = chi2umin.x[0]
    q0u_err = np.sqrt(chi2umin.hess_inv([1])[0])  # error is defined as the sqrt of the diagonal elements of the inverse Hessian matrix

    def chi2dq0(theta):
        """

        Calculate the chi2 value for down q0 hemisphere.

        Args:
            theta (array-like): Array containing q0 value.

        Returns:
            float: Ar value.
        """
        q0 = theta[0]  # theta is a 1-element array, extract the value
        mu_model_down = np.array([mu(zi, h0f, q0) for zi in z_down])

        resid_down = np.zeros(len(datos_sne_down))
        resid_down[hostyn_down == 1] = muceph_down[hostyn_down == 1] - mu_model_down[hostyn_down == 1]
        resid_down[hostyn_down == 0] = mu_sh0es_down[hostyn_down == 0] - mu_model_down[hostyn_down == 0]

        Ar = np.dot(resid_down, np.dot(inv_newcovd, resid_down))
        return Ar

    chi2dmin = minimize(chi2dq0, [-0.5], method='L-BFGS-B')
    q0d = chi2dmin.x[0]
    q0d_err = np.sqrt(chi2dmin.hess_inv([1])[0])  # error is defined as the sqrt of the diagonal elements of the inverse Hessian matrix

    return q0u, q0d, q0u_err, q0d_err


# Parallel mapping implementation.

# Healpix_dirs is a list of directions which represent each pixel in the healpix pixelation scheme.

def multi_hem_map(healpix_vec: np.ndarray, datos: Tuple, save = None):
    """
    Calculate multiple hemispherical maps.

    Args:
        healpix_vec (np.ndarray):  3D position vector for a given Healpix pixel.
        datos (Tuple): Tuple of 6 elements, which are arrays of Sne positions, Sne data, binary array of cepheid host entries, covariance matrix, fiducial h0 value, and q0 value.


    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of four numpy arrays representing the calculated hemispherical maps.
    """

    datos = datos

    h0u_aux, h0d_aux, h0u_err_aux, h0d_err_aux = hem_h0(healpix_vec, datos)
    q0u_aux, q0d_aux, q0u_err_aux, q0d_err_aux = hem_q0(healpix_vec, datos)

    return h0u_aux, h0d_aux, h0u_err_aux, h0d_err_aux, q0u_aux, q0d_aux, q0u_err_aux, q0d_err_aux


# Exec_map is a function that receives a list of healpix_dirs and maps the hemispheric comparison function to each healpix_dir in parallel.

def exec_map(healpix_dirs: np.ndarray, datos: Tuple, save = None):
    """
    Execute the calculation of multiple hemispherical maps using the function multi_hem_map.

    Args:
        healpix_dirs (np.ndarray): Array of healpix directions.
        datos (Tuple): Tuple containing data arrays (v1, r1, hostyn, cov_mat, h0f, q0f).
        

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple of four numpy arrays containing the best fitted parameters for each hemispher.
    """

    r1, v1, hostyn, cov_mat, h0f, q0f, pts, zup, zdown = datos
    
    args_list = [(healpix_dir, datos)
                 for healpix_dir in healpix_dirs]

    with Pool() as pool:
        results_map = list(
            tqdm(pool.starmap(multi_hem_map, args_list), total=len(healpix_dirs)))

    h0u, h0d, h0u_err, h0d_err, q0u, q0d, q0u_err, q0d_err = zip(*results_map)

    results_h0 = (h0u, h0d, h0u_err, h0d_err)
    results_q0 = (q0u, q0d, q0u_err, q0d_err)

    if save is not None:
        
        header_map = f'This is the data for the Hubble and q0 maps for the following parameters: \n {pts} points, q0f= {q0f}, h0f= {h0f}, zup= {zup}, zdown= {zdown}\n\n h0u h0u_err h0d h0d_err q0u q0u_err q0d q0d_err'

        filename_map = f'compilations/[NEW][MAP][SH0ES_CALIB](pts={pts}_hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt'
        save_data_map = np.column_stack([h0u, h0u_err, h0d, h0d_err, q0u, q0u_err, q0d, q0d_err])


        # Uncomment to save the map

        np.savetxt(filename_map, save_data_map, header=header_map)
    
    return results_h0, results_q0
    
def load_map_old(file_path):
    data = np.loadtxt(file_path, usecols=(0, 2), skiprows=4)
    h0 = data[:, 0]
    q0 = data[:, 1]
    return h0, q0