from typing import Tuple
import numpy as np

# import distance modulus from cosmology.py
from cosmology import mu

def hemispheric_dw(healpix_dir: np.ndarray, datos: Tuple):
    """
    Calculate the dw statistic for a given healpix direction.

    Args:
        healpix_dir (np.ndarray): 3D position vector for a given healpix pixel.
        datos (Tuple): Tuple containing data arrays.

    Returns:
        Tuple[float, float]: The dw statistics for up and down directions.
    """

    r1 = datos[0]
    v1 = datos[1]
    hostyn = datos[2]
    h0f = datos[4]
    q0f = datos[5]
    

    cos_angle = np.dot(v1, healpix_dir)

    # Determine north and south indices based on the cosine 
    up_indices = cos_angle >= 0
    down_indices = ~up_indices

    # Filter data based on up and down hemispheres
    up_data = r1[up_indices]
    down_data = r1[down_indices]

    up_hostyn = hostyn[up_indices]
    down_hostyn = hostyn[down_indices]

    # Extract redshifts and ensure they are floats
    up_redshift = up_data[:, 2].astype(float)
    down_redshift = down_data[:, 2].astype(float)

    up_hostyn = up_hostyn.astype(bool)
    down_hostyn = down_hostyn.astype(bool)

    # Calculate the mu values
    mu_sh0es_up = up_data[:, 5]
    muceph_up = up_data[:, 7]
    mu_sh0es_down = down_data[:, 5]
    muceph_down = down_data[:, 7]

    mu_model_up = np.array([mu(zi, h0f, q0f) for zi in up_redshift])
    mu_model_down = np.array([mu(zi, h0f, q0f) for zi in down_redshift])

    # Calculate residuals
    resid_up = np.zeros(len(up_data))

    resid_up[up_hostyn] =  muceph_up[up_hostyn] - mu_model_up[up_hostyn]
    resid_up[~up_hostyn] = mu_sh0es_up[~up_hostyn] - mu_model_up[~up_hostyn]

    resid_down = np.zeros(len(down_data))

    resid_down[down_hostyn] = muceph_down[down_hostyn] - mu_model_down[down_hostyn]
    resid_down[~down_hostyn] = mu_sh0es_down[~down_hostyn] - mu_model_down[~down_hostyn]

    # Calculate differences and dw statistics
    diff_up = np.diff(resid_up)
    diff_down = np.diff(resid_down)

    dw_up = np.sum(diff_up**2) / np.sum(resid_up**2)
    dw_down = np.sum(diff_down**2) / np.sum(resid_down**2)

    return dw_up, dw_down

def total_dw(healpix_dirs: np.ndarray, pts: int, zup: float, zdown: float, datos: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the total dw statistic for all healpix directions.

    Args:
        healpix_dirs (np.ndarray): Array of healpix directions.
        datos (Tuple): Tuple containing data arrays.

    Returns:
        Tuple[float, float]: The total dw statistics for up and down directions.
    """
    h0f = datos[4]
    q0f = datos[5]
     
    dw_ups = []
    dw_downs = []

    for healpix_dir in healpix_dirs:
        dw_up, dw_down = hemispheric_dw(healpix_dir, datos)
        dw_ups.append(dw_up)
        dw_downs.append(dw_down)

    dw_ups = np.array(dw_ups)
    dw_downs = np.array(dw_downs)

    #save dw statistics
    
    
    data_dw = np.column_stack((dw_ups, dw_downs))
    filename_dw = f'compilations/[DW][SH0ES_CALIB][HEALPIX](hf={h0f}_qf={q0f})({zup}>z>{zdown}).txt'
    header_dw = 'This is the data for the dw statistics for the following parameters: \n {} points, q0f= {}, h0f= {}, zup= {}, zdown= {}\n dw_up dw_down'.format(pts, q0f, h0f, zup, zdown)
    np.savetxt(filename_dw, data_dw, header=header_dw)
    

    return dw_ups, dw_downs, data_dw