import numpy as np
import healpy as hp
from healpix_vectors import get_healpix_vectors


def get_max_anisotropy(h0_up, h0_down, q0_up, q0_down, healpix_dirs):

    
    delta_h0 =np.abs(h0_up - h0_down)
    delta_q0 = np.abs(q0_up - q0_down)

    delta_h0_max_ind = np.argmax(np.abs(delta_h0))
    delta_q0_max_ind = np.argmax(np.abs(delta_q0))

    max_q0_anis_vec = healpix_dirs[delta_q0_max_ind]
    max_q0_anis_theta, max_q0_anis_phi = hp.vec2ang(max_q0_anis_vec)

    max_h0_anis_vec = healpix_dirs[delta_h0_max_ind]
    max_q0_anis_vec = healpix_dirs[delta_q0_max_ind]

    max_h0_anis_theta, max_h0_anis_phi = hp.vec2ang(max_h0_anis_vec)
    max_q0_anis_theta, max_q0_anis_phi = hp.vec2ang(max_q0_anis_vec)

    max_q0_anis_dec = np.degrees(np.pi/2 - max_q0_anis_theta)[0]
    max_q0_anis_ra = np.degrees(max_q0_anis_phi)[0]

    max_h0_anis_dec = np.degrees(np.pi/2 - max_h0_anis_theta)[0]
    max_h0_anis_ra = np.degrees(max_h0_anis_phi)[0]

    max_dir_h0 = [max_h0_anis_dec, max_h0_anis_ra]
    max_dir_q0 = [max_q0_anis_dec, max_q0_anis_ra]
     
    return max_dir_h0, max_dir_q0