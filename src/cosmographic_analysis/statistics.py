"""Statistical utilities for the cosmographic analysis.

Includes Gaussian fitting and Monte Carlo p-value computation.
"""

import numpy as np
from typing import Dict
from scipy.optimize import minimize
from scipy.stats import norm

def gaussian_likelihood(params: list, data: list) -> float:
    """
    Calculate the negative log-likelihood of the Gaussian distribution.

    Parameters:
        params (list): List of two parameters: mean and standard deviation.
        data (list): The data points to calculate likelihood for.

    Returns:
        float: The negative log-likelihood value.
    """
    mean, std = params
    log_likelihood = np.sum(norm.logpdf(data, loc=mean, scale=std))
    return -log_likelihood

def fit_gaussian(data: np.ndarray) -> tuple:
    """
    Fit a Gaussian distribution to the given data.

    Parameters:
        data (numpy.ndarray): List of data points to fit the Gaussian distribution to.

    Returns:
        tuple: A tuple containing:
            - x_gaussian (array): Array of x-values corresponding to the fitted Gaussian distribution.
            - y_gaussian (array): Array of y-values corresponding to the fitted Gaussian distribution.
    """
    initial_guess = [np.mean(data), np.std(data)]
    # Minimize the likelihood function
    result = minimize(gaussian_likelihood, initial_guess, args=(data,), method='L-BFGS-B')
    # Extract the best-fit parameters
    mu_best_fit, std_best_fit = result.x
    # Generate the x and y values for the Gaussian distribution
    x_gaussian = np.linspace(np.min(data), np.max(data), 100)
    y_gaussian = norm.pdf(x_gaussian, loc=mu_best_fit, scale=std_best_fit)
    return x_gaussian, y_gaussian


def map_statistics(h0: np.ndarray, q0: np.ndarray, delta_h0_data_max: np.ndarray, delta_q0_data_max: np.ndarray) -> Dict[str, float]:
    """
    Calculate and return various statistics for the provided data arrays.

    Args:
        h0 (np.ndarray): Array of h0 values.
        q0 (np.ndarray): Array of q0 values.
        delta_h0_data_max (np.ndarray): Array of delta_h0_data_max values.
        delta_q0_data_max (np.ndarray): Array of delta_q0_data_max values.

    Returns:
        Dict[str, float]: A dictionary containing the calculated statistics.
    """
    h0_mean = np.mean(h0)
    q0_mean = np.mean(q0)
    h0_dev = np.std(h0)
    q0_dev = np.std(q0)

    statistics = {
        "h0_mean": h0_mean,
        "q0_mean": q0_mean,
        "h0_std_dev": h0_dev,
        "q0_std_dev": q0_dev,
    }
    
    print("1 sigma h0: ", h0_mean,"+/-", h0_dev)
    print("1 sigma q0: ", q0_mean,"+/-", q0_dev)

    return statistics


def mc_statistics(maximum_anisotropy_data: np.ndarray, maximum_anisotropy_mc: np.ndarray) -> list :
    """
    Calculate and print Monte Carlo p-values.

    p-value = percentage of MC iterations where the simulated anisotropy
    exceeds the observed value. A small p-value (close to 0%) indicates
    the observed anisotropy is significantly larger than expected under
    the null hypothesis.

    Args: Receives an array of maximum anisotropy values for LCDM and ISO iterations.
    
    Returns: Returns a list of pvalues.
    
    """
    
    delta_h0_data_max = maximum_anisotropy_data[0]
    delta_q0_data_max = maximum_anisotropy_data[1]

    delta_h0_lcdm_max = maximum_anisotropy_mc[0]
    delta_q0_lcdm_max = maximum_anisotropy_mc[1]
    delta_h0_iso_max = maximum_anisotropy_mc[2]
    delta_q0_iso_max = maximum_anisotropy_mc[3]

    repetitions = len(delta_h0_lcdm_max)

    p_h0_iso_max = np.sum(delta_h0_iso_max > delta_h0_data_max) / repetitions * 100
    p_q0_iso_max = np.sum(delta_q0_iso_max > delta_q0_data_max) / repetitions * 100
    p_h0_lcdm_max = np.sum(delta_h0_lcdm_max > delta_h0_data_max) / repetitions * 100
    p_q0_lcdm_max = np.sum(delta_q0_lcdm_max > delta_q0_data_max) / repetitions * 100

    print(f"Porcentaje de repeticiones que dan delta_h0 mayor a los datos (ISO) = {p_h0_iso_max}")
    print(f"Porcentaje de repeticiones que dan delta_q0 mayor a los datos (ISO) = {p_q0_iso_max}\n")
    print(f"Porcentaje de repeticiones que dan delta_h0 mayor a los datos (LCDM) = {p_h0_lcdm_max}")
    print(f"Porcentaje de repeticiones que dan delta_q0 mayor a los datos (LCDM) = {p_q0_lcdm_max}\n")

    print(f"La media de delta_h0_max (ISO) es = {np.mean(delta_h0_iso_max)}")
    print(f"La media de delta_q0_max (ISO) es = {np.mean(delta_q0_iso_max)}\n")
    print(f"La media de delta_h0_max (LCDM) es = {np.mean(delta_h0_lcdm_max)}")
    print(f"La media de delta_q0_max (LCDM) es = {np.mean(delta_q0_lcdm_max)}\n")

    print(f"La desviación estándar de delta_h0_max (ISO) es = {np.std(delta_h0_iso_max)}")
    print(f"La desviación estándar de delta_q0_max (ISO) es = {np.std(delta_q0_iso_max)}\n")
    print(f"La desviación estándar de delta_h0_max (LCDM) es = {np.std(delta_h0_lcdm_max)}")
    print(f"La desviación estándar de delta_q0_max (LCDM) es = {np.std(delta_q0_lcdm_max)}")
    
    p_values = [p_h0_iso_max, p_q0_iso_max, p_h0_lcdm_max, p_q0_lcdm_max]

    return p_values
