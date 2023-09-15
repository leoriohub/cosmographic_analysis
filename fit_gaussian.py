import numpy as np
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