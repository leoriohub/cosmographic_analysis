import numpy as np
from typing import Dict

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


def mc_statistics(delta_h0_iso_max: np.ndarray, delta_q0_iso_max: np.ndarray, delta_h0_lcdm_max: np.ndarray, delta_q0_lcdm_max: np.ndarray, delta_h0_data_max: np.ndarray, delta_q0_data_max: np.ndarray) -> None:
    """
    Calculate and print Monte Carlo statistics.

    """

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
