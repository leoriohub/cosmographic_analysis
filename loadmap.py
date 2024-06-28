import numpy as np

def load_hubble_data(file_path):
    # Load data using numpy, skipping the first 3 lines (2 header lines + 1 line of column names)
    data = np.loadtxt(file_path, skiprows=3)
    
    return tuple(data.T)  # Transpose and return as a tuple of arrays