import numpy as np

def load_hubble_data(file_path):
    # Load data using numpy, skipping the first 4 lines (2 header lines + 1 line break + 1 line of column names)
    data = np.loadtxt(file_path, skiprows=4)
    return data

# Example usage:
if __name__ == "__main__":
    file_path = "hubble_data.txt"
    data = load_hubble_data(file_path)
    print("Data:")
    print(data)
