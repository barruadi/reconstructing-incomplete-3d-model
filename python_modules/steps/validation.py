import numpy as np

def calculate_rmse(original, reconstructed):
    mask = ~np.isnan(original)
    mse = np.mean((original[mask] - reconstructed[mask]) ** 2)
    return np.sqrt(mse)