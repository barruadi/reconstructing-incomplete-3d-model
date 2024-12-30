import numpy as np

def normalize_data(points):
    mean = np.nanmean(points, axis=0)
    std = np.nanstd(points, axis=0)
    normalized_points = (points - mean) / (std + 1e-8)
    return normalized_points, mean, std

# normalized_points, mean, std = normalize_data(initial_guess)

def svd_reconstruction(points):
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    reconstructed_points = np.dot(U, np.dot(np.diag(S), Vt))
    return reconstructed_points

def denormalize_data(points, mean, std):
    denormalized_points = (points * std) + mean
    denormalized_points[np.abs(denormalized_points) < 1e-8] = 0.0
    return denormalized_points

# reconstructed_points = denormalize_data(reconstructed_normalized, mean, std)