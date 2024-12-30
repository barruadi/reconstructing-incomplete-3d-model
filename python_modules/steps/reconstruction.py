import numpy as np

def svd_reconstruction(points):
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    reconstructed_points = np.dot(U, np.dot(np.diag(S), Vt))
    return reconstructed_points

def denormalize_data(points, mean, std):
    denormalized_points = (points * std) + mean
    denormalized_points[np.abs(denormalized_points) < 1e-8] = 0.0
    return denormalized_points

# reconstructed_points = denormalize_data(reconstructed_normalized, mean, std)