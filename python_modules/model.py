import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Dataset
def generate_sphere(radius=5, num_points=1000):
    theta = np.random.uniform(0, np.pi, num_points)  # Latitude
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Longitude

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.vstack([x, y, z]).T

# Generate a sphere point cloud
original_points = generate_sphere()

# Step 2: Introduce Missing Data
def introduce_missing_data(points, missing_ratio=0.2):
    mask = np.random.rand(*points.shape) < missing_ratio
    points_with_missing = points.copy()
    points_with_missing[mask] = np.nan
    return points_with_missing

points_with_missing = introduce_missing_data(original_points)

# Step 3: Replace Missing Data with Initial Guess
def replace_missing_with_mean(points):
    col_means = np.nanmean(points, axis=0)
    filled_points = np.where(np.isnan(points), col_means, points)
    return filled_points

initial_guess = replace_missing_with_mean(points_with_missing)

# Step 4: Apply SVD for Reconstruction
def svd_reconstruction(points):
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    reconstructed_points = np.dot(U, np.dot(np.diag(S), Vt))
    return reconstructed_points

reconstructed_points = svd_reconstruction(initial_guess)

# Step 5: Calculate RMSE (Root Mean Square Error)
def calculate_rmse(original, reconstructed):
    mask = ~np.isnan(original)
    mse = np.mean((original[mask] - reconstructed[mask]) ** 2)
    return np.sqrt(mse)

rmse = calculate_rmse(original_points, reconstructed_points)
print("RMSE:", rmse)

# Step 6: Visualize the Results
def visualize_3d(original, missing, reconstructed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='Original', alpha=0.6)
    ax.scatter(missing[:, 0], missing[:, 1], missing[:, 2], c='red', label='Missing', alpha=0.6)
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='green', label='Reconstructed', alpha=0.6)

    ax.legend()
    plt.show()

visualize_3d(original_points, points_with_missing, reconstructed_points)

# Step 7: Save the Reconstructed Data to CSV
reconstructed_df = pd.DataFrame(reconstructed_points, columns=['x', 'y', 'z'])
reconstructed_df.to_csv("reconstructed_point_cloud.csv", index=False)
print("Reconstructed point cloud saved to 'reconstructed_point_cloud.csv'")
