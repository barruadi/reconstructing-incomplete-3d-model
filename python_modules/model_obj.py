import os
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

type = input("Enter the type of dataset to load (1 for one dataset, 2 for mutilple dataset): ").strip()

if type == '1':
    # Function to load dataset from CSV
    def load_dataset(file_path):
        path = os.path.join(os.getcwd(), "model", file_path)
        return pd.read_csv(path).to_numpy()

    # Step 1: Load Original and Missing Data from CSV
    original_csv_path = input("Enter the path to the CSV file containing the original points: ").strip()
    missing_csv_path = input("Enter the path to the CSV file containing the points with missing values: ").strip()
    # print(os.getcwd())

    original_points = load_dataset(original_csv_path)
    points_with_missing = load_dataset(missing_csv_path)

    nan_count = np.isnan(points_with_missing).sum()
    print(f"Number of NaN values in points_with_missing: {nan_count}")
else:
    # Function to load datasets from multiple CSV files
    def load_multiple_datasets(file_paths):
        datasets = [pd.read_csv(file_path).to_numpy() for file_path in file_paths]
        return datasets

    # Step 1: Load Multiple Original and Missing Data from CSVs
    num_files = int(input("Enter the number of datasets to load: ").strip())
    original_csv_paths = [input(f"Enter the path to the CSV file containing the original points for dataset {i+1}: ").strip() for i in range(num_files)]
    missing_csv_paths = [input(f"Enter the path to the CSV file containing the points with missing values for dataset {i+1}: ").strip() for i in range(num_files)]

    original_datasets = load_multiple_datasets(original_csv_paths)
    missing_datasets = load_multiple_datasets(missing_csv_paths)

    # Combine datasets for processing
    original_points = np.vstack(original_datasets)
    points_with_missing = np.vstack(missing_datasets)


type_mean = input("Enter the type of mean to use (1 for global mean, 2 for local mean): ").strip()
if type_mean == '2':
    def replace_missing_with_local_mean(points, radius=0.5):
        filled_points = points.copy()
        non_missing_points = points[~np.isnan(points).any(axis=1)]  # Exclude missing points
        tree = KDTree(non_missing_points)  # Build KDTree using only non-missing points

        for i in range(len(points)):
            if np.isnan(points[i]).any():
               # Use only valid (non-NaN) parts of the query point
                query_point = points[i]
                if not np.isnan(query_point).any():
                    neighbors_idx = tree.query_ball_point(query_point, r=radius)
                    if neighbors_idx:
                        neighbors = non_missing_points[neighbors_idx]
                        local_mean = np.mean(neighbors, axis=0)
                        filled_points[i] = local_mean
                    else:
                        filled_points[i] = np.nanmean(non_missing_points, axis=0)  # Fallback to global mean if no neighbors

        return filled_points
    
    initial_guess = replace_missing_with_local_mean(points_with_missing)

else:
    # Step 2: Replace Missing Data with Initial Guess
    def replace_missing_with_mean(points):
        col_means = np.nanmean(points, axis=0)
        filled_points = np.where(np.isnan(points), col_means, points)
        return filled_points

    initial_guess = replace_missing_with_mean(points_with_missing)

# Step 3: Apply SVD for Reconstruction
def svd_reconstruction(points):
    U, S, Vt = np.linalg.svd(points, full_matrices=False)
    reconstructed_points = np.dot(U, np.dot(np.diag(S), Vt))
    return reconstructed_points

reconstructed_points = svd_reconstruction(initial_guess)

# Step 4: Calculate RMSE (Root Mean Square Error)
def calculate_rmse(original, reconstructed):
    mask = ~np.isnan(original)
    mse = np.mean((original[mask] - reconstructed[mask]) ** 2)
    return np.sqrt(mse)

rmse = calculate_rmse(original_points, reconstructed_points)
print("RMSE:", rmse)

# Step 5: Visualize the Results
def visualize_3d(original, missing, reconstructed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='Original', alpha=0.6)
    ax.scatter(missing[:, 0], missing[:, 1], missing[:, 2], c='red', label='Missing', alpha=0.6)
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='green', label='Reconstructed', alpha=0.6)

    ax.legend()
    plt.show()

visualize_3d(original_points, points_with_missing, reconstructed_points)

# Step 6: Save the Reconstructed Data to CSV
reconstructed_df = pd.DataFrame(reconstructed_points, columns=['x', 'y', 'z'])
reconstructed_df.to_csv("reconstructed_point_cloud.csv", index=False)
print("Reconstructed point cloud saved to 'reconstructed_point_cloud.csv'")
