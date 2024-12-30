import os
import pandas as pd
import numpy as np

def load_dataset(file_path):
    path = os.path.join(os.getcwd(), "model", file_path)
    return pd.read_csv(path).to_numpy()

def load_multiple_datasets(file_paths):
    datasets = [pd.read_csv(file_path).to_numpy() for file_path in file_paths]
    return datasets

def generate_sphere(radius=5, num_points=1000):
    theta = np.random.uniform(0, np.pi, num_points)  # Latitude
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Longitude

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return np.vstack([x, y, z]).T

def introduce_missing_data(points, missing_ratio=0.2):
    mask = np.random.rand(*points.shape) < missing_ratio
    points_with_missing = points.copy()
    points_with_missing[mask] = np.nan
    return points_with_missing