from scipy.spatial import KDTree
import numpy as np

def replace_missing_with_mean(points):
    col_means = np.nanmean(points, axis=0)
    filled_points = np.where(np.isnan(points), col_means, points)
    return filled_points

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

def normalize_data(points):
    mean = np.nanmean(points, axis=0)
    std = np.nanstd(points, axis=0)
    normalized_points = (points - mean) / (std + 1e-8)
    return normalized_points, mean, std

# normalized_points, mean, std = normalize_data(initial_guess)
