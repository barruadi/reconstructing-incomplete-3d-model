from scipy.spatial import KDTree
import numpy as np

def replace_missing_with_mean(points):
    col_means = np.nanmean(points, axis=0)
    filled_points = np.where(np.isnan(points), col_means, points)
    return filled_points

def replace_missing_with_local_mean(points, radius=0.5):
    filled_points = points.copy()
    non_missing_points = points[~np.isnan(points).any(axis=1)]
    tree = KDTree(non_missing_points)

    for i in range(len(points)):
        if np.isnan(points[i]).any():
            
            query_point = points[i]
            if not np.isnan(query_point).any():
                neighbors_idx = tree.query_ball_point(query_point, r=radius)
                if neighbors_idx:
                    neighbors = non_missing_points[neighbors_idx]
                    local_mean = np.mean(neighbors, axis=0)
                    filled_points[i] = local_mean
                else:
                    filled_points[i] = np.nanmean(non_missing_points, axis=0)

    return filled_points
