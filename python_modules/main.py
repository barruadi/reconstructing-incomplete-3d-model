from steps.data_preparation import *
from steps.data_preprocessing import *
from steps.reconstruction import *
from steps.validation import *
from steps.visualize import *

print("Welcome to 3D reconstructing program")

print("Step 1: Data Preparation")
type_model = int(input(">>> "))

match type_model:
    case 1:
        # SPHERE
        original_points = generate_sphere()
        points_with_missing = introduce_missing_data(original_points)

    case 2:
        # CSV
        type_dataset = int(input(">>> "))
        match type_dataset:
            case 1:
                # INPUT CSV PATH
                original_csv_path = input("Enter the path to the CSV file containing the original points: ").strip()
                missing_csv_path = input("Enter the path to the CSV file containing the points with missing values: ").strip()

                original_points = load_dataset(original_csv_path)
                points_with_missing = load_dataset(missing_csv_path)
            case 2:
                print("multiple datasets")

    case _:
        print("invalid input")

# REPLACE MISSING DATA WITH INITIAL GUESS
print("Step 2: Inital Guess")
print("1. Replace missing data with mean, 2. Replace missing data with local mean")
type_iniital_guess = int(input(">>> "))

match (type_iniital_guess):
    case 1:
        initial_guess = replace_missing_with_mean(points_with_missing)
    case 2:
        initial_guess = replace_missing_with_local_mean(points_with_missing)
    case _:
        print("invalid input")

# NORMALIZE DATA
normalized_points, mean, std = normalize_data(initial_guess)

# APPLY SVD FOR RECONSTRUCTION
reconstructed_normalized = svd_reconstruction(normalized_points)

# DENORMALIZE THE RECONSTRUCTED DATA
reconstructed_points = denormalize_data(reconstructed_normalized, mean, std)

# CALCULATE RMSE
rmse = calculate_rmse(original_points, reconstructed_points)
print("RMSE:", rmse)


# VISUALIZE THE RESULTS
visualize_3d(original_points, points_with_missing, reconstructed_points)
reconstructed_df = pd.DataFrame(reconstructed_points, columns=['x', 'y', 'z'])
reconstructed_df.to_csv("reconstructed_point_cloud.csv", index=False)
print("Reconstructed point cloud saved to 'reconstructed_point_cloud.csv'")
