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
            
        # INPUT CSV PATH
        original_csv_path = input("Enter the path to the CSV file containing the original points: ").strip()
        missing_csv_path = input("Enter the path to the CSV file containing the points with missing values: ").strip()
        original_points = load_dataset(original_csv_path)
        points_with_missing = load_dataset(missing_csv_path)

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

missing_initial_guess = initial_guess[np.isnan(points_with_missing).any(axis=1)]
# VISUALIZE THE RESULTS
# visualize_3d(original_points, points_with_missing, reconstructed_points)
visualize_preprocess(missing_initial_guess)
reconstructed_df = pd.DataFrame(reconstructed_points, columns=['x', 'y', 'z'])
reconstructed_df.to_csv("reconstructed.csv", index=False)
print("Reconstructed point cloud saved to 'reconstructed.csv'")

initial_guess_df = pd.DataFrame(missing_initial_guess, columns=['x', 'y', 'z'])
initial_guess_df.to_csv("initial_guess.csv", index=False)