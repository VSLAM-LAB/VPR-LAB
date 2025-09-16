import os
import numpy as np
import argparse

from create_groundtruth_matrix import create_GT
from utilities import load_yaml
from utilities import load_trajectory

from utilities import load_image_list_with_absolute_paths
from plot_functions.plot_functions import plot_clickable_GT

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def zero_diagonal_band(M, thickness=1):
    rows, cols = M.shape
    for offset in range(-thickness, thickness + 1):
        diag_indices = np.diag_indices(min(rows, cols))
        row_idx = diag_indices[0]
        col_idx = diag_indices[1] + offset
        # Filter out indices that go out of bounds
        mask = (col_idx >= 0) & (col_idx < cols)
        M[row_idx[mask], col_idx[mask]] = 0
    return M


def main():

    parser = argparse.ArgumentParser(description="Generate mask matrix from experiment config.")
    parser.add_argument("--exp_yaml", type=str, default="exp.yaml", help="Path to experiment YAML config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Inputs
    print(f"\n{SCRIPT_LABEL}: {args.exp_yaml}")
    exp_yaml = load_yaml(args.exp_yaml)

    # Experiment paths
    path_to_gt_file_db = exp_yaml["path_to_gt_file_db"]
    path_to_gt_file_query = exp_yaml["path_to_gt_file_query"]

    path_to_rgb_list_db = exp_yaml["path_to_rgb_list_db"]
    path_to_rgb_list_query = exp_yaml["path_to_rgb_list_query"]

    path_to_rgb_folder_db = exp_yaml["path_to_rgb_folder_db"]
    path_to_rgb_folder_query = exp_yaml["path_to_rgb_folder_query"]

    GT_path = exp_yaml["GT_path"]
    M_path = exp_yaml["M_path"]

    # # Load image lists
    #db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_query, path_to_rgb_folder_db, path_to_rgb_folder_query)
    #gt_db = load_trajectory(path_to_gt_file_db)
    #gt_query = load_trajectory(path_to_gt_file_query)

    # Create Mask
    GT = np.load(GT_path)
    M = np.ones_like(GT, dtype=np.uint8)
    M = zero_diagonal_band(M, thickness=40)

    #plot_clickable_GT(GT = M, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min')

    # Save matrices
    np.save(M_path, M)  

if __name__ == "__main__":
    main()