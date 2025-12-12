import argparse
import numpy as np
import matplotlib.pyplot as plt

from utilities import load_image_list_with_absolute_paths
from utilities import load_trajectory
from plot_functions.plot_functions import plot_clickable_GT
from evaluation_functions import precision_recall_curve, recall_at_k
from plot_functions.plot_functions import plot_precision_recall_curve, plot_recall_at_k

parser = argparse.ArgumentParser(description="Generate mask matrix from experiment config.")
    
parser.add_argument("--gt_file_db", type=str, required=True, help="Path to gt_file_db")
parser.add_argument("--gt_file_q", type=str, required=True, help="Path to gt_file_q")
parser.add_argument("--rgb_list_db", type=str, required=True, help="Path to rgb_list_db")
parser.add_argument("--rgb_list_q", type=str, required=True, help="Path to rgb_list_q")
parser.add_argument("--rgb_folder_db", type=str, required=True, help="Path to rgb_folder_db")
parser.add_argument("--rgb_folder_q", type=str, required=True, help="Path to rgb_folder_q")
parser.add_argument("--gt_matrix", type=str, required=True, help="Path to gt_matrix")
parser.add_argument("--mask_matrix", type=str, required=True, help="Path to mask_matrix")
parser.add_argument("--d_matrix", type=str, required=True, help="Path to d_matrix")

args = parser.parse_args()

# path_to_gt_file_db = args.gt_file_db 
# path_to_gt_file_q = args.gt_file_q 
# path_to_rgb_list_db = args.rgb_list_db
# path_to_rgb_list_q = args.rgb_list_q
# path_to_rgb_folder_db = args.rgb_folder_db
# path_to_rgb_folder_q = args.rgb_folder_q
path_to_GT = args.gt_matrix
path_to_M = args.mask_matrix
path_to_D = args.d_matrix

# db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_q, path_to_rgb_folder_db, path_to_rgb_folder_q)
GT = np.load(path_to_GT)
M = np.load(path_to_M)
D = np.load(path_to_D)

# gt_db = load_trajectory(path_to_gt_file_db)
# gt_query = load_trajectory(path_to_gt_file_q)


############################################################################################################################################
# matching = 'single-match-hard' # 'single-match' # 'single-match-hard'  # 'multi-match'
# r_gt = 10
# num_th = 100

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT, M = M, r_gt = r_gt, D = D, matching=matching, num_th=num_th)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)  
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
#                                     marker=None, color=None, line_style = ':' , label=None, 
#                                     fig = fig, ax = ax)

# plt.show()

############################################################################################################################################
# r_gt = 10

# k_values, recall_at_k_values, avg_probs = recall_at_k(GT = GT, M = M, r_gt = r_gt, D = D)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)  
# fig, ax = plot_recall_at_k(k_values, recall_at_k_values, avg_probs, marker='o', color=None, line_style = '--' , label=None, 
#     fig = fig, ax = ax)

# plt.show()

############################################################################################################################################