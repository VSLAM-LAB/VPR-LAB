import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utilities import load_image_list_with_absolute_paths
from utilities import load_trajectory
from utilities import load_yaml

from evaluation_functions import precision_recall_curve, recall_at_k
from plot_functions.plot_functions import plot_precision_recall_curve, plot_recall_at_k
from plot_functions.clickable_plot_functions import plot_precision_recall_curve_clickable
from plot_functions.clickable_plots import plot_clickable_matrix

parser = argparse.ArgumentParser(description="Generate mask matrix from experiment config.")
parser.add_argument("--exp_yaml", type=str, default="exp.yaml", help="Path to experiment YAML config file")

args = parser.parse_args()

# Inputs
exp_yaml = load_yaml(args.exp_yaml)

path_to_rgb_list_db = exp_yaml['rgb_list_db']
path_to_rgb_list_q = exp_yaml['rgb_list_q']
path_to_rgb_folder_db = Path(path_to_rgb_list_db).parent
path_to_rgb_folder_q = Path(path_to_rgb_list_q).parent
path_to_gt_file_db = path_to_rgb_folder_db / "groundtruth.csv"
path_to_gt_file_q = path_to_rgb_folder_q / "groundtruth.csv"

path_to_GT = Path(exp_yaml["log_dir"]) / "GT.npy"
path_to_M = Path(exp_yaml["log_dir"]) / "M.npy"
path_to_D = exp_yaml['log_dir'] + f"/D.npy"

db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_q, path_to_rgb_folder_db, path_to_rgb_folder_q)

D = np.load(path_to_D)

if path_to_GT.exists():
    GT = np.load(path_to_GT)
    gt_db = load_trajectory(path_to_gt_file_db)
    gt_query = load_trajectory(path_to_gt_file_q)
else:
    GT = None

if path_to_M.exists():
    M = np.load(path_to_M)
else:
    M = np.ones_like(D, dtype=np.uint8)


###################################################
# matching = 'single-match-hard' # 'single-match' # 'single-match-hard'  # 'multi-match'
# r_gt = 10
# num_th = 100

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT, M = M, r_gt = r_gt, D = D, matching=matching, num_th=num_th)
# plot_precision_recall_curve_clickable(
#     precisions, recalls, thresholds, stats_at_rand,
#     GT, M, r_gt, D=D, matching=matching)
# plt.show()

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
# matching = 'multi-match' # 'single-match' # 'single-match-hard'  # 'multi-match'
# r_gt = 10
# num_th = 100

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT, M = M, r_gt = r_gt, D = D, matching=matching, num_th=num_th)
# D[D > 0.8] = np.inf
# plot_clickable_matrix(matrix = D, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query)

# ############################################################################################################################################
# # Visualize Distance Matrix
# if GT is not None :
#     plot_clickable_matrix(matrix = (D * M), db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query)
# else:
#     plot_clickable_matrix(matrix = (D * M), db_df = db_df, query_df = query_df, lightglue_matching_func=True)


