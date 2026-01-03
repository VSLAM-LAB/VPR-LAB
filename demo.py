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

path_to_gt_file_db = exp_yaml['gt_file_db']
path_to_gt_file_q = exp_yaml['gt_file_q']
path_to_rgb_list_db = exp_yaml['rgb_list_db']
path_to_rgb_list_q = exp_yaml['rgb_list_q']
path_to_rgb_folder_db = Path(exp_yaml['rgb_folder_db']).parent
path_to_rgb_folder_q = Path(exp_yaml['rgb_folder_q']).parent

# path_to_GT = exp_yaml['gt_matrix']
# path_to_M = exp_yaml['mask_matrix']
path_to_D = exp_yaml['log_dir'] + f"/D.npy"

db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_q, path_to_rgb_folder_db, path_to_rgb_folder_q)
#GT = np.load(path_to_GT)
# M = np.load(path_to_M)
#D = np.load(path_to_D)
# D0 = np.load(path_to_D.replace('D_0_megaloc.npy','D_90_megaloc.npy'))
# D1 = np.load(path_to_D.replace('D_0_megaloc.npy','D_90_megaloc.npy'))
# D2 = np.load(path_to_D.replace('D_0_megaloc.npy','D_180_megaloc.npy'))
# D3 = np.load(path_to_D.replace('D_0_megaloc.npy','D_270_megaloc.npy'))
# D = np.minimum.reduce([D0, D1, D2, D3])
D = np.load(path_to_D.replace('D_0_megaloc.npy','D_90_megaloc.npy'))

# print(GT.shape)
gt_db = load_trajectory(path_to_gt_file_db)
gt_query = load_trajectory(path_to_gt_file_q)

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

#precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT, M = M, r_gt = r_gt, D = D, matching=matching, num_th=num_th)
#D[D > 0.6] = np.inf
plot_clickable_matrix(matrix = D, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query)
#D = D<0.9  # Example thresholding for visualization
#plot_clickable_matrix(matrix = D, db_df = db_df, query_df = query_df, lightglue_matching_func=True)
