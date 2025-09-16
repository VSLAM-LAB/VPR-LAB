
import numpy as np
import matplotlib.pyplot as plt
import os

from utilities import load_yaml
from create_groundtruth_matrix import compare_GT
from utilities import load_trajectory, create_Shannon
from utilities import load_image_list_with_absolute_paths
from utilities import combine_distance_matrix
from utilities import generate_definecolor_lines, generate_multicolumn_line
from evaluation_functions import precision_recall_curve, recall_at_k
from plot_functions.clickable_plot_functions import plot_precision_recall_curve_clickable
from plot_functions.best_k_clickable import overlay_distance_gt_mask_browser
from plot_functions.plot_functions import plot_precision_recall_curve, plot_recall_at_k
from plot_functions.plot_functions import plot_clickable_GT

###########################################################
# PAPER SETTINGS
paper_settings = {}
factor = 2
paper_settings = {
    'legend_on': False,
    'title_on': False,
    'xlabel_on': False,
    'ylabel_on': False,
    'top': False,
    'right': False,
    'grid_on': True,
    'grid_alpha': 0.4,
    'grid_style': '--',
    'line_width': 2.0 * factor,   
    'marker_size': 4 * factor,
    'settings': {
        'font.size': 18 * factor,
        'axes.titlesize': 18 * factor,
        'axes.labelsize': 18 * factor,
        'xtick.labelsize': 16 * factor,
        'ytick.labelsize': 16 * factor,
        'legend.fontsize': 16 * factor,
    }
}
plt.rcParams.update(paper_settings['settings'])
###########################################################
def format_latex_row(seq_name, recall_at_ones):
    recalls = np.array(recall_at_ones, dtype=float)

    def format_group(values):
        """Format one group: bold best, underline second best"""
        arr = np.array(values, dtype=float)
        sorted_idx = np.argsort(arr)
        best, second = sorted_idx[-1], sorted_idx[-2]

        out = []
        for i, val in enumerate(arr):
            if i == best:
                out.append(f"\\textbf{{{val:.2f}}}")
            elif i == second:
                out.append(f"\\underline{{{val:.2f}}}")
            else:
                out.append(f"{val:.2f}")
        return out

    # split into groups
    group1 = format_group(recalls[0:3])
    group2 = format_group(recalls[3:6])

    all_vals = group1 + group2
    return seq_name + " & " + " & ".join(all_vals) + " \\\\"

########################################################################################################################################
# Plot Dc for each method (Fig. 6: Evaluation of VPR approaches)
def get_results(matching_, GT_ , M_ , D_, r_gt_, methods_, seq_name_, db_df_ , query_df_, num_th_ = 100,  verbose=False):
    if verbose:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)  

    recall_at_ones = []
    recall_at_095 = []

    # for i, method in enumerate(methods_):
    #     precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_['GT'], M = M_['M'], r_gt = r_gt_, 
    #                                                                             D = D_[method]['D_0'], matching=matching_, num_th=num_th_)
        
    #     if verbose:
    #         fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
    #                                             marker=None, color=colors[i], line_style = ':' , label=method, 
    #                                             fig = fig, ax = ax, paper_settings = paper_settings)
    #         # # Example usage:
    #         # D  = np.random.rand(10, 10)
    #         # GT = (np.random.rand(10, 10) > 0.7).astype(int)
    #         # M  = (np.random.rand(10, 10) > 0.2).astype(int)
    #         # overlay_distance_gt_mask_browser(D = D_[method]['D_0'], GT = GT_, M = M_, k=3, r_gt = r_gt_, 
    #         #                                      db_df =  db_df_, query_df= query_df_ )
        
    #     recall_at_ones.append(precisions[-1])
    #     mask_095 = precisions >= 0.95
    #     if np.any(mask_095):
    #         recall_at_095.append(np.max(recalls[mask_095]))
    #     else:
    #         recall_at_095.append(0.0)
        
    #     # k_values, recall_at_k_values, avg_probs = recall_at_k(GT = GT_['GT'], M = M_['M'], r_gt = r_gt, D = D[method]['D_0'])
    #     # fig, ax = plot_recall_at_k(k_values, recall_at_k_values, avg_probs, marker='o', color=colors[i], line_style = '--' , label=method, 
    #     #                            fig = fig, ax = ax, paper_settings = paper_settings)

    if verbose:
        plt.show()

def run_sequence(exp_yaml_, seq_name_, methods_, methods_names_, colors_, rotations_):

    # Experiment paths
    path_to_gt_file_db = exp_yaml_["path_to_gt_file_db"]
    path_to_gt_file_query = exp_yaml_["path_to_gt_file_query"]

    path_to_rgb_list_db = exp_yaml_["path_to_rgb_list_db"]
    path_to_rgb_list_query = exp_yaml_["path_to_rgb_list_query"]

    path_to_rgb_folder_db = exp_yaml_["path_to_rgb_folder_db"]
    path_to_rgb_folder_query = exp_yaml_["path_to_rgb_folder_query"]

    D_path = exp_yaml_["D_path"]
    GT_path = exp_yaml_["GT_path"]
    M_path = exp_yaml_["M_path"]

    r_gt = exp_yaml_["r_gt"]

    # Load image lists
    db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_query, path_to_rgb_folder_db, path_to_rgb_folder_query)
    gt_db = load_trajectory(path_to_gt_file_db)
    gt_query = load_trajectory(path_to_gt_file_query)
    
    M = {}
    M['M'] = np.load(M_path)

    GT = {}
    GT['GT'] = np.load(GT_path)
    path = str.replace(GT_path, 'GT', 'GT_superpoint_lightglue')
    if os.path.exists(path): GT['GT_superpoint_lightglue'] = np.load(path)
    path = str.replace(GT_path, 'GT', 'GT_orb_knn')
    if os.path.exists(path): GT['GT_orb_knn'] = np.load(path)
    path = str.replace(GT_path, 'GT', 'GT_sift_knn')
    if os.path.exists(path): GT['GT_sift_knn'] = np.load(path)

    I, I_bin = create_Shannon(db_df, query_df)
    # for idx, val in enumerate(np.linspace(1, 0.5, 50)):
    #     M[idx] = I_bin[idx] * M['M'] 
    # M['I097rand']  = np.random.permutation(I_bin['097'].flatten()).reshape(I_bin['097'].shape) * M['M']

    plot_clickable_GT(GT = I, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query)


    D = {}
    for method in methods_:
        D[method] = {}
        for rotation in rotations_:
            D[method][rotation] = np.load(str.replace(D_path, 'D.npy', f'{rotation}_{method}.npy'))
        D[method]['Dc'] = combine_distance_matrix(D[method]['D_0'], D[method]['D_90'], D[method]['D_180'], D[method]['D_270'], mode = '0')
        #D[method]['Dc_I'] = np.where(I_bin == 1, D[method]['Dc'], np.inf)

    ########################################################################################################################################
    # matching = 'multi-match'
    # get_results(matching_ = matching, GT_ = GT , M_ = M , D_ = D, r_gt_ = r_gt, seq_name_ = seq_name_, 
    #             methods_ = methods, num_th_ = 100,  verbose=True, db_df_ = db_df, query_df_ = query_df)

    ########################################################################################################################################
    # matching = 'single-match-hard'
    # num_th = 100
    # GT_ = GT['GT_superpoint_lightglue']
    # M_ = M['M']
    # D_ = D['megaloc']['Dc']
    # precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M_, r_gt = r_gt, D = D_, matching=matching, num_th=num_th)

    # plot_precision_recall_curve_clickable(precisions = precisions, recalls = recalls, thresholds = thresholds, stats_at_rand = stats_at_rand,
    #                                     GT = GT_, M = M_, r_gt = r_gt, D = D_, matching=matching,
    #                                     marker='o', color='k', label=matching, paper_settings = paper_settings)
    # plt.show()
    ########################################################################################################################################
    # GT_ =  GT['GT_superpoint_lightglue'] # GT_superpoint_lightglue 
    # GT_ = np.where(M, GT_, np.inf)
    # plot_clickable_GT(GT = GT_, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min', paper_settings = paper_settings)
    
    ########################################################################################################################################
    # Figure Effect of ground-truth curation on VPR evaluation.

    # GT_r = np.where(GT['GT'] < 10, GT, np.inf)
    # GT_r = np.where(M['M'] == 1, GT_r, np.inf)
    # GT['GT_superpoint_lightglue'] = np.where(M['M'] == 1, GT['GT_superpoint_lightglue'], np.inf)
    # GT['GT_sift_knn'] = np.where(M['M'] == 1, GT['GT_sift_knn'], np.inf)
    # GT['GT_orb_knn'] = np.where(M['M'] == 1, GT['GT_orb_knn'], np.inf)
    #compare_GT(GT_r, GT['GT'], "GT_r", "GT_r")
    #compare_GT(GT_r, GT['GT_superpoint_lightglue'], "GT_r", "Superpoint + LightGlue")
    #compare_GT(GT_r, GT['GT_sift_knn'], "GT_r", "SIFT + KNN")
    #compare_GT(GT_r, GT['GT_orb_knn'], "GT_r", "ORB + KNN")
    ########################################################################################################################################

#################

##########################################################################################################################################
# Main script

exp_yamls = ["exp_ASPA135.yaml",
            # "exp_ASPA135.yaml", 
            #  "exp_ASPA136.yaml",
            #  "exp_Q1_2023-12-26_P1.yaml",
            #  "exp_Q2_DJI_202401121044_005.yaml",
            #  "exp_Q2_DJI_202401121140_006.yaml",
            #  "exp_Q3_DJI_202401111904_001.yaml",
            #  "exp_Q3_DJI_202401111932_002.yaml",
            #  "exp_Q3_DJI_202401111932_003.yaml",
            #  "exp_Q3_DJI_202401112014_004.yaml"
             ]
seq_names = ['ASPA 135'
            #'ASPA 135',
            # 'ASPA 136', 'Q1', 'Q2 005', 'Q2 006', 'Q3 001', 'Q3 002', 'Q3 003', 'Q3 004'
            ]


methods = ['salad_noRot', 'salad_Rot', 'megaloc']
methods_names = ['SALAD noRot', 'SALAD Rot', 'MegaLoc']
colors = list(plt.cm.tab20.colors) 
rotations = ['D_0', 'D_90' ,'D_180' ,'D_270']

########################################################################################################################################
for exp_yaml, seq_name in zip(exp_yamls, seq_names):
    run_sequence(exp_yaml_ = load_yaml(exp_yaml), seq_name_ = seq_name, 
                 methods_ = methods, methods_names_ = methods_names, 
                 colors_ = colors, rotations_ = rotations)
    #break


# latex_colors = generate_definecolor_lines(methods, colors)
# line = generate_multicolumn_line(methods, methods_names)
# print(latex_colors)
# print(line)



# ########################################################################################################################################
# # # Effect of ground-truth curation on VPR evaluation.
# matching = 'single-match'
# num_th = 100
# GT_ = GT_superpoint_lightglue
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)  

# method = 'megaloc'
# GT_ = GT
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
#                                     marker='o', color=colors[0], line_style = '-' , label=method, 
#                                     fig = fig, ax = ax, paper_settings = paper_settings)

# method = 'megaloc'
# GT_ = GT_orb_knn
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
#                                     marker='o', color=colors[1], line_style = '-' , label=method, 
#                                     fig = fig, ax = ax, paper_settings = paper_settings)


# method = 'megaloc'
# GT_ = GT_sift_knn
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
#                                     marker='o', color=colors[2], line_style = '-' , label=method, 
#                                     fig = fig, ax = ax, paper_settings = paper_settings)

# method = 'megaloc'
# GT_ = GT_superpoint_lightglue
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, 
#                                     marker='o', color=colors[3], line_style = '-' , label=method, 
#                                     fig = fig, ax = ax, paper_settings = paper_settings)
# print(precisions)
# plt.show()

#########################################################################################################################
# # Plot D_0, D_90, D_180, D_270 and Dc for one method
# method = 'salad_Rot'
# matching = 'multi-match'
# num_th = 100
# GT_ = GT_superpoint_lightglue
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111)  

# for i, rotation in enumerate(rotations):
#     precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                             D = D[method][rotation], matching=matching, num_th=num_th)
#     fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[i], label=rotation, fig = fig, ax = ax)
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[i+1], label='Dc', fig = fig, ax = ax)

#########################################################################################################################

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D['salad']['D_0'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='*', color=colors[i+2], label='D inv_0', fig = fig, ax = ax)

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D['salad']['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[i+2], label='D inv_c', fig = fig, ax = ax)

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D['megaloc']['D_0'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[i+3], label='D mega_0', fig = fig, ax = ax)

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_, M = M, r_gt = r_gt, 
#                                                                         D = D['megaloc']['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[i+3], label='D mega_c', fig = fig, ax = ax)

# method = methods[0]
# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_superpoint_lightglue, M = M, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[1], label='Dc', fig = fig, ax = ax)

# precisions, recalls, thresholds, stats_at_rand = precision_recall_curve(GT = GT_superpoint_lightglue, M = M_I, r_gt = r_gt, 
#                                                                         D = D[method]['Dc'], matching=matching, num_th=num_th)
# fig, ax = plot_precision_recall_curve(precisions, recalls, thresholds, stats_at_rand, marker='o', color=colors[2], label='Dc_I', fig = fig, ax = ax)

#ax.set_xlim(0, 5000) 
#plt.show()
#################

########################################################################################################################################
#plot_clickable_GT(GT = GT_orb_knn, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min')
#plot_clickable_GT(GT = GT_sift_knn, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min')
# plot_clickable_GT(GT = D, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min')
#plot_clickable_GT(GT = GT, db_df = db_df, query_df = query_df, gt_db = gt_db, gt_query = gt_query, radius_px=10, pick='min')
#################

########################################################################################################################################
#compare_GT(GT_orb_knn, GT_sift_knn, "ORB + KNN", "SIFT + KNN")
# compare_GT(GT_orb_knn, GT_superpoint_lightglue, "ORB + KNN", "Superpoint + LightGlue")
#compare_GT(GT_sift_knn, GT_superpoint_lightglue, "SIFT + KNN", "Superpoint + LightGlue")

# GT_r = np.where(GT < 30, GT, np.inf)
# GT_r = np.where(M, GT_r, np.inf)
# GT_superpoint_lightglue = np.where(M, GT_superpoint_lightglue, np.inf)
# GT_sift_knn = np.where(M, GT_sift_knn, np.inf)
# GT_orb_knn = np.where(M, GT_orb_knn, np.inf)

#compare_GT(GT_r, GT_superpoint_lightglue, "GT_r", "Superpoint + LightGlue")
#compare_GT(GT_r, GT_r, "GT", "Superpoint + LightGlue")
#compare_GT(GT_r, GT_superpoint_lightglue, "GT", "Superpoint + LightGlue")
#compare_GT(GT_sift_knn, GT_superpoint_lightglue, "GT_sift_knn", "Superpoint + LightGlue")
#compare_GT(GT_orb_knn, GT_superpoint_lightglue, "GT_orb_knn", "Superpoint + LightGlue")

################
