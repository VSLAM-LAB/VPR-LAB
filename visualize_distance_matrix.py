
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilities import show_clickable_distance_matrix
from utilities import load_yaml

# Inputs
exp_yaml = load_yaml("exp.yaml")

# Experiment paths
gt_db_path = exp_yaml["gt_db_path"]
gt_query_path = exp_yaml["gt_query_path"]
rgb_folder_db_path = exp_yaml["rgb_folder_db_path"]
rgb_folder_query_path = exp_yaml["rgb_folder_query_path"]
rgb_txt_db_path = exp_yaml["rgb_txt_db_path"]
rgb_txt_query_path = exp_yaml["rgb_txt_query_path"]
distance_matrix_path = exp_yaml["distance_matrix_path"]
gt_matrix_path = exp_yaml["gt_matrix_path"]
output_folder = exp_yaml["output_folder"]


db_df = pd.read_csv(rgb_txt_db_path, sep=' ', header=None)
query_df = pd.read_csv(rgb_txt_query_path, sep=' ', header=None)
db_df[1] = db_df[1].apply(lambda x: os.path.join(rgb_folder_db_path, x))
query_df[1] = query_df[1].apply(lambda x: os.path.join(rgb_folder_query_path, x))

distance_matrix = np.load(distance_matrix_path)
gt_matrix = np.load(gt_matrix_path)

# # Plot distance matrix heatmap
# plt.figure(figsize=(8, 6))
# plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
# plt.colorbar(label="Distance")
# plt.xlabel("Query Descriptors")
# plt.ylabel("Database Descriptors")
# plt.title("Pairwise Distance Matrix")

# plt.tight_layout()
# plt.show()

show_clickable_distance_matrix(distance_matrix, query_df=query_df, db_df=db_df)
#show_clickable_distance_matrix(gt_matrix, query_df=query_df, db_df=db_df)

# Flatten to 1D arrays
# x = distance_matrix.flatten()
# y = gt_matrix.flatten()

# plt.figure(figsize=(6, 6))
# plt.scatter(x, y, s=5, alpha=0.5)
# plt.xlabel("Distance Matrix Values")
# plt.ylabel("Ground Truth Matrix Values")
# plt.title("Distance vs Ground Truth")
# plt.grid(True)
# plt.show()