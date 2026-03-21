import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utilities import load_image_list_with_absolute_paths
from utilities import load_yaml

from plot_functions.clickable_plots import plot_clickable_matrix

parser = argparse.ArgumentParser(description="Generate mask matrix from experiment config.")
parser.add_argument("--exp_yaml", type=str, default="exp.yaml", help="Path to experiment YAML config file")

args = parser.parse_args()

 # Inputs
exp_yaml = load_yaml(args.exp_yaml)

path_to_gt_file_db = exp_yaml['gt_file_db']
path_to_gt_file_q = path_to_gt_file_db
path_to_rgb_list_db = exp_yaml['rgb_list_db']
path_to_rgb_list_q = path_to_rgb_list_db
path_to_rgb_folder_db = Path(exp_yaml['rgb_folder_db']).parent
path_to_rgb_folder_q = path_to_rgb_folder_db

path_to_D = exp_yaml['log_dir'] + f"/D_0.npy"

db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_q, path_to_rgb_folder_db, path_to_rgb_folder_q)
D = np.load(path_to_D)

plot_clickable_matrix(matrix = D, db_df = db_df, query_df = query_df, lightglue_matching_func=True)


import shutil
import os
import numpy as np
import matplotlib.pyplot as plt

results = []
all_indexes = {}

# Sampling
for th in np.linspace(0.1, 1.0, num=50):
    i = 0
    j = 0
    indexes = [0]
    while i < D.shape[0] and j < D.shape[1]:
        while j < D.shape[1]:
            D_value = D[i, j]
            if D_value > th:
                indexes.append(j-1)
                i = j
            j += 1
    results.append((th, len(indexes)))
    all_indexes[th] = indexes

ths, lengths = zip(*results)

# Define source and destination folder pairs
base_dir = "/home/alejandro/BommieToolkit/colmap_ws/colmap_images/rig1"
cameras = ["rgb_0", "camera2"]  # update as needed

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ths, lengths, marker='o', markersize=3)
ax.set_xlabel('Threshold (th)')
ax.set_ylabel('len(indexes)')
ax.set_title('Click on the plot to select a threshold')
ax.grid(True)

def on_click(event):
    if event.inaxes != ax:
        return
    clicked_th = event.xdata
    nearest_th = min(all_indexes.keys(), key=lambda t: abs(t - clicked_th))
    indexes = all_indexes[nearest_th]

    print(f"\nSelected threshold: {nearest_th:.4f} → {len(indexes)} images")

    # Show the subsampled D matrix
    D_filtered = D[np.ix_(indexes, indexes)]

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(D, cmap='viridis', aspect='auto')
    axes[0].set_title('Original D matrix')
    axes[0].set_xlabel('j')
    axes[0].set_ylabel('i')
    plt.colorbar(axes[0].images[0], ax=axes[0])

    axes[1].imshow(D_filtered, cmap='viridis', aspect='auto')
    axes[1].set_title(f'Filtered D matrix (th={nearest_th:.2f}, {len(indexes)} frames)')
    axes[1].set_xlabel('j')
    axes[1].set_ylabel('i')
    plt.colorbar(axes[1].images[0], ax=axes[1])

    plt.tight_layout()
    plt.show()

    # Copy images
    for camera in cameras:
        src_dir = os.path.join(base_dir, camera)
        dst_dir = os.path.join(base_dir, f"{camera}_filtered")
        os.makedirs(dst_dir, exist_ok=True)

        all_images = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        selected_images = [all_images[i] for i in indexes if i < len(all_images)]
        for fname in selected_images:
            shutil.copy(os.path.join(src_dir, fname), dst_dir)
        print(f"Copied {len(selected_images)} images to {dst_dir}")

    plt.close(fig)

fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()