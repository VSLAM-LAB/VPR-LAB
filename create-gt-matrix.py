
import argparse
import cv2
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from utilities import load_yaml
from utilities import load_trajectory
from utilities import load_image_list_with_absolute_paths
from utilities import lightglue_matching
from plot_functions.plot_functions import plot_clickable_GT

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "

def create_GT(gt_db, gt_query, metric = 'L2', 
              filter = None, 
              db_df = None, query_df = None ):
    print('Create GT', f'(metric : {metric})')
    if metric == 'L2':
        GT = np.linalg.norm(gt_db[:, None, :] - gt_query[None, :, :], axis=2)

    if filter == None:
        return GT
    
    min_num_inliers = filter['min_num_inliers']
    r_gt = filter['r_gt']
    ransac_thresh = filter['ransac_thresh']

    print(f"Filter GT (class: {filter['class']}, min_num_inliers: {filter['min_num_inliers']}, r_gt: {filter['r_gt']}, ransac_thresh: {filter['ransac_thresh']})")

    if filter['class'] == 'superpoint_lightglue' or filter['class'] == 'sift_lightglue':    
        for row in tqdm(range(GT.shape[0])):
            for col in range(GT.shape[1]):
                if GT[row, col] < r_gt: 
                    q_img_path = query_df.iloc[col][1] if 0 <= col < len(query_df) else None
                    db_img_path = db_df.iloc[row][1] if 0 <= row < len(db_df) else None

                    if filter['class'] == 'superpoint_lightglue':
                        kpts0, kpts1 = lightglue_matching(q_img_path, db_img_path, plot=False, features='superpoint')
                    if filter['class'] == 'sift_lightglue':
                        kpts0, kpts1 = lightglue_matching(q_img_path, db_img_path, plot=False, features='sift')

                    pts0 = kpts0.detach().cpu().numpy().astype(np.float32) 
                    pts1 = kpts1.detach().cpu().numpy().astype(np.float32)
                   
                    if len(pts0) < min_num_inliers or len(pts1) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue

                    _, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thresh) 
                    mask = inliers.ravel().astype(bool)
                    pts0 = pts0[mask]
                    pts1 = pts1[mask]
                    if len(pts0) < min_num_inliers or len(pts1) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue
                else:
                    GT[row, col] = np.inf

    if filter['class'] == 'orb_knn' or filter['class'] == 'sift_knn':
        n_features=2048 #4000
        ratio=0.75 

        if filter['class'] == 'orb_knn':
            feat = cv2.ORB_create(nfeatures=n_features)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if filter['class'] == 'sift_knn':  
            feat = cv2.SIFT_create(nfeatures=n_features)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)             

        for row in tqdm(range(GT.shape[0])):
            for col in range(GT.shape[1]):
                if GT[row, col] < r_gt: 
                    q_img_path = query_df.iloc[col][1] if 0 <= col < len(query_df) else None
                    db_img_path = db_df.iloc[row][1] if 0 <= row < len(db_df) else None
                    img1 = cv2.imread(q_img_path)
                    img2 = cv2.imread(db_img_path)
                    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    
                    k1, d1 = feat.detectAndCompute(g1, None)
                    k2, d2 = feat.detectAndCompute(g2, None)

                    if d1 is None or d2 is None or len(k1) < min_num_inliers or len(k2) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue
                    
                    knn = bf.knnMatch(d1, d2, k=2)

                    good = []
                    for pair in knn:
                        if len(pair) < 2:
                            continue
                        m, n = pair
                        if m.distance < ratio * n.distance:
                            good.append(m)

                    if len(good) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue

                    pts1 = np.float32([k1[m.queryIdx].pt for m in good]) 
                    pts2 = np.float32([k2[m.trainIdx].pt for m in good])  

                    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)

                    if mask is None:
                        GT[row, col] = np.inf
                        continue
                    
                    inliers = int(mask.sum())
                    if inliers < min_num_inliers:
                        GT[row, col] = np.inf
                        continue
                else:
                    GT[row, col] = np.inf
    return GT

def compare_GT(GT_1, GT_2, label1, label2):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    # Convert to binary using your rule (valid entries are != inf)
    A = (GT_1 != np.inf)
    B = (GT_2 != np.inf)

    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")

    # Masks
    overlap = A & B               # green
    only_A  = A & ~B              # red
    only_B  = ~A & B              # blue

    H, W = A.shape
    rgb = np.ones((H, W, 3), dtype=float)   # start with white background (1,1,1)

    # rgb[only_A]   = (0.9, 0.5, 0.5)   # soft red
    # rgb[only_B]   = (0.5, 0.5, 0.9)   # soft blue
    # rgb[overlap]  = (0.5, 0.9, 0.5)   # soft green

    rgb[only_A]   = (0.9, 0.5, 0.5)   # soft red
    rgb[only_B]   = (0.9,0.7,0.3)   # soft blue
    rgb[overlap]  = (0.5,0.9,0.5)   # soft green

    plt.figure(figsize=(8, 6))
    plt.imshow(rgb, aspect="auto", interpolation="nearest")
    #plt.xlabel("Query")
    #plt.ylabel("Database")

    # put x-axis on top
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # hide the first tick label (usually 0) on x-axis only
    labels = [item.get_text() for item in ax.get_xticklabels()]
    if labels:
        labels[1] = ""   # blank out the first one
    ax.set_xticklabels(labels)
    #ax.set_xticklabels([])
    ax.set_yticklabels([])

    # remove outer box (spines)
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    # legend
    legend_patches = [
        # Patch(color=(0.9,0.5,0.5), label=f"{label1} only"),
        # Patch(color=(0.5,0.9,0.5), label="Overlap"),
        # Patch(color=(0.5,0.5,0.9), label=f"{label2} only"),
        Patch(color=(0.9,0.5,0.5), label=f"{label1} only"),
        Patch(color=(0.9,0.5,0.5), label="Overlap"),
        Patch(color=(0.9,0.5,0.5), label=f"{label2} only"),
    ]

    #plt.legend(handles=legend_patches, loc="upper right", frameon=True)
    plt.tight_layout()
    plt.show()


def main():

    parser = argparse.ArgumentParser(description="Generate ground truth matrix from experiment config.")
    parser.add_argument("--gt_file_db", type=str, required=True, help="Path to gt_file_db")
    parser.add_argument("--gt_file_q", type=str, required=True, help="Path to gt_file_q")
    parser.add_argument("--rgb_list_db", type=str, required=True, help="Path to rgb_list_db")
    parser.add_argument("--rgb_list_q", type=str, required=True, help="Path to rgb_list_q")
    parser.add_argument("--rgb_folder_db", type=str, required=True, help="Path to rgb_folder_db")
    parser.add_argument("--rgb_folder_q", type=str, required=True, help="Path to rgb_folder_q")
    parser.add_argument("--gt_matrix", type=str, required=True, help="Path to gt_matrix")

    #parser.add_argument("--exp_yaml", type=str, default="exp.yaml", help="Path to experiment YAML config file")
    #parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Inputs
    #print(f"\n{SCRIPT_LABEL}: {args.exp_yaml}")
    #exp_yaml = load_yaml(args.exp_yaml)
    
    # Experiment paths
    path_to_gt_file_db = args.gt_file_db 
    path_to_gt_file_q = args.gt_file_q 
    path_to_rgb_list_db = args.rgb_list_db
    path_to_rgb_list_q = args.rgb_list_q
    path_to_rgb_folder_db = args.rgb_folder_db
    path_to_rgb_folder_q = args.rgb_folder_q
    print(f'path_to_gt_file_db : {path_to_gt_file_db}')
    print(f'path_to_gt_file_q : {path_to_gt_file_q}')
    print(f'path_to_rgb_list_db : {path_to_rgb_list_db}')
    print(f'path_to_rgb_list_q : {path_to_rgb_list_q}')
    print(f'path_to_rgb_folder_db : {path_to_rgb_folder_db}')
    print(f'path_to_rgb_folder_q : {path_to_rgb_folder_q}')   

    GT_path = args.gt_matrix
    print(f'GT_path : {GT_path}')

    # # Load image lists
    db_df, query_df = load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_q, path_to_rgb_folder_db, path_to_rgb_folder_q)
    gt_db = load_trajectory(path_to_gt_file_db)
    gt_query = load_trajectory(path_to_gt_file_q)

    # Create GT
    GT = create_GT(gt_db = gt_db, gt_query = gt_query, metric = 'L2', filter = None, db_df = db_df, query_df = query_df)
    np.save(GT_path, GT)  

    # filter = {}
    # filter['min_num_inliers'] = 30
    # filter['ransac_thresh'] = 5
    # filter['r_gt'] = exp_yaml['r_gt']

    # # filter['class'] = 'orb_knn'
    # # GT = create_GT(gt_db = gt_db, gt_query = gt_query, metric = 'L2', filter = filter, db_df = db_df, query_df = query_df)
    # # np.save(str.replace(GT_path, 'GT', f"GT_{filter['class']}"), GT)  

    # # filter['class'] = 'sift_knn'
    # # GT = create_GT(gt_db = gt_db, gt_query = gt_query, metric = 'L2', filter = filter, db_df = db_df, query_df = query_df)
    # # np.save(str.replace(GT_path, 'GT', f"GT_{filter['class']}"), GT)  

    # filter['class'] = 'superpoint_lightglue'
    # GT = create_GT(gt_db = gt_db, gt_query = gt_query, metric = 'L2', filter = filter, db_df = db_df, query_df = query_df)
    # np.save(str.replace(GT_path, 'GT', f"GT_{filter['class']}"), GT) 

    # # filter['class'] = 'sift_lightglue'
    # # GT = create_GT(gt_db = gt_db, gt_query = gt_query, metric = 'L2', filter = filter, db_df = db_df, query_df = query_df)
    # # np.save(str.replace(GT_path, 'GT', f"GT_{filter['class']}"), GT) 

if __name__ == "__main__":
    main()