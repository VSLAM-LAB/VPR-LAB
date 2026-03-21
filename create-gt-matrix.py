import argparse
import os
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm
from utilities import (
    lightglue_matching,
    load_image_list_with_absolute_paths,
    load_trajectory,
    load_yaml,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def create_GT(gt_db, gt_query, metric="L2", filter=None, db_df=None, query_df=None):
    print("Create GT", f"(metric : {metric})")
    if metric == "L2":
        GT = np.linalg.norm(gt_db[:, None, :] - gt_query[None, :, :], axis=2)

    if filter == None:
        return GT

    min_num_inliers = filter["min_num_inliers"]
    r_gt = filter["r_gt"]
    ransac_thresh = filter["ransac_thresh"]

    print(
        f"Filter GT (class: {filter['class']}, min_num_inliers: {filter['min_num_inliers']}, r_gt: {filter['r_gt']}, ransac_thresh: {filter['ransac_thresh']})"
    )

    if filter["class"] == "superpoint_lightglue" or filter["class"] == "sift_lightglue":
        for row in tqdm(range(GT.shape[0])):
            for col in range(GT.shape[1]):
                if GT[row, col] < r_gt:
                    q_img_path = (
                        query_df.iloc[col][1] if 0 <= col < len(query_df) else None
                    )
                    db_img_path = db_df.iloc[row][1] if 0 <= row < len(db_df) else None

                    if filter["class"] == "superpoint_lightglue":
                        kpts0, kpts1 = lightglue_matching(
                            q_img_path, db_img_path, plot=False, features="superpoint"
                        )
                    if filter["class"] == "sift_lightglue":
                        kpts0, kpts1 = lightglue_matching(
                            q_img_path, db_img_path, plot=False, features="sift"
                        )

                    pts0 = kpts0.detach().cpu().numpy().astype(np.float32)
                    pts1 = kpts1.detach().cpu().numpy().astype(np.float32)

                    if len(pts0) < min_num_inliers or len(pts1) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue

                    _, inliers = cv2.findHomography(
                        pts0, pts1, cv2.RANSAC, ransac_thresh
                    )
                    mask = inliers.ravel().astype(bool)
                    pts0 = pts0[mask]
                    pts1 = pts1[mask]
                    if len(pts0) < min_num_inliers or len(pts1) < min_num_inliers:
                        GT[row, col] = np.inf
                        continue
                else:
                    GT[row, col] = np.inf

    if filter["class"] == "orb_knn" or filter["class"] == "sift_knn":
        n_features = 2048  # 4000
        ratio = 0.75

        if filter["class"] == "orb_knn":
            feat = cv2.ORB_create(nfeatures=n_features)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        if filter["class"] == "sift_knn":
            feat = cv2.SIFT_create(nfeatures=n_features)
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        for row in tqdm(range(GT.shape[0])):
            for col in range(GT.shape[1]):
                if GT[row, col] < r_gt:
                    q_img_path = (
                        query_df.iloc[col][1] if 0 <= col < len(query_df) else None
                    )
                    db_img_path = db_df.iloc[row][1] if 0 <= row < len(db_df) else None
                    img1 = cv2.imread(q_img_path)
                    img2 = cv2.imread(db_img_path)
                    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    k1, d1 = feat.detectAndCompute(g1, None)
                    k2, d2 = feat.detectAndCompute(g2, None)

                    if (
                        d1 is None
                        or d2 is None
                        or len(k1) < min_num_inliers
                        or len(k2) < min_num_inliers
                    ):
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


def main():

    parser = argparse.ArgumentParser(
        description="Generate ground truth matrix from experiment config."
    )

    parser.add_argument(
        "--exp_yaml",
        type=str,
        default="exp.yaml",
        help="Path to experiment YAML config file",
    )
    args = parser.parse_args()

    # Inputs
    print(f"{SCRIPT_LABEL}: {args.exp_yaml}")
    exp_yaml = load_yaml(args.exp_yaml)

    # Experiment paths
    path_to_rgb_list_db = exp_yaml["rgb_list_db"]
    path_to_rgb_list_q = exp_yaml["rgb_list_q"]
    path_to_rgb_folder_db = Path(path_to_rgb_list_db).parent
    path_to_rgb_folder_q = Path(path_to_rgb_list_q).parent
    path_to_gt_file_db = path_to_rgb_folder_db / "groundtruth.csv"
    path_to_gt_file_q = path_to_rgb_folder_q / "groundtruth.csv"

    GT_path = Path(exp_yaml["log_dir"]) / "GT.npy"

    # Load image lists
    db_df, query_df = load_image_list_with_absolute_paths(
        path_to_rgb_list_db,
        path_to_rgb_list_q,
        path_to_rgb_folder_db,
        path_to_rgb_folder_q,
    )
    gt_db = load_trajectory(path_to_gt_file_db)
    gt_query = load_trajectory(path_to_gt_file_q)

    # Create GT
    GT = create_GT(
        gt_db=gt_db,
        gt_query=gt_query,
        metric="L2",
        filter=None,
        db_df=db_df,
        query_df=query_df,
    )
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
