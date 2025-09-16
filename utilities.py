import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import yaml

import os
import pandas as pd
import cv2

def make_absolute_paths(db_df, query_df, path_to_rgb_folder_db, path_to_rgb_folder_query):
    db_df[1] = db_df[1].apply(lambda x: os.path.join(path_to_rgb_folder_db, x))
    query_df[1] = query_df[1].apply(lambda x: os.path.join(path_to_rgb_folder_query, x))

    return db_df, query_df

def load_image_list(path_to_rgb_list_db, path_to_rgb_list_query):
    db_df = pd.read_csv(path_to_rgb_list_db, sep=' ', header=None)
    query_df = pd.read_csv(path_to_rgb_list_query, sep=' ', header=None)
    
    return db_df, query_df

def load_image_list_with_absolute_paths(path_to_rgb_list_db, path_to_rgb_list_query, path_to_rgb_folder_db, path_to_rgb_folder_query):

    db_df, query_df = load_image_list(path_to_rgb_list_db, path_to_rgb_list_query)
    db_df, query_df = make_absolute_paths(db_df, query_df, path_to_rgb_folder_db, path_to_rgb_folder_query)

    return db_df, query_df

def load_trajectory(path_to_gt_file):
    col_names = ["ts", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    gt = pd.read_csv(path_to_gt_file, sep=r"\s+", header=None, names=col_names)
    gt_df = gt[["tx","ty","tz"]].to_numpy(dtype=float) 
    
    return gt_df



def load_yaml(path: str = "exp.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p.resolve()}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}

def unrotate_kps(kps_rot, k, H, W):
    import torch
    # k is how many times you rotated CCW by 90° to create the rotated image
    x_r, y_r = kps_rot[:, 0].clone(), kps_rot[:, 1].clone()
    if k == 0:
        x, y = x_r, y_r
    elif k == 1:   # 90° CCW
        x = (W - 1) - y_r
        y = x_r
    elif k == 2:   # 180°
        x = (W - 1) - x_r
        y = (H - 1) - y_r
    elif k == 3:   # 270° CCW
        x = y_r
        y = (H - 1) - x_r
    else:
        raise ValueError("k must be 0..3")
    return torch.stack([x, y], dim=-1)

def lightglue_matching(path_to_image0, path_to_image1, plot=False, features='superpoint'):
    from lightglue import LightGlue, SuperPoint, SIFT
    from lightglue.utils import load_image, rbd
    from lightglue import viz2d
    import torch

    # --- Models on GPU ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if features == 'superpoint':
        extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    if features == 'sift':
        extractor = SIFT(max_num_keypoints=2048).eval().to(device)

    matcher = LightGlue(features=features).eval().to(device)

    # --- Load images as Torch tensors (3,H,W) in [0,1] ---
    timg0 = load_image(path_to_image0).to(device)
    timg1 = load_image(path_to_image1).to(device)

    # --- Extract local features ---
    feats0 = extractor.extract(timg0)   # auto-resize inside

    max_num_matches = -1
    best_k = 0
    best_feats0 = None
    best_feats1 = None
    for k in range(4):
        timg1_rotated = torch.rot90(timg1, k, dims=(1, 2))
        feats1_k = extractor.extract(timg1_rotated)
        out_k = matcher({'image0': feats0, 'image1': feats1_k})
        feats0_k, feats1_k, out_k = [rbd(x) for x in [feats0, feats1_k, out_k]]   # remove batch dim
        matches_k = out_k['matches']   # (K,2) long
        num_k = len(matches_k)
        if num_k > max_num_matches:
            max_num_matches = num_k
            matches = matches_k
            best_feats0 = feats0_k
            best_feats1 = feats1_k
            best_k = k
    
    # --- Keypoints in matched order (Torch tensors on CPU) ---
    H1, W1 = timg1.shape[-2], timg1.shape[-1]

    kpts0 = best_feats0['keypoints'][matches[:, 0]]
    kpts1 = best_feats1['keypoints'][matches[:, 1]]
    kpts1 = unrotate_kps(kpts1, best_k, H1, W1)             # (K,2) mapped to original image1 coords

    if plot:
        if len(kpts0) == 0 or len(kpts1) == 0:
            print("No matches found.")
            return None, None
        ax = viz2d.plot_images([timg0.cpu(), timg1.cpu()])
        viz2d.plot_matches(kpts0.cpu(), kpts1.cpu(), color=None, lw=0.8, axes=ax)
        #ax0 = ax[0] if isinstance(ax, (list, tuple, np.ndarray)) else ax
        #fig = ax0.figure

        return kpts0, kpts1 #, fig, ax
        

    return kpts0, kpts1

def shannon_entropy(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
    p = hist / (gray.size + 1e-9)
    p = p[p>0]
    return float(-(p*np.log2(p)).sum())

def mutual_information(img1, img2):
    """
    MI(img1, img2) in bits for uint8 images of the same size.
    Uses a single bincount over 256*256 bins (very fast).
    """
    a = img1.ravel().astype(np.int32)
    b = img2.ravel().astype(np.int32)
    N = a.size
    # linear joint index: (a << 8) + b  == a*256 + b
    lin = (a << 8) + b
    joint = np.bincount(lin, minlength=256*256).reshape(256, 256)

    # Marginals
    cx = joint.sum(axis=1)
    cy = joint.sum(axis=0)

    # Helper: sum c*log(c) with 0*log(0)=0
    def clogc(x):
        x = x.astype(np.float64, copy=False)
        nz = x > 0
        return (x[nz] * np.log(x[nz])).sum()

    Nf = float(N)
    mi_nats = (clogc(joint) - clogc(cx) - clogc(cy) + Nf*np.log(Nf)) / Nf
    return mi_nats / np.log(2.0)  # bits

def create_Shannon(db_df, query_df):
    entropies_db = np.zeros(len(db_df))
    entropies_query = np.zeros(len(query_df))
    I = np.zeros((len(db_df), len(query_df)))

    for row in range(len(db_df)):
        db_img_path = db_df.iloc[row][1]
        print(f"DB image path: {db_img_path}")
        db_img = plt.imread(db_img_path)
        
        if len(db_img.shape) == 3:
            gray = cv2.cvtColor(db_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = db_img
        
        shannon_entropy_db = shannon_entropy(gray)
        entropies_db[row] = shannon_entropy_db
        print(f"Shannon Entropy for DB image {row}: {shannon_entropy_db}")

    for row in range(len(query_df)):
        query_img_path = query_df.iloc[row][1]
        print(f"Query image path: {query_img_path}")
        query_img = plt.imread(query_img_path)
        
        if len(query_img.shape) == 3:
            gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = query_img
        
        shannon_entropy_query = shannon_entropy(gray)
        entropies_query[row] = shannon_entropy_query
        print(f"Shannon Entropy for Query image {row}: {shannon_entropy_query}")

    for row in range(len(db_df)):
        for col in range(len(query_df)):
            #I[row, col] = entropies_db[row] + entropies_query[col]
            I[row, col] = min(entropies_db[row], entropies_query[col])
    # I = np.zeros((len(db_df), len(query_df)))
    # for row in range(len(db_df)):
    #     db_img_path = db_df.iloc[row][1]
    #     db_img = plt.imread(db_img_path)
    #     print(row)
    #     for col in range(len(query_df)):
    #         #print(col)
    #         query_img_path = query_df.iloc[col][1]
    #         query_img = plt.imread(query_img_path)
    #         I[row, col] = mutual_information(db_img, query_img)

    I_min = np.min(I)
    I_max = np.max(I)
    I_median = np.median(I)
    I_std = np.std(I)
    #I_bin = (I >= (I_median - 0.1*(I_median-I_min))).astype(int)
    #I_bin = (I >= (I_median)).astype(int)
    I_bin = {}

    # for idx, val in enumerate(np.linspace(1, 0.5, 50)):
    #     I_bin[idx] = (I <= (I_median - val*(I_median-I_min))).astype(int)
   
    return I, I_bin
    # entropies = shannon_entropy(gray)
    # db_img_path = db_df.iloc[row][1]
    #             print(f"DB image path: {db_img_path}")
    #             db_img = plt.imread(db_img_path)


def combine_distance_matrix(D_0, D_90, D_180, D_270, mode ='0', n=2):

    #
    if mode == '0':
        D = np.minimum.reduce([D_0, D_90, D_180, D_270])

    #
    if mode == '1':
        stacked = np.stack([D_0, D_90, D_180, D_270], axis=-1)
        mins = np.min(stacked, axis=-1)
        sorted_vals = np.sort(stacked, axis=-1)
        mask = mins < sorted_vals[...,1] / n
        D = np.where(mask, mins, np.inf)
    if mode == '2':
        stacked = np.stack([D_0, D_90, D_180, D_270], axis=-1)
        D = -1/n * np.log(np.sum(np.exp(-n * stacked), axis=-1))

    return D

def generate_definecolor_lines(methods, colors):
    lines = []
    for method, c in zip(methods, colors):
        hex_code = mcolors.to_hex(c, keep_alpha=False).lstrip("#").upper()
        lines.append(f"\\definecolor{{{method}}}{{HTML}}{{{hex_code}}}")
    return "\n".join(lines)

def generate_multicolumn_line(methods, methods_names, ncols=4):
    entries = [f"\\myline{{{m}}} {name}" for m, name in zip(methods, methods_names)]
    joined = r"\quad ".join(entries)   # build outside the f-string
    return f"\\multicolumn{{{ncols}}}{{c}}{{{joined}}} \\\\"