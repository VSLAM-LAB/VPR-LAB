import numpy as np
import matplotlib.pyplot as plt
import math

def get_GT_P(GT, M, r_gt):
    # Binary ground-truth positives
    GT_P = (GT <= r_gt) & M 

    # total number of positive entries
    num_GT_P_multi = np.sum(GT_P)  

    # number of columns with at least one positive
    num_GT_P_single = np.sum(np.any(GT_P, axis=0))

    return GT_P, num_GT_P_multi, num_GT_P_single

def get_P(th_d, D = 0, S = 0, M = 0, GT_P = 0, matching='multi-match'): 

    if S == 0: # Use distance matrix:
        if matching == 'multi-match':
            P =(D <= th_d) & M
            num_P = np.sum(P)   
        elif 'single-match' in matching:
            P = np.zeros_like(D, dtype=bool)
            num_P = 0
            for col in range(D.shape[1]):
                col_distances = D[:, col]
                col_mask = M[:, col]
                valid_indices = np.where(col_mask)[0]
                if len(valid_indices) == 0:
                    continue
                
                if not ('hard' in matching):
                    col_GT_P = GT_P[:, col]
                    if np.sum(col_GT_P) == 0:
                        continue

                valid_distances = col_distances[valid_indices]
                min_index = valid_indices[np.argmin(valid_distances)]
                d = valid_distances.min()
                    
                if d <= th_d:
                    num_P += 1
                    P[min_index][col] = 1

    return P, num_P

def get_TP(P, GT_P):
    TP = P & GT_P
    num_TP = np.sum(TP)
    return TP, num_TP

def get_precision(num_TP, num_P):
    precision = num_TP / num_P if num_P > 0 else 0
    return precision

def get_recall(num_TP, num_GT_P):
    recall = num_TP / num_GT_P if num_GT_P > 0 else 0
    #recall = num_TP #/ num_GT_P if num_GT_P > 0 else 0
    return recall

def precision_recall_curve(GT, M, r_gt, D = 0, S = 0, matching='multi-match', num_th=100):
    # GT: ground truth matrix (db x q)
    # M: mask matrix (db x q)
    # r_gt: ground truth radius
    # D: distance matrix (db x q)
    # S: similarity matrix (db x q)
    # matching: 'multi-match' or 'single-match'
    # num_th: number of thresholds to evaluate

    precisions = np.zeros(num_th)
    recalls = np.zeros(num_th)

    GT_P, num_GT_P_multi, num_GT_P_single = get_GT_P(GT = GT, M = M, r_gt = r_gt)

    if S == 0: # Use distance matrix:
        min_d = np.min(D)
        max_d = np.max(D[np.isfinite(D)])
        thresholds = np.linspace(min_d, max_d, num_th)

        # --- Stats of count of ones per column ---
        col_counts = np.sum(GT_P, axis=0)   # number of 1s in each column
        stats_at_rand = {
            "col_min": np.min(col_counts)/ (GT.shape[0]),
            "col_max": np.max(col_counts)/ (GT.shape[0]),
            "col_mean": np.mean(col_counts) / (GT.shape[0]),
            "col_std": np.std(col_counts),
        }
        
        for index, th_d in enumerate(thresholds):
            P, num_P = get_P(th_d = th_d, D = D, S = S, M = M, GT_P = GT_P, matching=matching)     
            _, num_TP = get_TP(P = P, GT_P = GT_P)         
                
            precisions[index] = get_precision(num_TP = num_TP, num_P = num_P)
            if matching == 'multi-match':
                recalls[index] = get_recall(num_TP = num_TP, num_GT_P = num_GT_P_multi)
            elif 'single-match' in matching:
                recalls[index] = get_recall(num_TP = num_TP, num_GT_P = num_GT_P_single)

    return precisions, recalls, thresholds, stats_at_rand

def recall_at_k(GT, M, r_gt, D = 0, S = 0):
    k_values = [1, 5, 10, 20, 50, 100]
    masked_D = np.where(M, D, np.inf)
    GT_P, _, num_GT_P_single = get_GT_P(GT = GT, M = M, r_gt = r_gt)
    recall_at_k = []
    for k in k_values:
        indices = []
        values = []
        gt_match = []
        for j in range(masked_D.shape[1]):
            col = masked_D[:, j]
            valid_count = np.isfinite(col).sum()
            k_eff = min(k, valid_count) 
            if k_eff > 0:
                top_idx = np.argpartition(col, k_eff-1)[:k_eff]
                sorted_idx = top_idx[np.argsort(col[top_idx])]
                indices.append(sorted_idx)
                values.append(col[sorted_idx])
                gt_match.append(np.any(GT_P[sorted_idx, j] == 1))
            else:
                indices.append(np.array([], dtype=int))
                values.append(np.array([], dtype=float))
                gt_match.append(False)
        recall_at_k.append(sum(gt_match) / num_GT_P_single)

    n_queries = GT.shape[1]
    avg_probs = []
    for k in k_values:
        probs = []
        for j in range(n_queries):
            N = int(M[:, j].sum())       # total valid candidates
            P = int(GT_P[:, j].sum())    # number of positives
            if P == 0 or N == 0:
                continue  # skip queries with no positives
            elif k > N - P:
                p = 1.0
            else:
                num = math.comb(N - P, k)
                den = math.comb(N, k)
                p = 1 - num / den
            probs.append(p)
        avg_probs.append(np.mean(probs) if len(probs) > 0 else 0.0)


    return k_values, recall_at_k, np.array(avg_probs)
