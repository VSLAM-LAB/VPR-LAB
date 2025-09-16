import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from evaluation_functions import get_GT_P, get_P, get_TP

# import seaborn
# from matplotlib import rc
# rc('text', usetex=True)
# seaborn.set(font_scale = 1.3)  # adjust so it roughly matches the caption font size

# import matplotlib as mpl
# mpl.rcParams["pgf.texsystem"] = "pdflatex"
# mpl.rcParams["text.usetex"] = True
# print(mpl.rcParams["text.usetex"])
# print(mpl.rcParams["pgf.texsystem"])

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def overlay_distance_gt_mask(D, GT, M, k=5, r_gt=0.5):
    """
    Overlay between distance matrix D, ground truth GT (float), and mask M.

    Args:
        D   : 2D numpy array (distance matrix)
        GT  : 2D numpy array (float distances, same shape as D)
        M   : 2D numpy array (binary mask, same shape as D)
        k   : number of lowest values per column to highlight
        r_gt: radius threshold, values <= r_gt are positives
    """
    # Normalize D for background grayscale
    D_norm = (D - D.min()) / (D.max() - D.min() + 1e-9)
    img = plt.cm.gray(1 - D_norm)[:, :, :3]  # invert so low distances are bright

    # Mask out areas with black
    img[M == 0] = [0, 0, 0]

    # For each column, find lowest-k distances (ignoring masked)
    for j in range(D.shape[1]):
        col = D[:, j].copy()
        col[M[:, j] == 0] = np.inf  # ignore masked entries
        idx = np.argpartition(col, k)[:k]  # indices of lowest k

        for i in idx:
            if M[i, j] == 0:
                continue
            if GT[i, j] <= r_gt:
                img[i, j] = [0, 1, 0]  # green = positive match
            else:
                img[i, j] = [1, 0, 0]  # red = negative match

def plot_precision_recall_curve_clickable(
    precisions, recalls, thresholds, stats_at_rand,
    GT, M, r_gt, D=0, S=0, matching='multi-match',
    fig=None, ax=None, marker='o', color='b', label=None, paper_settings = None
):
    
    # LABELS FACTOR
    factor = 3.5
    plt.rcParams.update({
        "font.size": factor * 9,       # matches IEEE two-column body text
        "axes.titlesize": factor *9,
        "axes.labelsize": factor *9,
        "xtick.labelsize": factor *8,
        "ytick.labelsize": factor *8,
        "legend.fontsize": factor *8,
     })
               
    precisions = np.asarray(precisions, dtype=float)
    recalls    = np.asarray(recalls, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    print(len(thresholds))
    print(len(recalls))
    print(len(precisions))
    # Sort by recall for visualization (keep thresholds aligned)
    order = np.argsort(recalls)
    recalls     = recalls[order]
    precisions  = precisions[order]
    thresholds  = thresholds[order]

    mask = (recalls >= 0.001) & (precisions >= 0.001)
    recalls    = recalls[mask]
    precisions = precisions[mask]
    thresholds = thresholds[mask]

    # Marker size for GT heatmap overlays
    base_size = 20
    size_factor = 2.5 if matching == 'single-match' else 1.0
    heatmap_marker_size = base_size * size_factor

    # Create fig/ax if needed
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # PR curve
    ax.plot(recalls, precisions, marker=marker, color=color, label=label, lw=1.5)
    sc = ax.scatter(recalls, precisions, s=40, color=color)

    #ax.set_title('Single-Match Precision–Recall Curve')
    #ax.set_xlabel('Recall')
    #ax.set_ylabel('Precision')
    ax.grid(True, alpha=0.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)

    # if stats_at_rand is not None:
    #     mean = stats_at_rand.get("col_mean")
    #     vmin = stats_at_rand.get("col_min")
    #     vmax = stats_at_rand.get("col_max")
    #     if mean is not None:
    #         ax.axhline(y=float(mean), linestyle='--', linewidth=2,
    #                    label=f'Mean@Rand = {mean:.2f}', color=color)
    #     if vmin is not None:
    #         ax.axhline(y=float(vmin), linestyle=':', linewidth=1.5,
    #                    label=f'Min@Rand = {vmin:.2f}', color=color)
    #     if vmax is not None:
    #         ax.axhline(y=float(vmax), linestyle='-.', linewidth=1.5,
    #                    label=f'Max@Rand = {vmax:.2f}', color=color)
        #ax.legend(loc='lower left')

    # --- GT_P window handles ---
    gt_fig = gt_ax = gt_im = tp_scatter = fp_scatter = None

    # Selection marker on PR plot
    sel_marker = ax.scatter([], [], s=500, facecolors='none',
                            edgecolors='magenta', linewidths=8.5)

    def ensure_gt_window(first_GT_P):
        nonlocal gt_fig, gt_ax, gt_im, tp_scatter, fp_scatter
        if gt_fig is None:
            gt_fig, gt_ax = plt.subplots(figsize=(6, 5))
            #gt_ax.set_title('GT_P + TP + FP')

            # GT LABELS
            #gt_ax.set_xlabel('Query index')
            #gt_ax.set_ylabel('Database index')

            # base GT_P in gray
            from matplotlib.colors import ListedColormap

            # GT COLOR
            cmap = ListedColormap(["white", (0.5,0.5,0.9)])     
            #cmap = ListedColormap(["white", (0.9,0.7,0.3)])
            #cmap = ListedColormap(["white", (0.0,0.0,0.0)])

            # gt_im = gt_ax.imshow(first_GT_P.astype(int), cmap='Greys',
            #                      vmin=0, vmax=1, aspect='auto')
            gt_im = gt_ax.imshow(first_GT_P.astype(int), cmap=cmap,
                     vmin=0, vmax=1, aspect='auto')
            
            # empty scatter overlays for TP/FP
            tp_scatter = gt_ax.scatter([], [], s=heatmap_marker_size,
                                       c='green', marker='s', alpha=1.0, label="TP")
            fp_scatter = gt_ax.scatter([], [], s=heatmap_marker_size,
                                       c='red', marker='s', alpha=0.8, label="FP")
                        
            #gt_ax.legend(loc='upper right')
            gt_fig.tight_layout()

 

    # Helper: nearest point in screen (pixel) space
    pts_data = np.column_stack([recalls, precisions])
    def nearest_index(event):
        if event.inaxes is not ax:
            return None
        disp_pts = ax.transData.transform(pts_data)
        click_xy = np.array([event.x, event.y])
        d2 = np.sum((disp_pts - click_xy)**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] <= (12**2):
            return idx
        return None

    def on_click(event):
        nonlocal tp_scatter, fp_scatter
        idx = nearest_index(event)
        if idx is None:
            return

        th   = float(thresholds[idx])
        prec = float(precisions[idx])
        rec  = float(recalls[idx])

        # masks from your helpers
        GT_P, num_GT_P, _ = get_GT_P(GT=GT, M=M, r_gt=r_gt)
        P,    num_P    = get_P(th_d=th, D=D, S=S, M=M, GT_P = GT_P, matching=matching)
        TP,   num_TP   = get_TP(P=P, GT_P=GT_P)
        FP = P & (~GT_P)
        num_FP = np.sum(FP)

        sel_marker.set_offsets([[rec, prec]])

        ensure_gt_window(GT_P)
        gt_im.set_data(GT_P.astype(int))

        # rgba_matrix = gt_im.cmap(gt_im.get_array())     # RGBA floats [0,1]
        # rgb_matrix  = (rgba_matrix[:, :, :3] * 255).astype(np.uint8)
        # print(rgb_matrix)    # (H, W, 3)
        # out_name = f"/home/alejandro/test0.npy"
        # np.save(out_name, rgb_matrix)
        # print(f"Saved RGB matrix to {out_name}")

        # update scatter coordinates (row = y, col = x)
        tp_y, tp_x = np.where(TP)
        fp_y, fp_x = np.where(FP)
        tp_scatter.set_offsets(np.c_[tp_x, tp_y])
        fp_scatter.set_offsets(np.c_[fp_x, fp_y])

        # gt_ax.set_title(f"GT_P(gray) + TP(green) + FP(red)\n"
        #                 f"r_gt={r_gt:.6g} | th={th:.6g}, "
        #                 f"#GT_P={num_GT_P}, #TP={num_TP}, #FP={num_FP}")
        # gt_ax.spines['top'].set_visible(False)
        # gt_ax.spines['right'].set_visible(False)
        # gt_ax.set_xticks([])
        # gt_ax.set_yticks([])
        # gt_ax.set_xlabel('')
        # gt_ax.set_ylabel('')
        gt_fig.canvas.draw_idle()

        print(f"Clicked point -> th={th:.6g}, P={prec:.3f}, R={rec:.3f} | "
              f"#GT_P={num_GT_P}, #P={num_P}, #TP={num_TP}, #FP={num_FP}", flush=True)

        fig.canvas.draw_idle()
        plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)

    try:
        plt.show(block=False)
    except TypeError:
        plt.show()

    return fig, ax


