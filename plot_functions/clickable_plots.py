
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plot_settings_default = {}
factor = 2
plot_settings_default = {
    'legend_on': True,
    'title_on': True,
    'xlabel_on': True,
    'ylabel_on': True,
    'top': True,
    'right': True,
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

def _move_figure(fig, x, y):
    """Move figure to position (x, y) in pixels on screen."""
    backend = plt.get_backend()
    mng = fig.canvas.manager
    try:
        if backend == "TkAgg":
            mng.window.wm_geometry(f"+{x}+{y}")
        elif backend in ["WXAgg", "wxAgg"]:
            mng.window.SetPosition((x, y))
        else:  # QtAgg, Qt5Agg, Qt6Agg
            mng.window.move(x, y)
    except Exception as e:
        print(f"[warn] Could not move figure: {e}")

def plot_trajectory(traj, fig, ax, label, color = 'b',  mode="3d"):
    x, y, z = traj[:,0], traj[:,1], traj[:,2]
    title = ax.get_title()

    if mode == "3d":
        ax.plot(x, y, z, label=label, color=color)
        ax.scatter(x[0], y[0], z[0], c=color, marker="o", label="Start")
        ax.scatter(x[-1], y[-1], z[-1], c=color, marker="x", label="End")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        # if not '3D' in title:
        #     ax.set_title(ax.get_title() + " (3D view)")

    elif mode == "2d":
        ax.plot(x, y, label=label, color=color)
        ax.scatter(x[0], y[0], c=color, marker="o", label="Start")
        ax.scatter(x[-1], y[-1], c=color, marker="x", label="End")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        # if not '2D' in title:
        #     ax.set_title(ax.get_title() + " (2D view)")

    ax.legend()
    return fig, ax

def plot_warped_image(kpts0, kpts1, path_to_image0, path_to_image1, fig, ax):
    if len(kpts0) < 4 or len(kpts1) < 4:
        return fig, ax

    import cv2

    # --- Convert to NumPy for OpenCV ---
    pts0 = kpts0.detach().cpu().numpy().astype(np.float32)  # (K,2)
    pts1 = kpts1.detach().cpu().numpy().astype(np.float32)  # (K,2)

    # --- Estimate homography (guard for failures) ---
    print(len(pts0), len(pts1))
    H, inliers = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    print(len(inliers))
    if H is None:
        raise RuntimeError("cv2.findHomography failed; not enough good matches or points are degenerate.")

    mask = inliers.ravel().astype(bool)
    pts0 = pts0[mask]
    pts1 = pts1[mask]
    print(len(pts0))
    print(len(pts1))
    # --- Load images via OpenCV for warping (H,W,C in BGR) ---
    cv_img0 = cv2.imread(path_to_image0, cv2.IMREAD_COLOR)
    cv_img1 = cv2.imread(path_to_image1, cv2.IMREAD_COLOR)
    h, w = cv_img1.shape[:2]

    # --- Warp and display ---
    warped = cv2.warpPerspective(cv_img0, H, (w, h))
    projected_kpts = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2), H).reshape(-1, 2)

    alpha = 0.25
    blue_bgr = (0.7058823529411765*255, 0.4666666666666667*255, 0.12156862745098039*255)
    orange_bgr = (0.054901960784313725*255, 0.4980392156862745*255, 1.0*255)

    tint_filter = np.full_like(cv_img1, blue_bgr, dtype=np.uint8)
    tinted_img = cv2.addWeighted(cv_img1, 1 - alpha, tint_filter, alpha, 0)

    # create a mask for the warped query image and the database image
    fg_mask = (warped != 0).any(axis=-1)
    fg_mask_uint8 = fg_mask.astype(np.uint8) * 255
    fg_contours, _ = cv2.findContours(fg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_bg = (warped == 0).all(axis=-1)
    mask_uint8 = mask_bg.astype(np.uint8) * 255
    bg_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # remove the overlap from the background mask to keep coloring clear
    composite = warped.copy()
    composite[mask_bg] = tinted_img[mask_bg]

    cv2.drawContours(composite, bg_contours, -1, blue_bgr, thickness=4)
    cv2.drawContours(composite, fg_contours, -1, orange_bgr, thickness=3)

    # plot actual keypoints from the database image and projected keypoints from the query image
    #plt.figure(figsize=(14, 14))
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    #ax.scatter(pts1[:, 0], pts1[:, 1], c='blue', marker='o', s=100, label='Actual Keypoints')
    #ax.scatter(projected_kpts[:, 0], projected_kpts[:, 1], c='orange', s=150, marker='x', label='Projected Keypoints')
    ax.scatter(pts1[:, 0], pts1[:, 1], c='blue', marker='o', s=10, label='Actual Keypoints')
    ax.scatter(projected_kpts[:, 0], projected_kpts[:, 1], c='orange', s=10, marker='x', label='Projected Keypoints')
    ax.axis('off')

    return fig, ax
    #plt.show()

def plot_clickable_matrix(
    matrix, db_df, query_df, gt_db=None, gt_query=None, q_path_col=1, db_path_col=1,
    radius_px=50, pick='min', plot_settings = plot_settings_default, lightglue_matching_func=True
):
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    assert pick in ('min', 'max'), "pick must be 'min' or 'max'"

    # --- Main matrix heatmap ---
    fig_main, ax_main = plt.subplots(figsize=(8, 6))
    _move_figure(fig_main, 200, 200)
    im = ax_main.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax_main, label="Distance")
    ax_main.set_xlabel("Query")
    ax_main.set_ylabel("Database")
    ax_main.set_title("Main matrix heatmap")
    ax_main.xaxis.set_ticks_position('top')
    ax_main.xaxis.set_label_position('top')
    labels = [item.get_text() for item in ax_main.get_xticklabels()]
    if labels:
        labels[1] = ""   # blank out the first one
    ax_main.set_xticklabels(labels)

    # Click marker + annotation
    marker = ax_main.scatter([], [], s=80, facecolors="none", edgecolors="white", linewidths=1.8)
    ann = ax_main.annotate(
        "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.8),
    )
    ann.set_visible(False)

    windows = {"rgb_fig": None, "traj_fig": None, "ligthglue_fig": None, "warp_fig": None}

    def _safe_imread(p):
        try:
            return plt.imread(p)
        except Exception as e:
            print(f"[warn] Failed to read image: {p} ({e})")
            return None

    def _pick_extreme_near(matrix, x, y, r=5, mode='min'):
        """
        GT: 2D array
        (x, y): float click coords in image (col=x, row=y)
        r: radius in pixels (inclusive)
        mode: 'min' or 'max'
        Returns: row, col, val or (None, None, None) if none in radius
        """
        if np.isnan(x) or np.isnan(y):
            return None, None, None

        # Candidate window bounds (clip to GT)
        cmin = max(int(np.floor(x - r)), 0)
        cmax = min(int(np.ceil (x + r)), matrix.shape[1] - 1)
        rmin = max(int(np.floor(y - r)), 0)
        rmax = min(int(np.ceil (y + r)), matrix.shape[0] - 1)

        if cmax < cmin or rmax < rmin:
            return None, None, None

        # Build local grid
        cols = np.arange(cmin, cmax + 1)
        rows = np.arange(rmin, rmax + 1)
        CC, RR = np.meshgrid(cols, rows)  # shape (Hloc, Wloc)

        # Mask by circular radius
        d2 = (CC - x) ** 2 + (RR - y) ** 2
        mask = d2 <= (r ** 2)
        if not np.any(mask):
            return None, None, None

        patch = matrix[rmin:rmax + 1, cmin:cmax + 1]
        vals = patch[mask]

        if mode == 'min':
            k = np.argmin(vals)
        else:
            k = np.argmax(vals)

        # Map flat index back to row/col
        rr, cc = np.where(mask)
        rr_sel = rr[k] + rmin
        cc_sel = cc[k] + cmin
        return int(rr_sel), int(cc_sel), float(matrix[int(rr_sel), int(cc_sel)])

    def onclick(event):
        if event.inaxes is not ax_main:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Find extreme within radius around the click
        row, col, val = _pick_extreme_near(matrix, event.xdata, event.ydata, r=radius_px, mode=pick)
        if row is None:
            # Nothing within radius: ignore
            print("[info] No pixel within radius; click ignored.")
            return

        # Update marker + annotation
        marker.set_offsets([[col, row]])
        ax_main.set_title(f"Clicked (snapped): row={row}, col={col}, value={val:.6f} [{pick}]")
        ann.xy = (col, row)
        ann.set_text(f"({row}, {col}) = {val:.6f}")
        ann.set_visible(True)
        fig_main.canvas.draw_idle()

        # ---------- RGB window ----------
        if not lightglue_matching_func:
            if windows["rgb_fig"] is not None:
                plt.close(windows["rgb_fig"])
                windows["rgb_fig"] = None

            fig_rgb, ax_rgb = plt.subplots(1, 2, figsize=(10, 4))
            _move_figure(fig_rgb, 1200, 200)
            windows["rgb_fig"] = fig_rgb

            if 0 <= col < len(query_df):
                q_img_path = query_df.iloc[col][q_path_col]
                q_img = _safe_imread(q_img_path)
                if q_img is not None:
                    ax_rgb[0].imshow(q_img)
                    ax_rgb[0].set_title(f"Query {col}\n{os.path.basename(str(q_img_path))}")
                else:
                    ax_rgb[0].text(0.5, 0.5, "Failed to load query", ha="center", va="center")
            else:
                ax_rgb[0].text(0.5, 0.5, "Query N/A", ha="center", va="center")
            ax_rgb[0].axis("off")

            if 0 <= row < len(db_df):
                db_img_path = db_df.iloc[row][db_path_col]
                db_img = _safe_imread(db_img_path)
                if db_img is not None:
                    ax_rgb[1].imshow(db_img)
                    ax_rgb[1].set_title(f"DB {row}\n{os.path.basename(str(db_img_path))}")
                else:
                    ax_rgb[1].text(0.5, 0.5, "Failed to load db", ha="center", va="center")
            else:
                ax_rgb[1].text(0.5, 0.5, "DB N/A", ha="center", va="center")
            ax_rgb[1].axis("off")

            fig_rgb.suptitle(f"Distance = {val:.6f} (m)")
            fig_rgb.tight_layout()
            plt.show(block=False)

        # ---------- Trajectories window ----------
        if gt_db is not None and gt_query is not None:
            if windows["traj_fig"] is not None:
                plt.close(windows["traj_fig"])
                windows["traj_fig"] = None

            fig_traj, ax_traj = plt.subplots()
            _move_figure(fig_traj, 2300, 200)
            windows["traj_fig"] = fig_traj

            fig_traj, ax_traj = plot_trajectory(gt_db, fig_traj, ax_traj, label='gt_db', color='r', mode='2d')
            fig_traj, ax_traj = plot_trajectory(gt_query, fig_traj, ax_traj, label='gt_query', color='k', mode='2d')

            ax_traj.scatter(gt_db[row,0], gt_db[row,1], gt_db[row,2], c='blue', marker="d", label="db")
            ax_traj.scatter(gt_query[col,0], gt_query[col,1], gt_query[col,2], c='orange', marker="d", label="query")
            ax_traj.plot([gt_db[row,0], gt_query[col,0]], [gt_db[row,1], gt_query[col,1]],
                    linestyle="--", linewidth=2, color="g", label="match")

            ax_traj.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
            ax_traj.set_aspect('equal', adjustable="box")
            fig_traj.tight_layout()

            ax_traj.legend().set_visible(plot_settings['legend_on'])
            ax_traj.spines['top'].set_visible(plot_settings['top'])
            ax_traj.spines['right'].set_visible(plot_settings['right'])
            ax_traj.spines['bottom'].set_visible(plot_settings['top'])
            ax_traj.spines['left'].set_visible(plot_settings['right'])
            if not plot_settings['top']: ax_traj.set_xticklabels([])
            if not plot_settings['top']: ax_traj.set_yticklabels([])
            ax_traj.grid(
                plot_settings['grid_on'],
                alpha=plot_settings['grid_alpha'],
                linestyle=plot_settings['grid_style']
            )
            if plot_settings['xlabel_on']: ax_traj.set_xlabel('X [m]')
            else: ax_traj.set_xlabel('')
            if plot_settings['ylabel_on']: ax_traj.set_ylabel('Y [m]')
            else: ax_traj.set_ylabel('')
            if plot_settings['title_on']: ax_traj.set_title("Trajectories")
            #plt.rcParams.update(plot_settings['settings'])

            plt.show(block=False)

        # ---------- LightGlue + Warped (unchanged) ----------
        if lightglue_matching_func:
            from utilities import lightglue_matching
            print("lightglue_matching")

            q_img_path = query_df['path_rgb_0'].iloc[col] if 0 <= col < len(query_df) else None
            db_img_path = db_df['path_rgb_0'].iloc[row] if 0 <= row < len(db_df) else None
            
            print(f"Query image path: {q_img_path}")
            print(f"Database image path: {db_img_path}")
            kpts0, kpts1 = lightglue_matching(q_img_path, db_img_path, plot=True)

            if windows["warp_fig"] is not None:
                plt.close(windows["warp_fig"])
                windows["warp_fig"] = None

            if not (kpts0 is None or kpts1 is None):
                fig_warp, ax_warp = plt.subplots()
                _move_figure(fig_warp, 3300, 200)
                windows["warp_fig"] = fig_warp
                #ax_warp.set_title("Warped Image")
                print("plot_warped_image")
                fig_warp, ax_warp = plot_warped_image(kpts0, kpts1, q_img_path, db_img_path, fig=fig_warp, ax=ax_warp)
                ax_warp.set_aspect('equal', adjustable="box")
                fig_warp.tight_layout()
                plt.show(block=False)

        plt.pause(0.001)

    fig_main.canvas.mpl_connect("button_press_event", onclick)

    plt.show()
    return fig_main, ax_main