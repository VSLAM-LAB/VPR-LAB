import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os

def overlay_distance_gt_mask_browser(D, GT, M, db_df, query_df, k=5, r_gt=0.5, click_thresh=1, output_dir="output_matches"):
    """
    Interactive visualization of D, GT, and M with image browsing.
    On click: saves the query and its best k DB matches as separate PDFs.
    DB matches get a green or red border depending on correctness.
    """
    nrows, ncols = D.shape

    # Background from GT
    img = np.ones((nrows, ncols, 3))
    img[GT <= r_gt] = [0, 0, 0]   # black = true match
    img[M == 0] = [1, 1, 1]       # masked = white

    # Overlay lowest-k D
    for j in range(ncols):
        col = D[:, j].copy()
        col[M[:, j] == 0] = np.inf
        idx = np.argpartition(col, k)[:k]
        for i in idx:
            if M[i, j] == 0:
                continue
            if GT[i, j] <= r_gt:
                img[i, j] = [0, 1, 0]  # green
            else:
                img[i, j] = [1, 0, 0]  # red

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"Overlay (k={k}, r_gt={r_gt})\nClick a column to save images")
    ax.set_xlabel("Queries")
    ax.set_ylabel("Database")

    os.makedirs(output_dir, exist_ok=True)

    def save_clean_image(image_path, out_path, border_color=None, border_width=20):
        """Save an image as a clean PDF, optionally with a border."""
        fig, ax = plt.subplots()
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis("off")

        # Add border if needed
        if border_color is not None:
            rect = patches.Rectangle(
                (0, 0), img.shape[1], img.shape[0],
                linewidth=border_width, edgecolor=border_color, facecolor="none"
            )
            ax.add_patch(rect)

        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def onclick(event):
        if event.inaxes != ax:
            return
        x = int(round(event.xdata))
        if x < 0 or x >= ncols:
            return

        # Define neighborhood around click
        left = max(0, x - click_thresh)
        right = min(ncols, x + click_thresh + 1)
        candidate_cols = range(left, right)

        # Count true positives per candidate
        tp_counts = []
        for j_cand in candidate_cols:
            col_masked = (M[:, j_cand] == 1)
            col_true = (GT[:, j_cand] <= r_gt) & col_masked
            tp_counts.append(np.sum(col_true))

        if len(tp_counts) == 0:
            print("No candidate columns found")
            return

        max_tp = np.max(tp_counts)
        if max_tp > 0:
            j = candidate_cols[np.argmax(tp_counts)]
            print(f"Picked column {j} (clicked {x}, max TPs={max_tp} in vicinity)")
        else:
            # fallback: pick nearest column to click
            j = x
            print(f"No TPs in vicinity, falling back to clicked column {j}")

        #j = 37
        # Best k matches for this query
        col = D[:, j].copy()
        col[M[:, j] == 0] = np.inf
        best_idx = np.argpartition(col, k)[:k]
        best_idx = best_idx[np.argsort(col[best_idx])]  # sort by distance

        # Query path (no border)
        query_path = query_df[1][j]
        query_out = os.path.join(output_dir, f"query_{j}.pdf")
        save_clean_image(query_path, query_out)
        print(f"\nSaved query {j} → {query_out}")

        # Save DB matches with colored borders (brighter red/green, thicker)
        for rank, i in enumerate(best_idx):
            db_path = db_df[1][i]
            match_type = "TP" if GT[i, j] <= r_gt else "FP"
            border_color = "#00FF00" if match_type == "TP" else "#FF0000"  # bright green / bright red
            db_out = os.path.join(output_dir, f"query_{j}_db_{i}_rank{rank+1}.pdf")
            save_clean_image(db_path, db_out, border_color=border_color, border_width=20)
            print(f"  Rank {rank+1}: saved DB {i} ({match_type}) → {db_out}")

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
