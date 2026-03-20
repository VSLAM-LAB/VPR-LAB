"""
vpr_sampler.py — VPR Sequence Sampler
======================================
Reads a VPR distance matrix D (nxn) and an rgb.csv file, runs a
threshold-sweep sampling strategy, and lets you interactively pick a
threshold by clicking on the curve.  On each click it shows the original
vs filtered D matrix and saves the selected images to an output folder.

Usage
-----
    python vpr_sampler.py \
        --D_matrix  path/to/D.npy \
        --rgb_csv   path/to/rgb.csv \
        --img_root  path/to/image/root \
        --output    path/to/output_folder \
        [--n_thresholds 50] \
        [--th_min 0.1] \
        [--th_max 1.0]

CSV format
----------
The csv must have at least two columns:
    ts_rgb_0   — timestamp in nanoseconds (integer)
    path_rgb_0 — relative path, e.g.  rgb_0/image_name.png

The absolute path of each image is resolved as:
    <img_root> / <path_rgb_0>
"""

import argparse
import shutil
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive VPR sequence sampler"
    )
    p.add_argument("--D_matrix",      required=True,  help="Path to D matrix (.npy, shape NxN)")
    p.add_argument("--rgb_csv",       required=True,  help="Path to rgb.csv with columns ts_rgb_0 and path_rgb_0")
    p.add_argument("--img_root",      required=True,  help="Root folder that path_rgb_0 values are relative to")
    p.add_argument("--output",        required=True,  help="Destination folder for sampled images")
    p.add_argument("--n_thresholds",  type=int,   default=50,  help="Number of threshold steps to sweep (default: 50)")
    p.add_argument("--th_min",        type=float, default=0.1, help="Minimum threshold (default: 0.1)")
    p.add_argument("--th_max",        type=float, default=1.0, help="Maximum threshold (default: 1.0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_inputs(D_path: str, csv_path: str, img_root: str):
    """Load distance matrix and image list; return (D, image_paths)."""
    D = np.load(D_path)
    assert D.ndim == 2 and D.shape[0] == D.shape[1], \
        f"D must be a square 2-D array, got shape {D.shape}"

    df = pd.read_csv(csv_path)
    required = {"ts_rgb_0 (ns)", "path_rgb_0"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    root = Path(img_root)
    image_paths = [root / rel for rel in df["path_rgb_0"]]

    n = D.shape[0]
    if len(image_paths) != n:
        raise ValueError(
            f"D matrix has {n} rows but CSV has {len(image_paths)} entries."
        )

    return D, image_paths, df


# ---------------------------------------------------------------------------
# Sampling strategy  (from sample_sequence.py)
# ---------------------------------------------------------------------------

def run_threshold_sweep(D: np.ndarray, th_min: float, th_max: float, n_steps: int):
    """
    For every threshold in [th_min, th_max] run the greedy sampler and
    record the number of selected frames.

    Returns
    -------
    all_indexes : dict  {float_th -> list[int]}
    results     : list[(th, n_selected)]
    """
    all_indexes = {}
    results     = []

    for th in np.linspace(th_min, th_max, num=n_steps):
        indexes = _sample_indexes(D, th)
        results.append((th, len(indexes)))
        all_indexes[th] = indexes

    return all_indexes, results


def _sample_indexes(D: np.ndarray, th: float):
    """
    Greedy forward sampler: walk along the diagonal; whenever the distance
    between the current anchor frame i and the next frame j exceeds `th`,
    record j-1 as a keyframe and reset the anchor to j.
    """
    indexes = [0]
    i = 0
    j = 0
    while i < D.shape[0] and j < D.shape[1]:
        while j < D.shape[1]:
            if D[i, j] > th:
                indexes.append(j - 1)
                i = j
            j += 1
    return indexes


# ---------------------------------------------------------------------------
# Image saving
# ---------------------------------------------------------------------------

def save_selected_images(image_paths, indexes, output_dir: Path):
    """Clear output_dir and copy the selected images directly into it."""
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file():
                f.unlink()
        print(f"  Cleared previous save in: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx in indexes:
        if idx >= len(image_paths):
            continue
        src = image_paths[idx]
        if not src.exists():
            print(f"  [warn] image not found: {src}")
            continue
        shutil.copy(src, output_dir / src.name)
        copied += 1

    print(f"  Saved {copied} images → {output_dir}")


# ---------------------------------------------------------------------------
# Interactive plot
# ---------------------------------------------------------------------------

def launch_interactive(D, image_paths, all_indexes, results, output_dir: Path):
    """
    Show the threshold-vs-count curve.  Each click picks the nearest
    evaluated threshold, opens a side-by-side D matrix comparison, and
    saves the images.  The main curve window stays open for further clicks.
    """
    ths, lengths = zip(*results)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ths, lengths, marker="o", markersize=3, color="steelblue")
    ax.set_xlabel("Threshold (th)")
    ax.set_ylabel("Number of selected frames")
    ax.set_title("Click on the curve to select a threshold\n"
                 "(each click saves the selected images and shows the filtered matrix)")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # Keep track of the last comparison figure so we can close it on the
    # next click (avoids an ever-growing stack of windows).
    state = {"fig2": None, "vline": None}

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        clicked_th = event.xdata
        nearest_th = min(all_indexes.keys(), key=lambda t: abs(t - clicked_th))
        indexes    = all_indexes[nearest_th]

        print(f"\nSelected threshold : {nearest_th:.4f}")
        print(f"  → {len(indexes)} frames selected out of {D.shape[0]}")

        # ---- update vertical marker on curve ----
        if state["vline"]:
            state["vline"].remove()
        state["vline"] = ax.axvline(nearest_th, color="crimson",
                                    linestyle="--", linewidth=1.2, label=f"th={nearest_th:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

        # ---- close previous comparison window ----
        if state["fig2"] is not None:
            try:
                plt.close(state["fig2"])
            except Exception:
                pass

        # ---- show D matrices ----
        D_filtered = D[np.ix_(indexes, indexes)]

        fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig2.suptitle(f"Threshold = {nearest_th:.4f}  |  {len(indexes)} / {D.shape[0]} frames",
                      fontsize=11)

        im0 = axes[0].imshow(D, cmap="viridis", aspect="auto")
        axes[0].set_title("Original D matrix")
        axes[0].set_xlabel("j")
        axes[0].set_ylabel("i")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(D_filtered, cmap="viridis", aspect="auto")
        axes[1].set_title(f"Filtered D matrix  ({len(indexes)} frames)")
        axes[1].set_xlabel("j (subsampled)")
        axes[1].set_ylabel("i (subsampled)")
        plt.colorbar(im1, ax=axes[1])

        # ---- save button ----
        ax_btn = fig2.add_axes([0.44, 0.01, 0.12, 0.05])
        btn = matplotlib.widgets.Button(ax_btn, "Save images", color="0.85", hovercolor="0.7")

        def on_save(_event):
            btn.label.set_text("Saving …")
            fig2.canvas.draw_idle()
            save_selected_images(image_paths, indexes, output_dir)
            btn.label.set_text("Saved ✓")
            btn.color = "0.6"
            fig2.canvas.draw_idle()

        btn.on_clicked(on_save)
        # Keep a reference so the button isn't garbage-collected
        fig2._save_btn = btn

        fig2.tight_layout(rect=[0, 0.07, 1, 1])
        fig2.show()
        state["fig2"] = fig2

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Loading inputs …")
    D, image_paths, df = load_inputs(args.D_matrix, args.rgb_csv, args.img_root)
    print(f"  D shape      : {D.shape}")
    print(f"  Images found : {len(image_paths)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSweeping {args.n_thresholds} thresholds "
          f"from {args.th_min} to {args.th_max} …")
    all_indexes, results = run_threshold_sweep(
        D, args.th_min, args.th_max, args.n_thresholds
    )

    ths, lengths = zip(*results)
    print(f"  Range of selected frames : {min(lengths)} – {max(lengths)}")

    print("\nOpening interactive plot …")
    print("  Click anywhere on the curve to select a threshold.")
    print("  Close the plot window to exit.\n")
    launch_interactive(D, image_paths, all_indexes, results, output_dir)


if __name__ == "__main__":
    main()
