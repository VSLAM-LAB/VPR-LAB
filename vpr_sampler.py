"""
vpr_sampler.py — VPR Sequence Sampler
======================================
Reads a VPR distance matrix D (nxn) and an rgb.csv file, runs a
threshold-sweep sampling strategy, and lets you interactively pick a
threshold by clicking on the curve.  On each click it shows the original
vs filtered D matrix.  A "Save images" button copies the selected frames
to the output folder (and optionally to extra camera folders).

Usage — single camera
---------------------
    python vpr_sampler.py \
        --D_matrix  path/to/D.npy \
        --rgb_csv   path/to/rgb.csv \
        --img_root  path/to/image/root \
        --output    path/to/output_folder \
        [--n_thresholds 50] \
        [--th_min 0.1] \
        [--th_max 1.0]

Usage — multiple camera folders (same img_root, same sampling)
--------------------------------------------------------------
    python vpr_sampler.py \
        --D_matrix      path/to/D.npy \
        --rgb_csv       path/to/rgb.csv \
        --img_root      path/to/image/root \
        --output        path/to/output_folder \
        --extra_folders rgb_1 depth_0 rgb_2

    Each name in --extra_folders is a sub-folder relative to --img_root.
    Images are sorted alphabetically and matched by position to the D-matrix
    row index.  Filtered images are saved to <img_root>/<folder>_filtered/,
    which is cleared before every save.

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
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.widgets

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive VPR sequence sampler"
    )
    p.add_argument("--D_matrix",      required=True,
                   help="Path to D matrix (.npy, shape NxN)")
    p.add_argument("--rgb_csv",       required=True,
                   help="Path to rgb.csv with columns ts_rgb_0 and path_rgb_0")
    p.add_argument("--img_root",      required=True,
                   help="Root folder that path_rgb_0 values are relative to")
    p.add_argument("--output",        required=True,
                   help="Destination folder for sampled images from rgb_csv")
    p.add_argument("--extra_folders", nargs="*", default=[],
                   help="Additional sub-folders under img_root to filter with "
                        "the same indexes (saved to <folder>_filtered/ next to "
                        "each source folder)")
    p.add_argument("--n_thresholds",  type=int,   default=50,
                   help="Number of threshold steps to sweep (default: 50)")
    p.add_argument("--th_min",        type=float, default=0.1,
                   help="Minimum threshold (default: 0.1)")
    p.add_argument("--th_max",        type=float, default=1.0,
                   help="Maximum threshold (default: 1.0)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_inputs(D_path: str, csv_path: str, img_root: str):
    """Load distance matrix and image list from csv; return (D, image_paths, df)."""
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


def collect_extra_folder_images(img_root: str, folder_names: list, n: int):
    """
    For each folder name, collect sorted images and return a list of
    (src_dir, dst_dir, [image_path, ...]) tuples.

    Images are matched by sorted order — index k in the D matrix corresponds
    to the k-th image (alphabetically) in each folder.
    Filtered images are saved to <img_root>/<folder_name>_filtered/.
    """
    root = Path(img_root)
    folders = []
    for name in folder_names:
        src = root / name
        if not src.is_dir():
            print(f"  [warn] extra folder not found, skipping: {src}")
            continue
        imgs = sorted([
            f for f in src.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ])
        if len(imgs) != n:
            print(f"  [warn] {src} has {len(imgs)} images but D has {n} rows — skipping")
            continue
        dst = src.parent / f"{name}_filtered"
        folders.append((src, dst, imgs))
        print(f"  Extra folder : {src}  ->  {dst}")
    return folders


# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------

def run_threshold_sweep(D: np.ndarray, th_min: float, th_max: float, n_steps: int):
    """
    Sweep thresholds and return:
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
    Greedy forward sampler: walk from anchor i; when D[i,j] > th record j-1
    as a keyframe and reset anchor to j.
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

def _clear_and_save(src_paths: list, indexes: list, dst_dir: Path, label: str):
    """Clear dst_dir (files only) then copy the indexed images into it."""
    if dst_dir.exists():
        for f in dst_dir.iterdir():
            if f.is_file():
                f.unlink()
        print(f"  Cleared : {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx in indexes:
        if idx >= len(src_paths):
            continue
        src = Path(src_paths[idx])
        if not src.exists():
            print(f"  [warn] not found: {src}")
            continue
        shutil.copy(src, dst_dir / src.name)
        copied += 1

    print(f"  Saved {copied} images -> {dst_dir}  [{label}]")


def save_all(image_paths, extra_folders, indexes, output_dir: Path):
    """Save primary camera + all extra folders using the same indexes."""
    _clear_and_save(image_paths, indexes, output_dir, "primary")
    for _src_dir, dst_dir, imgs in extra_folders:
        _clear_and_save(imgs, indexes, dst_dir, _src_dir.name)


# ---------------------------------------------------------------------------
# Interactive plot
# ---------------------------------------------------------------------------

def launch_interactive(D, image_paths, extra_folders, all_indexes, results,
                       output_dir: Path):
    """
    Threshold-vs-count curve with click-to-select interaction.
    A Save button on the comparison window writes all folders at once.
    """
    ths, lengths = zip(*results)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ths, lengths, marker="o", markersize=3, color="steelblue")
    ax.set_xlabel("Threshold (th)")
    ax.set_ylabel("Number of selected frames")
    ax.set_title("Click on the curve to select a threshold")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    state = {"fig2": None, "vline": None}

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return

        clicked_th = event.xdata
        nearest_th = min(all_indexes.keys(), key=lambda t: abs(t - clicked_th))
        indexes    = all_indexes[nearest_th]

        print(f"\nSelected threshold : {nearest_th:.4f}")
        print(f"  -> {len(indexes)} frames selected out of {D.shape[0]}")

        # ---- vertical marker on curve ----
        if state["vline"]:
            state["vline"].remove()
        state["vline"] = ax.axvline(nearest_th, color="crimson",
                                    linestyle="--", linewidth=1.2,
                                    label=f"th={nearest_th:.3f}")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw_idle()

        # ---- close previous comparison window ----
        if state["fig2"] is not None:
            try:
                plt.close(state["fig2"])
            except Exception:
                pass

        # ---- D matrix comparison ----
        D_filtered = D[np.ix_(indexes, indexes)]

        fig2, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig2.suptitle(
            f"Threshold = {nearest_th:.4f}  |  {len(indexes)} / {D.shape[0]} frames",
            fontsize=11
        )

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
        n_extra = len(extra_folders)
        n_total = 1 + n_extra
        btn_label = f"Save images ({n_total} folder{'s' if n_total > 1 else ''})"
        ax_btn = fig2.add_axes([0.40, 0.01, 0.20, 0.06])
        btn = matplotlib.widgets.Button(ax_btn, btn_label,
                                        color="0.85", hovercolor="0.7")

        def on_save(_event):
            btn.label.set_text("Saving ...")
            fig2.canvas.draw_idle()
            save_all(image_paths, extra_folders, indexes, output_dir)
            btn.label.set_text("Saved!")
            btn.color = "0.6"
            fig2.canvas.draw_idle()

        btn.on_clicked(on_save)
        fig2._save_btn = btn  # prevent GC

        fig2.tight_layout(rect=[0, 0.08, 1, 1])
        fig2.show()
        state["fig2"] = fig2

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("Loading inputs ...")
    D, image_paths, df = load_inputs(args.D_matrix, args.rgb_csv, args.img_root)
    print(f"  D shape      : {D.shape}")
    print(f"  Images found : {len(image_paths)}")

    # Extra folders
    extra_folders = []
    if args.extra_folders:
        print(f"\nResolving {len(args.extra_folders)} extra folder(s) ...")
        extra_folders = collect_extra_folder_images(
            args.img_root, args.extra_folders, D.shape[0]
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSweeping {args.n_thresholds} thresholds "
          f"from {args.th_min} to {args.th_max} ...")
    all_indexes, results = run_threshold_sweep(
        D, args.th_min, args.th_max, args.n_thresholds
    )

    ths, lengths = zip(*results)
    print(f"  Range of selected frames : {min(lengths)} - {max(lengths)}")

    print("\nOpening interactive plot ...")
    print("  Click anywhere on the curve to select a threshold.")
    print("  Press 'Save images' on the comparison window to write all folders.")
    print("  Close the curve window to exit.\n")
    launch_interactive(D, image_paths, extra_folders, all_indexes, results,
                       output_dir)


if __name__ == "__main__":
    main()
