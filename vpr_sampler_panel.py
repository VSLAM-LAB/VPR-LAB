"""
vpr_sampler_panel.py — VPR Sequence Sampler (Panel web app)
=============================================================
Browser-based version of vpr_sampler.py.  No display required — runs
headlessly on an HPC and is accessed via an SSH tunnel from your laptop.

Usage
-----
    panel serve vpr_sampler_panel.py \
        --args \
            --D_matrix      path/to/D.npy \
            --rgb_csv       path/to/rgb.csv \
            --img_root      path/to/image/root \
            --output        path/to/output_folder \
            [--extra_folders rgb_1 depth_0] \
            [--n_thresholds 50] \
            [--th_min 0.1] \
            [--th_max 1.0]

SSH tunnel (run on your laptop)
--------------------------------
    ssh -L 5006:localhost:5006 user@hpc-hostname

Then open:
    http://localhost:5006/vpr_sampler_panel

Workflow
--------
1. Click anywhere on the threshold curve — the nearest evaluated threshold
   is selected and the D matrix comparison updates instantly.
2. Press "Save images" to copy the selected frames to disk.
   All output folders are cleared before each save.

Install
-------
    pip install panel bokeh matplotlib numpy pandas
"""

import argparse
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import panel as pn
import panel.widgets as pnw
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Span, TapTool

pn.extension(sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VPR Sequence Sampler — Panel app")
    p.add_argument("--D_matrix",      required=True)
    p.add_argument("--rgb_csv",       required=True)
    p.add_argument("--img_root",      required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--extra_folders", nargs="*", default=[])
    p.add_argument("--n_thresholds",  type=int,   default=50)
    p.add_argument("--th_min",        type=float, default=0.1)
    p.add_argument("--th_max",        type=float, default=1.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_inputs(D_path, csv_path, img_root):
    D = np.load(D_path)
    assert D.ndim == 2 and D.shape[0] == D.shape[1], \
        f"D must be square, got {D.shape}"
    df = pd.read_csv(csv_path)
    missing = {"ts_rgb_0 (ns)", "path_rgb_0"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    root = Path(img_root)
    image_paths = [root / rel for rel in df["path_rgb_0"]]
    if len(image_paths) != D.shape[0]:
        raise ValueError(f"D has {D.shape[0]} rows but CSV has {len(image_paths)} entries.")
    return D, image_paths


def collect_extra_folders(img_root, folder_names, n):
    root, folders = Path(img_root), []
    for name in folder_names:
        src = root / name
        if not src.is_dir():
            print(f"[warn] extra folder not found, skipping: {src}")
            continue
        imgs = sorted([f for f in src.iterdir()
                       if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
        if len(imgs) != n:
            print(f"[warn] {src} has {len(imgs)} images, expected {n} — skipping")
            continue
        folders.append((src, src.parent / f"{name}_filtered", imgs))
    return folders


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def run_threshold_sweep(D, th_min, th_max, n_steps):
    all_indexes, results = {}, []
    for th in np.linspace(th_min, th_max, num=n_steps):
        idxs = _sample_indexes(D, th)
        results.append((th, len(idxs)))
        all_indexes[th] = idxs
    return all_indexes, results


def _sample_indexes(D, th):
    indexes = [0]
    i = j = 0
    while i < D.shape[0] and j < D.shape[1]:
        while j < D.shape[1]:
            if D[i, j] > th:
                indexes.append(j - 1)
                i = j
            j += 1
    return indexes


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _clear_and_save(src_paths, indexes, dst_dir, label):
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        for f in dst_dir.iterdir():
            if f.is_file():
                f.unlink()
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for idx in indexes:
        if idx >= len(src_paths):
            continue
        src = Path(src_paths[idx])
        if not src.exists():
            print(f"[warn] not found: {src}")
            continue
        shutil.copy(src, dst_dir / src.name)
        copied += 1
    print(f"Saved {copied} -> {dst_dir}  [{label}]")
    return copied


def save_all(image_paths, extra_folders, indexes, output_dir):
    total = _clear_and_save(image_paths, indexes, output_dir, "primary")
    for src_dir, dst_dir, imgs in extra_folders:
        total += _clear_and_save(imgs, indexes, dst_dir, src_dir.name)
    return total


# ---------------------------------------------------------------------------
# Matplotlib D-matrix figure -> PNG pane
# ---------------------------------------------------------------------------

def make_matrix_png(D, indexes, nearest_th):
    D_filtered = D[np.ix_(indexes, indexes)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f"th = {nearest_th:.4f}   |   {len(indexes)} / {D.shape[0]} frames",
        fontsize=11
    )
    im0 = axes[0].imshow(D, cmap="viridis", aspect="auto")
    axes[0].set_title("Original D")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(D_filtered, cmap="viridis", aspect="auto")
    axes[1].set_title(f"Filtered D  ({len(indexes)} frames)")
    axes[1].set_xlabel("j (sub)"); axes[1].set_ylabel("i (sub)")
    plt.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Bokeh curve with real tap interaction
# ---------------------------------------------------------------------------

def make_bokeh_curve(ths, lengths, all_indexes, D, image_paths,
                     extra_folders, output_dir, matrix_pane,
                     status_md, save_btn, state):
    th_arr = np.array(ths)

    source = ColumnDataSource(dict(th=ths, n=lengths))

    p = figure(
        width=600, height=350,
        title="Click on the curve to select a threshold",
        x_axis_label="Threshold (th)",
        y_axis_label="Selected frames",
        tools="tap,reset",
        toolbar_location="above",
    )
    p.line("th", "n", source=source, line_width=2, color="steelblue")
    p.circle("th", "n", source=source, size=6, color="steelblue",
             hover_color="crimson", nonselection_alpha=0.4)
    p.grid.grid_line_dash = [6, 4]

    vline = Span(location=0, dimension="height",
                 line_color="crimson", line_dash="dashed", line_width=2,
                 visible=False)
    p.add_layout(vline)

    # TapTool fires selected.on_change when the user clicks a point
    def on_selection_change(attr, old, new):
        if not new:
            return
        # 'new' is a list of selected indices into the ColumnDataSource
        idx      = new[0]
        nearest_th = ths[idx]
        indexes    = all_indexes[nearest_th]
        state["th"]      = nearest_th
        state["indexes"] = indexes

        # update vline
        vline.location = nearest_th
        vline.visible  = True

        # update D matrix pane
        png_bytes = make_matrix_png(D, indexes, nearest_th)
        matrix_pane.object = png_bytes

        # update status
        status_md.object = (
            f"**Threshold:** `{nearest_th:.4f}`  |  "
            f"**Selected:** {len(indexes)} / {D.shape[0]} frames"
        )
        save_btn.disabled = False

        # clear selection so the same point can be clicked again
        source.selected.indices = []

    source.selected.on_change("indices", on_selection_change)

    return p


# ---------------------------------------------------------------------------
# Full Panel app
# ---------------------------------------------------------------------------

def build_app(D, image_paths, extra_folders, all_indexes, results, output_dir):
    ths     = [r[0] for r in results]
    lengths = [r[1] for r in results]

    state = {"th": None, "indexes": None}

    # ---- reactive widgets ----
    status_md = pn.pane.Markdown(
        "Click on the curve to select a threshold.",
        styles={"background": "#f0f4f8", "padding": "8px", "border-radius": "4px"},
    )

    n_total  = 1 + len(extra_folders)
    save_btn = pnw.Button(
        name=f"Save images ({n_total} folder{'s' if n_total > 1 else ''})",
        button_type="primary", disabled=True, width=240,
    )

    matrix_pane = pn.pane.PNG(
        object=None, sizing_mode="stretch_width",
        min_height=300,
    )
    matrix_placeholder = pn.pane.Markdown(
        "_Select a threshold to see the D matrix comparison._",
        sizing_mode="stretch_width",
    )

    # show placeholder until first click
    matrix_col = pn.Column(matrix_placeholder, sizing_mode="stretch_width")

    def on_save(event):
        if state["indexes"] is None:
            return
        save_btn.name     = "Saving..."
        save_btn.disabled = True
        save_all(image_paths, extra_folders, state["indexes"], output_dir)
        save_btn.name     = "Saved ✓"
        save_btn.disabled = False
        out_dirs = [str(output_dir)] + [str(dst) for _, dst, _ in extra_folders]
        status_md.object = (
            f"**Saved** {len(state['indexes'])} images across "
            f"{n_total} folder(s):\n\n"
            + "\n\n".join(f"- `{d}`" for d in out_dirs)
        )

    save_btn.on_click(on_save)

    # wrap the matrix pane so it swaps in after first click
    def on_selection_done(attr, old, new):
        # called from bokeh; swap placeholder for real pane on first click
        if matrix_placeholder in matrix_col:
            matrix_col.clear()
            matrix_col.append(matrix_pane)

    # ---- Bokeh curve ----
    bokeh_curve = make_bokeh_curve(
        ths, lengths, all_indexes, D, image_paths,
        extra_folders, output_dir,
        matrix_pane, status_md, save_btn, state,
    )

    # hook the swap-in callback onto the source inside the figure
    bokeh_curve.renderers[1].data_source.selected.on_change(
        "indices", on_selection_done
    )

    curve_pane = pn.pane.Bokeh(bokeh_curve, sizing_mode="stretch_width")

    # ---- layout ----
    sidebar = pn.Column(
        pn.pane.Markdown("## VPR Sequence Sampler"),
        pn.pane.Markdown(
            f"**D:** `{D.shape[0]}×{D.shape[1]}`  |  "
            f"**Images:** {len(image_paths)}  |  "
            f"**Extra folders:** {len(extra_folders)}"
        ),
        pn.layout.Divider(),
        curve_pane,
        pn.layout.Divider(),
        save_btn,
        pn.layout.Divider(),
        status_md,
        width=640, margin=(10, 20),
    )

    main = pn.Column(
        pn.pane.Markdown("### D matrix comparison"),
        matrix_col,
        sizing_mode="stretch_both", margin=(10, 10),
    )

    return pn.Row(sidebar, main, sizing_mode="stretch_both")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

args         = parse_args()
D, img_paths = load_inputs(args.D_matrix, args.rgb_csv, args.img_root)
extra        = collect_extra_folders(args.img_root, args.extra_folders, D.shape[0])
output_dir   = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"D shape      : {D.shape}")
print(f"Images       : {len(img_paths)}")
print(f"Extra folders: {len(extra)}")

all_indexes, results = run_threshold_sweep(
    D, args.th_min, args.th_max, args.n_thresholds
)
ths, lengths = zip(*results)
print(f"Threshold range: {min(ths):.2f} – {max(ths):.2f}  |  "
      f"frames range: {min(lengths)} – {max(lengths)}")

app = build_app(D, img_paths, extra, all_indexes, results, output_dir)
app.servable()
