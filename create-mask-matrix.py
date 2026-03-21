import argparse
import os
from pathlib import Path

import numpy as np
from utilities import load_yaml

SCRIPT_LABEL = f"\033[35m[{os.path.basename(__file__)}]\033[0m "


def zero_diagonal_band(M, thickness=1):
    rows, cols = M.shape
    for offset in range(-thickness, thickness + 1):
        diag_indices = np.diag_indices(min(rows, cols))
        row_idx = diag_indices[0]
        col_idx = diag_indices[1] + offset
        # Filter out indices that go out of bounds
        mask = (col_idx >= 0) & (col_idx < cols)
        M[row_idx[mask], col_idx[mask]] = 0
    return M


def main():

    parser = argparse.ArgumentParser(
        description="Generate mask matrix from experiment config."
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
    GT_path = Path(exp_yaml["log_dir"]) / "GT.npy"
    M_path = Path(exp_yaml["log_dir"]) / "M.npy"

    # Create Mask
    GT = np.load(GT_path)
    M = np.ones_like(GT, dtype=np.uint8)
    M = zero_diagonal_band(M, thickness=40)

    # Save matrix
    np.save(M_path, M)


if __name__ == "__main__":
    main()
