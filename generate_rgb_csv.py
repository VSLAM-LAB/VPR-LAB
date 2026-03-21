"""
generate_rgb_csv.py
--------------------
Reads a folder of RGB images (and optionally a paired depth folder) whose
filenames encode a nanosecond timestamp, then writes a rgb.csv file that
matches the format:

    ts_rgb_0 (ns),path_rgb_0,ts_depth_0 (ns),path_depth_0

Two filename conventions are supported automatically:

  1. Raw nanoseconds  : 1000048746833289347.png
                        → timestamp = 1000048746833289347
                        → CSV path  = rgb_0/1000048746833289347.png

  2. Dotted sec.usec  : 2989.023949.png
                        → timestamp = 2989023949000  (sec * 1e9 + usec * 1e3)
                        → CSV path  = rgb_0/2989.023949.png

Usage
-----
    python generate_rgb_csv.py \\
        --rgb_dir   path/to/rgb_0   \\
        --depth_dir path/to/depth_0 \\   # optional
        --output    rgb.csv             # default: rgb.csv

If --depth_dir is omitted the ts_depth_0 and path_depth_0 columns are left
empty (but the header is still written so downstream tools don't break).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

# Pattern A: pure integer stem  e.g. 1000048746833289347
_RE_RAW_NS = re.compile(r"^(\d+)$")

# Pattern B: dotted sec.usec    e.g. 2989.023949
_RE_DOT_SEC_USEC = re.compile(r"^(\d+)\.(\d{6})$")


def parse_timestamp_ns(stem: str) -> int | None:
    """
    Return the nanosecond timestamp encoded in a filename stem, or None if
    the stem does not match any known pattern.
    """
    m = _RE_RAW_NS.match(stem)
    if m:
        return int(m.group(1))

    m = _RE_DOT_SEC_USEC.match(stem)
    if m:
        seconds = int(m.group(1))
        microseconds = int(m.group(2))
        return seconds * 1_000_000_000 + microseconds * 1_000

    return None


# ---------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def collect_images(folder: Path) -> list[tuple[int, Path]]:
    """
    Walk *folder* (non-recursively) and return a sorted list of
    (timestamp_ns, filepath) for every image whose stem encodes a timestamp.
    """
    results: list[tuple[int, Path]] = []

    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        ts = parse_timestamp_ns(p.stem)
        if ts is None:
            print(f"  [skip] {p.name}  — unrecognised filename pattern", file=sys.stderr)
            continue
        results.append((ts, p))

    results.sort(key=lambda x: x[0])
    return results


# ---------------------------------------------------------------------------
# CSV generation
# ---------------------------------------------------------------------------

def build_csv(
    rgb_folder: Path,
    depth_folder: Path | None,
    output_path: Path,
) -> None:
    print(f"Scanning RGB folder : {rgb_folder}")
    rgb_images = collect_images(rgb_folder)
    if not rgb_images:
        print("ERROR: no recognisable image files found in RGB folder.", file=sys.stderr)
        sys.exit(1)
    print(f"  Found {len(rgb_images)} RGB images.")

    depth_map: dict[int, Path] = {}
    if depth_folder is not None:
        print(f"Scanning depth folder: {depth_folder}")
        depth_images = collect_images(depth_folder)
        depth_map = {ts: p for ts, p in depth_images}
        print(f"  Found {len(depth_images)} depth images.")

    # Determine the relative folder names to use in the CSV paths.
    # We replicate the convention seen in the sample: "rgb_0/stem.png"
    rgb_dir_name   = rgb_folder.name   # e.g. "rgb_0"
    depth_dir_name = depth_folder.name if depth_folder else "depth_0"

    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["ts_rgb_0 (ns)", "path_rgb_0", "ts_depth_0 (ns)", "path_depth_0"]
        )

        matched = 0
        unmatched = 0
        for ts_rgb, rgb_path in rgb_images:
            rgb_csv_path = f"{rgb_dir_name}/{rgb_path.name}"

            if depth_map:
                depth_path = depth_map.get(ts_rgb)
                if depth_path is not None:
                    depth_csv_path = f"{depth_dir_name}/{depth_path.name}"
                    matched += 1
                else:
                    # No depth frame at exactly this timestamp — leave blank
                    depth_csv_path = ""
                    unmatched += 1
                writer.writerow([ts_rgb, rgb_csv_path, ts_rgb, depth_csv_path])
            else:
                # No depth folder supplied
                writer.writerow([ts_rgb, rgb_csv_path, "", ""])

    print(f"\nWrote {len(rgb_images)} rows → {output_path}")
    if depth_map:
        print(f"  Depth matched : {matched}")
        if unmatched:
            print(f"  Depth missing : {unmatched}  (cells left empty)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rgb.csv from a folder of timestamped images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--rgb_dir",
        required=True,
        type=Path,
        help="Folder containing RGB images (filenames = nanosecond timestamps).",
    )
    parser.add_argument(
        "--depth_dir",
        type=Path,
        default=None,
        help="Optional folder containing depth images with matching timestamps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rgb.csv"),
        help="Output CSV file path (default: rgb.csv).",
    )
    args = parser.parse_args()

    if not args.rgb_dir.is_dir():
        print(f"ERROR: --rgb_dir '{args.rgb_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    if args.depth_dir is not None and not args.depth_dir.is_dir():
        print(
            f"ERROR: --depth_dir '{args.depth_dir}' is not a directory.", file=sys.stderr
        )
        sys.exit(1)

    build_csv(args.rgb_dir, args.depth_dir, args.output)


if __name__ == "__main__":
    main()
