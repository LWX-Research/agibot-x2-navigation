#!/usr/bin/env python3
"""Merge FAST-LIO scans_*.pcd files into a downsampled XYZ point cloud."""

import argparse
import re
from pathlib import Path


DEFAULT_INPUT_DIR = (
    Path(__file__).resolve().parent / "src/spark-fast-lio/spark_fast_lio/PCD"
)


def natural_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Directory containing scans_*.pcd (default: this workspace's PCD directory)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output PCD path (default: INPUT_DIR/map_merged.pcd)",
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.05,
        help="Voxel size in metres; use 0 to disable downsampling (default: 0.05)",
    )
    args = parser.parse_args()
    if not 0 <= args.voxel_size < float("inf"):
        parser.error("--voxel-size must be a finite, non-negative number")

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")
    files = sorted(
        (path for path in input_dir.glob("scans_*.pcd") if path.is_file()),
        key=natural_key,
    )
    if not files:
        parser.error(f"No scans_*.pcd files found in {input_dir}")
    output = (args.output or input_dir / "map_merged.pcd").expanduser().resolve()
    if output in files:
        parser.error("Output must not overwrite an input scan")
    if output.suffix.lower() != ".pcd":
        parser.error("--output must have a .pcd extension")

    try:
        import open3d as o3d
    except ImportError:
        parser.error("Open3D is required: python3 -m pip install 'numpy<2' open3d")

    merged = o3d.geometry.PointCloud()
    for path in files:
        cloud = o3d.io.read_point_cloud(str(path))
        cloud.remove_non_finite_points()
        if cloud.is_empty():
            parser.error(f"Scan is unreadable or contains no finite points: {path}")
        print(f"Loading {path.name}: {len(cloud.points)} points")
        merged += cloud

    print(f"Total input points: {len(merged.points)}")
    if args.voxel_size:
        merged = merged.voxel_down_sample(args.voxel_size)
    print(f"Output points: {len(merged.points)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), merged, write_ascii=False, compressed=False):
        parser.error(f"Failed to write point cloud: {output}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
