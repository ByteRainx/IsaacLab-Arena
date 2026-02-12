# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
Merge multiple single-episode HDF5 demo files into one combined HDF5 file.

Isaac Lab Mimic's annotate_demos.py and generate_dataset.py expect a single HDF5
file with multiple demo groups (data/demo_0, data/demo_1, ...).

This script takes individual episode files (e.g., vr_episode0.hdf5, vr_episode1.hdf5, ...)
and merges them into one file with the expected structure.

Usage:
    python merge_hdf5_demos.py
    python merge_hdf5_demos.py --input_dir /path/to/demos --output_file /path/to/merged.hdf5
    python merge_hdf5_demos.py --input_dir ./demos --output_file ./demos/merged_demos.hdf5 --max_episodes 10
"""

import argparse
import glob
import os
import re

import h5py
import numpy as np


def natural_sort_key(path: str) -> list:
    """Sort key for natural (human-friendly) sorting of filenames with numbers."""
    basename = os.path.basename(str(path))
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", basename)]


def find_original_hdf5_files(demos_dir: str) -> list[str]:
    """
    Find original (non-_with_images) HDF5 files in the demos directory.
    Excludes merged/annotated/generated files.
    """
    pattern = os.path.join(demos_dir, "vr_episode*.hdf5")
    all_files = glob.glob(pattern)
    # Exclude _with_images files
    original_files = [f for f in all_files if "_with_images" not in f]
    return sorted(original_files, key=natural_sort_key)


def copy_group_recursive(src_group: h5py.Group, dst_group: h5py.Group):
    """Recursively copy all datasets and subgroups from src to dst."""
    for key in src_group.keys():
        item = src_group[key]
        if isinstance(item, h5py.Dataset):
            data = item[:]
            dst_group.create_dataset(key, data=data)
            # Copy attributes
            for attr_name, attr_val in item.attrs.items():
                dst_group[key].attrs[attr_name] = attr_val
        elif isinstance(item, h5py.Group):
            sub_dst = dst_group.create_group(key)
            # Copy group attributes
            for attr_name, attr_val in item.attrs.items():
                sub_dst.attrs[attr_name] = attr_val
            copy_group_recursive(item, sub_dst)


def merge_hdf5_files(
    input_files: list[str],
    output_file: str,
    max_episodes: int | None = None,
) -> int:
    """
    Merge multiple HDF5 files into one combined file.

    Args:
        input_files: List of input HDF5 file paths.
        output_file: Output merged HDF5 file path.
        max_episodes: Maximum number of episodes to include. None = all.

    Returns:
        Number of episodes merged.
    """
    if max_episodes is not None:
        input_files = input_files[:max_episodes]

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    num_merged = 0

    with h5py.File(output_file, "w") as f_out:
        data_group = f_out.create_group("data")

        for idx, input_path in enumerate(input_files):
            demo_key = f"demo_{idx}"
            print(f"  [{idx}] {os.path.basename(input_path)} -> data/{demo_key}")

            try:
                with h5py.File(input_path, "r") as f_in:
                    # Find the demo group in the source file
                    if "data" not in f_in:
                        print(f"    WARNING: No 'data' group found, skipping.")
                        continue

                    src_data = f_in["data"]
                    # Get the first (usually only) demo key
                    src_demo_keys = sorted(src_data.keys())
                    if not src_demo_keys:
                        print(f"    WARNING: Empty 'data' group, skipping.")
                        continue

                    src_demo = src_data[src_demo_keys[0]]

                    # Create destination demo group
                    dst_demo = data_group.create_group(demo_key)

                    # Copy all data from source demo to destination
                    copy_group_recursive(src_demo, dst_demo)

                    # Copy demo-level attributes
                    for attr_name, attr_val in src_demo.attrs.items():
                        dst_demo.attrs[attr_name] = attr_val

                    # Report what was copied
                    keys = list(dst_demo.keys())
                    if "actions" in dst_demo:
                        n_steps = dst_demo["actions"].shape[0]
                        print(f"    Copied: {n_steps} steps, keys: {keys}")
                    else:
                        print(f"    Copied: keys: {keys}")

                    num_merged += 1

            except Exception as e:
                print(f"    ERROR: {e}, skipping.")
                continue

        # Store total count as attribute
        data_group.attrs["total"] = num_merged

    return num_merged


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple single-episode HDF5 demo files into one combined file."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./demos",
        help="Directory containing individual HDF5 episode files (default: ./demos)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output merged HDF5 file path (default: <input_dir>/merged_demos.hdf5)",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to merge (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_file = args.output_file or os.path.join(input_dir, "merged_demos.hdf5")

    print(f"Scanning for HDF5 episode files in: {input_dir}")
    input_files = find_original_hdf5_files(input_dir)

    if not input_files:
        print("No episode HDF5 files found.")
        return

    print(f"Found {len(input_files)} episode files")
    if args.max_episodes:
        print(f"Limiting to first {args.max_episodes} episodes")

    print(f"Output: {output_file}\n")

    num_merged = merge_hdf5_files(input_files, output_file, args.max_episodes)

    print(f"\nMerge complete!")
    print(f"  Episodes merged: {num_merged}")
    print(f"  Output file: {output_file}")

    # Verify
    with h5py.File(output_file, "r") as f:
        demo_keys = sorted(f["data"].keys())
        print(f"  Demo groups: {len(demo_keys)} ({demo_keys[0]} ... {demo_keys[-1]})")


if __name__ == "__main__":
    main()
