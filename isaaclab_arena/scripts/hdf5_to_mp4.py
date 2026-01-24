# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
Convert HDF5 demonstration files to MP4 videos.

This script converts camera frames stored in HDF5 demonstration files to MP4
videos. It supports Arena-style datasets where frames are stored under
data/demo_*/obs/camera_obs/<camera_name>.
"""

import argparse
import os

import cv2
import h5py

DEFAULT_VIDEO_HEIGHT = 480
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_FRAMERATE = 30

DEFAULT_INPUT_KEYS = [
    "left_wrist_cam",
    "right_wrist_cam",
    "head_cam",
    "robot_pov_cam_rgb",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HDF5 demonstration files to MP4 videos."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input HDF5 file containing demonstration data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory path where the output MP4 files will be saved.",
    )
    parser.add_argument(
        "--input_keys",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_KEYS,
        help="List of camera keys to process.",
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help="Height of the output video in pixels.",
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help="Width of the output video in pixels.",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=DEFAULT_FRAMERATE,
        help="Frames per second for the output video.",
    )
    return parser.parse_args()


def get_num_demos(hdf5_file: str) -> int:
    """Get the number of demonstrations in the HDF5 file."""
    with h5py.File(hdf5_file, "r") as f:
        return len(f["data"].keys())


def _resolve_frames_path(demo_group: h5py.Group, input_key: str) -> str | None:
    """Resolve the dataset path for a given camera key."""
    if "camera_obs" in demo_group.get("obs", {}):
        cam_group = demo_group["obs"]["camera_obs"]
        if input_key in cam_group:
            return f"{demo_group.name}/obs/camera_obs/{input_key}"
    if "obs" in demo_group and input_key in demo_group["obs"]:
        return f"{demo_group.name}/obs/{input_key}"
    return None


def write_demo_to_mp4(
    hdf5_file: str,
    demo_id: int,
    input_key: str,
    output_dir: str,
    video_height: int,
    video_width: int,
    framerate: int,
) -> None:
    """Convert frames from an HDF5 file to an MP4 video."""
    with h5py.File(hdf5_file, "r") as f:
        demo_group = f[f"data/demo_{demo_id}"]
        frames_path = _resolve_frames_path(demo_group, input_key)
        if frames_path is None:
            print(f"[WARN] demo_{demo_id}: key '{input_key}' not found.")
            return

        frames = f[frames_path]
        output_path = os.path.join(output_dir, f"demo_{demo_id}_{input_key}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            output_path, fourcc, framerate, (video_width, video_height)
        )

        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(
                frame, (video_width, video_height), interpolation=cv2.INTER_CUBIC
            )
            video.write(frame)

        video.release()


def main():
    """Main function to convert all demonstrations to MP4 videos."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    num_demos = get_num_demos(args.input_file)
    print(f"Found {num_demos} demonstrations in {args.input_file}")

    for demo_id in range(num_demos):
        for input_key in args.input_keys:
            write_demo_to_mp4(
                args.input_file,
                demo_id,
                input_key,
                args.output_dir,
                args.video_height,
                args.video_width,
                args.framerate,
            )


if __name__ == "__main__":
    main()
