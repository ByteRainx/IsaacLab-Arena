# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""
Convert HDF5 demonstration files to LeRobot v2.1 dataset format.

Output format follows GR00T-LeRobot v2.1 standard (per-episode files):
  data/chunk-000/episode_000000.parquet  (one per episode)
  videos/chunk-000/observation.images.faceImg/episode_000000.mp4  (one per episode per camera)
  meta/info.json, episodes.jsonl, tasks.jsonl

Camera naming aligned with convert2lerobot.py:
  head_cam -> faceImg, left_wrist_cam -> leftImg, right_wrist_cam -> rightImg

Actions read from original vr_episodeX.hdf5 (state-replay _with_images has zero actions).
Camera images + state read from _with_images.hdf5.
Auto-detects and skips duplicate initial frames.

Usage:
    python hdf5_to_lerobot.py
    python hdf5_to_lerobot.py --input_dir /path/to/demos --output_dir /path/to/output
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision

# ============================================================
# Configuration - aligned with convert2lerobot.py
# ============================================================
FPS = 50  # Simulation step rate: 1 / (dt * decimation) = 1 / (0.005 * 4) = 50Hz
REPO_ID = "isaaclab_arena_demos"
DEFAULT_TASK = "put_blocks_to_color"
CHUNKS_SIZE = 1000  # Episodes per chunk (v2.1 standard)

# Camera name mapping: HDF5 key → LeRobot key (aligned with convert2lerobot.py)
CAMERA_NAME_MAPPING = {
    "head_cam": "faceImg",
    "left_wrist_cam": "leftImg",
    "right_wrist_cam": "rightImg",
}

# State observation keys (order determines observation.state layout)
STATE_KEYS = [
    "joint_pos",          # (18,)
    "eef_pos",            # (3,)
    "eef_quat",           # (4,)
    "gripper_pos",        # (1,)
    "right_eef_pos",      # (3,)
    "right_eef_quat",     # (4,)
    "right_gripper_pos",  # (1,)
]
# Total state dim = 34

# v2.1 path templates
DATA_PATH_TEMPLATE = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH_TEMPLATE = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ============================================================
# Utility functions
# ============================================================
def natural_sort_key(path: str) -> list:
    """Sort key for natural (human-friendly) sorting of filenames with numbers."""
    basename = os.path.basename(str(path))
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]


def find_hdf5_file_pairs(demos_dir: Path) -> list[dict]:
    """
    Find matching pairs: original HDF5 (actions) + _with_images HDF5 (cameras + state).
    """
    img_pattern = str(demos_dir / "*_with_images.hdf5")
    img_files = sorted(glob.glob(img_pattern), key=natural_sort_key)

    pairs = []
    for img_path in img_files:
        orig_path = img_path.replace("_with_images.hdf5", ".hdf5")
        name = os.path.basename(orig_path).replace(".hdf5", "")

        if os.path.exists(orig_path):
            pairs.append({
                "original": Path(orig_path),
                "with_images": Path(img_path),
                "name": name,
            })
        else:
            print(f"  WARNING: No original HDF5 for {os.path.basename(img_path)}, skipping.")

    return pairs


def find_data_demo(f: h5py.File) -> str:
    """Find the demo group key that contains actual trajectory data."""
    data_group = f["data"]
    for key in sorted(data_group.keys()):
        demo = data_group[key]
        if "actions" in demo or "camera_obs" in demo or "obs" in demo:
            return key
    raise ValueError("No valid demo group found in HDF5 file")


def count_duplicate_initial_frames(cam_data: np.ndarray, max_check: int = 30) -> int:
    """Count how many initial frames are identical (duplicates of the first frame)."""
    if len(cam_data) < 2:
        return 0
    frame0 = cam_data[0]
    n_dup = 1
    for i in range(1, min(max_check, len(cam_data))):
        if np.array_equal(cam_data[i], frame0):
            n_dup += 1
        else:
            break
    return n_dup


def get_video_metadata(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of", "json", str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd).decode("utf-8")
        probe_data = json.loads(output)
        stream = probe_data["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split("/"))
        fps = num / den
        return {
            "dtype": "video",
            "shape": [stream["height"], stream["width"], 3],
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.width": stream["width"],
                "video.height": stream["height"],
                "video.fps": fps,
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    except Exception as e:
        print(f"  WARNING: ffprobe failed for {video_path}: {e}")
        return None


def extract_episode_data(
    original_path: Path,
    with_images_path: Path,
    min_skip: int = 1,
) -> dict:
    """
    Extract episode data from paired HDF5 files:
      - Actions from original HDF5
      - Camera images + state from _with_images HDF5

    Returns dict with: actions, state, cameras, num_frames, skip_n
    """
    with h5py.File(with_images_path, "r") as f_img:
        img_demo_key = find_data_demo(f_img)
        img_demo = f_img[f"data/{img_demo_key}"]

        cam_group = img_demo["camera_obs"] if "camera_obs" in img_demo else img_demo.get("obs", img_demo)

        # Auto-detect duplicate initial frames
        ref_cam_key = None
        for hdf5_key in CAMERA_NAME_MAPPING:
            if hdf5_key in cam_group:
                ref_cam_key = hdf5_key
                break

        if ref_cam_key is not None:
            ref_cam_data = cam_group[ref_cam_key][:]
            skip_n = count_duplicate_initial_frames(ref_cam_data)
            skip_n = max(skip_n, min_skip)
        else:
            skip_n = min_skip

        total_frames_img = cam_group[ref_cam_key].shape[0] if ref_cam_key else 0

        # State observations
        obs_group = img_demo["obs"] if "obs" in img_demo else img_demo
        state_parts = []
        for key in STATE_KEYS:
            if key in obs_group:
                state_parts.append(obs_group[key][skip_n:].astype(np.float32))
        if state_parts:
            state = np.concatenate(state_parts, axis=1)
        else:
            state = np.zeros((total_frames_img - skip_n, 0), dtype=np.float32)

        # Camera observations
        cameras = {}
        for hdf5_key, lerobot_key in CAMERA_NAME_MAPPING.items():
            if hdf5_key in cam_group:
                cameras[lerobot_key] = cam_group[hdf5_key][skip_n:]

    # Actions from original HDF5
    with h5py.File(original_path, "r") as f_orig:
        orig_demo_key = find_data_demo(f_orig)
        orig_demo = f_orig[f"data/{orig_demo_key}"]
        actions = orig_demo["actions"][skip_n:].astype(np.float32)

    # Ensure frame counts match
    num_frames = min(len(actions), len(state))
    if cameras:
        first_cam = list(cameras.values())[0]
        num_frames = min(num_frames, len(first_cam))

    actions = actions[:num_frames]
    state = state[:num_frames]
    for k in cameras:
        cameras[k] = cameras[k][:num_frames]

    return {
        "actions": actions,
        "state": state,
        "cameras": cameras,
        "num_frames": num_frames,
        "skip_n": skip_n,
    }


def write_episode_video(frames: np.ndarray, video_path: Path, fps: int) -> None:
    """Write frames to mp4 video using torchvision (h264 codec, matching GR00T convert)."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    # frames: (N, H, W, 3) uint8 RGB -> torch tensor (N, H, W, 3) uint8
    frames_tensor = torch.from_numpy(frames)
    torchvision.io.write_video(str(video_path), frames_tensor, fps, video_codec="h264")


def write_episode_parquet(
    episode_index: int,
    global_index_start: int,
    actions: np.ndarray,
    state: np.ndarray,
    num_frames: int,
    task_index: int,
    output_dir: Path,
) -> Path:
    """Write per-episode parquet file in v2.1 format."""
    episode_chunk = episode_index // CHUNKS_SIZE
    rel_path = DATA_PATH_TEMPLATE.format(episode_chunk=episode_chunk, episode_index=episode_index)
    parquet_path = output_dir / rel_path
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Build data dict
    data = {
        "observation.state": [row for row in state],
        "action": [row for row in actions],
        "timestamp": np.arange(num_frames, dtype=np.float64) / FPS,
        "episode_index": np.full(num_frames, episode_index, dtype=np.int64),
        "index": np.arange(global_index_start, global_index_start + num_frames, dtype=np.int64),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "task_index": np.full(num_frames, task_index, dtype=np.int64),
        "next.reward": np.zeros(num_frames, dtype=np.float64),
        "next.done": np.zeros(num_frames, dtype=bool),
    }
    # Last frame is done
    data["next.reward"][-1] = 1.0
    data["next.done"][-1] = True

    df = pd.DataFrame(data)
    df.to_parquet(parquet_path)
    return parquet_path


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 demonstration files to LeRobot v2.1 dataset format."
    )
    parser.add_argument(
        "--input_dir", type=str, default="./demos",
        help="Directory containing HDF5 files (default: ./demos)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for LeRobot dataset (default: <input_dir>/lerobot_dataset)",
    )
    parser.add_argument(
        "--task", type=str, default=DEFAULT_TASK,
        help="Task description for all episodes",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force overwrite existing output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "lerobot_dataset"
    task = args.task

    # --- Discover HDF5 file pairs ---
    print(f"Scanning for HDF5 file pairs in: {input_dir}")
    pairs = find_hdf5_file_pairs(input_dir)
    if not pairs:
        print("No matching HDF5 file pairs found.")
        return

    print(f"Found {len(pairs)} episode pairs:")
    for p in pairs:
        print(f"  - {p['name']}")

    # --- Handle output directory ---
    if output_dir.exists():
        if args.force:
            print(f"\n--force: Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"\nERROR: Output directory already exists: {output_dir}")
            print("  Use --force to overwrite, or specify a different --output_dir")
            return

    # --- Create directory structure ---
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- Analyze first file for dimensions ---
    first_data = extract_episode_data(pairs[0]["original"], pairs[0]["with_images"])
    action_dim = first_data["actions"].shape[1]
    state_dim = first_data["state"].shape[1]
    available_cameras = list(first_data["cameras"].keys())
    cam0_shape = first_data["cameras"][available_cameras[0]].shape[1:] if available_cameras else (480, 640, 3)

    print("\nDataset configuration:")
    print(f"  FPS:          {FPS}")
    print(f"  Action dim:   {action_dim}")
    print(f"  State dim:    {state_dim}")
    print(f"  Video shape:  {cam0_shape}")
    print(f"  Cameras:      {available_cameras}")
    print(f"  Task:         {task}")
    print(f"  Output:       {output_dir}")
    print("  Format:       LeRobot v2.1 (per-episode files)")

    # --- Process each HDF5 pair as one episode ---
    episodes_info = []
    total_frames = 0
    video_meta_cache = {}

    for episode_index, pair in enumerate(pairs):
        print(f"\n[Episode {episode_index}] {pair['name']}")

        try:
            ep_data = extract_episode_data(pair["original"], pair["with_images"])
        except Exception as e:
            print(f"  ERROR: {e}, skipping.")
            continue

        num_frames = ep_data["num_frames"]
        skip_n = ep_data["skip_n"]

        if num_frames <= 0:
            print("  WARNING: No valid frames, skipping.")
            continue

        actions = ep_data["actions"]
        state = ep_data["state"]
        cameras = ep_data["cameras"]

        print(f"  Skipped initial frames: {skip_n}")
        print(f"  Usable frames: {num_frames}")
        print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")

        # --- Write per-episode parquet ---
        write_episode_parquet(
            episode_index=episode_index,
            global_index_start=total_frames,
            actions=actions,
            state=state,
            num_frames=num_frames,
            task_index=0,
            output_dir=output_dir,
        )

        # --- Write per-episode videos ---
        episode_chunk = episode_index // CHUNKS_SIZE
        for cam_key, cam_frames in cameras.items():
            video_key = f"observation.images.{cam_key}"
            video_rel = VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            video_path = output_dir / video_rel
            write_episode_video(cam_frames, video_path, FPS)

            # Cache video metadata from first episode
            if video_key not in video_meta_cache:
                meta = get_video_metadata(str(video_path))
                if meta:
                    video_meta_cache[video_key] = meta

        # --- Track episode info ---
        episodes_info.append({
            "episode_index": episode_index,
            "tasks": [task],
            "length": num_frames,
        })
        total_frames += num_frames
        print(f"  Saved. (episode {episode_index})")

    # --- Write meta/tasks.jsonl ---
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    # --- Write meta/episodes.jsonl ---
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in episodes_info:
            f.write(json.dumps(ep) + "\n")

    # --- Build features dict ---
    features = {}
    # Video features (from cached metadata)
    for video_key, meta in video_meta_cache.items():
        features[video_key] = meta

    # State/action features
    features["observation.state"] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": None,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": [action_dim],
        "names": None,
    }
    features["timestamp"] = {"dtype": "float64", "shape": [1]}
    features["episode_index"] = {"dtype": "int64", "shape": [1]}
    features["index"] = {"dtype": "int64", "shape": [1]}
    features["frame_index"] = {"dtype": "int64", "shape": [1]}
    features["task_index"] = {"dtype": "int64", "shape": [1]}
    features["next.reward"] = {"dtype": "float64", "shape": [1]}
    features["next.done"] = {"dtype": "bool", "shape": [1]}

    # --- Write meta/info.json ---
    num_episodes = len(episodes_info)
    info = {
        "codebase_version": "v2.1",
        "robot_type": None,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes,
        "total_chunks": num_episodes // CHUNKS_SIZE,
        "chunks_size": CHUNKS_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": DATA_PATH_TEMPLATE,
        "video_path": VIDEO_PATH_TEMPLATE,
        "features": features,
    }
    info_path = meta_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print("\n" + "=" * 60)
    print("Conversion complete!")
    print("  Format:     LeRobot v2.1")
    print(f"  Output:     {output_dir}")
    print(f"  Episodes:   {num_episodes}")
    print(f"  Frames:     {total_frames}")
    print(f"  FPS:        {FPS}")
    print(f"  Cameras:    {available_cameras}")
    print("=" * 60)


if __name__ == "__main__":
    main()
