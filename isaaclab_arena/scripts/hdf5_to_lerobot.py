# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Convert HDF5 demonstration files to LeRobot v2.1 dataset format.

Reads HDF5 files produced by record_ex001_remote_demos.py (with --enable_cameras),
where each HDF5 file is a single episode containing actions, camera images, and
state observations.

Output format follows LeRobot v2.1 standard (per-episode files):
  data/chunk-000/episode_000000.parquet  (one per episode)
  videos/chunk-000/observation.images.faceImg/episode_000000.mp4  (per camera)
  meta/info.json, episodes.jsonl, tasks.jsonl

Camera naming aligned with convert2lerobot.py:
  head_cam -> faceImg, left_wrist_cam -> leftImg, right_wrist_cam -> rightImg

State mode (--state_mode):
  joints -- observation.state = previous frame's action (joint positions)
  ee     -- observation.state = end-effector pose [pos(3)+euler(3)+gripper(1)] x 2 arms

Auto-detects and skips duplicate initial frames.

Usage:
    python hdf5_to_lerobot.py --state_mode joints
    python hdf5_to_lerobot.py --state_mode ee --input_dir /path/to/data
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
from scipy.spatial.transform import Rotation as R

# ============================================================
# Configuration - aligned with convert2lerobot.py
# ============================================================
FPS = 50
REPO_ID = 'isaaclab_arena_demos'
DEFAULT_TASK = 'put_blocks_to_color'
CHUNKS_SIZE = 1000

CAMERA_NAME_MAPPING = {
    'head_cam': 'faceImg',
    'left_wrist_cam': 'leftImg',
    'right_wrist_cam': 'rightImg',
}

DATA_PATH_TEMPLATE = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
VIDEO_PATH_TEMPLATE = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'


# ============================================================
# Utility functions
# ============================================================
def natural_sort_key(path: str) -> list:
    """Sort key for natural (human-friendly) sorting of filenames with numbers."""
    basename = os.path.basename(str(path))
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', basename)]


def find_hdf5_files(data_dir: Path) -> list[Path]:
    """Find all HDF5 files in the data directory, sorted naturally."""
    pattern = str(data_dir / '*.hdf5')
    files = sorted(glob.glob(pattern), key=natural_sort_key)
    return [Path(f) for f in files]


def find_data_demo(f: h5py.File) -> str:
    """Find the demo group key that contains actual trajectory data."""
    data_group = f['data']
    for key in sorted(data_group.keys()):
        demo = data_group[key]
        if 'actions' in demo or 'camera_obs' in demo or 'obs' in demo:
            return key
    raise ValueError('No valid demo group found in HDF5 file')


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


def get_video_metadata(video_path: str) -> dict | None:
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=height,width,codec_name,pix_fmt,r_frame_rate',
        '-of', 'json', str(video_path),
    ]
    try:
        output = subprocess.check_output(cmd).decode('utf-8')
        probe_data = json.loads(output)
        stream = probe_data['streams'][0]
        height = int(stream['height'])
        width = int(stream['width'])
        num, den = map(int, stream['r_frame_rate'].split('/'))
        fps = num / den
        return {
            'dtype': 'video',
            'shape': [height, width, 3],
            'names': ['height', 'width', 'channels'],
            'video_info': {
                'video.height': height,
                'video.width': width,
                'video.fps': fps,
                'video.codec': stream['codec_name'],
                'video.pix_fmt': stream['pix_fmt'],
                'video.channels': 3,
                'video.is_depth_map': False,
                'has_audio': False,
            },
        }
    except Exception as e:
        print(f'  WARNING: ffprobe failed for {video_path}: {e}')
        return None


def _quat_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to euler angles (roll, pitch, yaw)."""
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], axis=1)
    return R.from_quat(quat_xyzw).as_euler('xyz').astype(np.float32)


def extract_episode_data(hdf5_path: Path, state_mode: str = 'joints', min_skip: int = 1) -> dict:
    """Extract episode data from a single HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file.
        state_mode: 'joints' uses previous frame's action as state;
                    'ee' uses end-effector pose [pos+euler+gripper] x 2 arms.
        min_skip: Minimum number of initial frames to skip.

    Returns dict with: actions, state, cameras, num_frames, skip_n
    """
    with h5py.File(hdf5_path, 'r') as f:
        demo_key = find_data_demo(f)
        demo = f[f'data/{demo_key}']

        cam_group = demo['camera_obs'] if 'camera_obs' in demo else None
        obs_group = demo['obs'] if 'obs' in demo else demo

        # Auto-detect duplicate initial frames using a reference camera
        ref_cam_key = None
        ref_cam_source = None
        for hdf5_key in CAMERA_NAME_MAPPING:
            if cam_group is not None and hdf5_key in cam_group:
                ref_cam_key = hdf5_key
                ref_cam_source = cam_group
                break
            if hdf5_key in obs_group:
                ref_cam_key = hdf5_key
                ref_cam_source = obs_group
                break

        if ref_cam_key is not None and ref_cam_source is not None:
            ref_cam_data = ref_cam_source[ref_cam_key][:]
            skip_n = count_duplicate_initial_frames(ref_cam_data)
            skip_n = max(skip_n, min_skip)
            total_frames = ref_cam_source[ref_cam_key].shape[0]
        else:
            skip_n = min_skip
            total_frames = demo['actions'].shape[0]

        n_frames = total_frames - skip_n

        actions = demo['actions'][skip_n:].astype(np.float32)

        if state_mode == 'joints':
            # state[t] = action[t-1]; state[0] = action[0] (no previous frame)
            state = np.empty_like(actions)
            state[0] = actions[0]
            if len(actions) > 1:
                state[1:] = actions[:-1]
        elif state_mode == 'ee':
            # [eef_pos(3), euler(3), gripper(1)] x 2 arms = 14D
            state_parts = []
            has_left = (
                'eef_pos' in obs_group
                and 'eef_quat' in obs_group
                and 'gripper_pos' in obs_group
            )
            has_right = (
                'right_eef_pos' in obs_group
                and 'right_eef_quat' in obs_group
                and 'right_gripper_pos' in obs_group
            )

            if has_left:
                state_parts.append(obs_group['eef_pos'][skip_n:].astype(np.float32))
                state_parts.append(_quat_to_euler(obs_group['eef_quat'][skip_n:].astype(np.float32)))
                state_parts.append(obs_group['gripper_pos'][skip_n:].astype(np.float32))

            if has_right:
                state_parts.append(obs_group['right_eef_pos'][skip_n:].astype(np.float32))
                state_parts.append(_quat_to_euler(obs_group['right_eef_quat'][skip_n:].astype(np.float32)))
                state_parts.append(obs_group['right_gripper_pos'][skip_n:].astype(np.float32))

            if state_parts:
                state = np.concatenate(state_parts, axis=1)
            else:
                raise ValueError(
                    f'state_mode="ee" requires eef_pos/eef_quat/gripper_pos in obs, '
                    f'available keys: {list(obs_group.keys())}'
                )
        else:
            raise ValueError(f'Unknown state_mode: {state_mode!r}. Use "joints" or "ee".')

        # Camera observations (prefer camera_obs group, fallback to obs)
        cameras = {}
        for hdf5_key, lerobot_key in CAMERA_NAME_MAPPING.items():
            if cam_group is not None and hdf5_key in cam_group:
                cameras[lerobot_key] = cam_group[hdf5_key][skip_n:]
            elif hdf5_key in obs_group:
                cameras[lerobot_key] = obs_group[hdf5_key][skip_n:]

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
        'actions': actions,
        'state': state,
        'cameras': cameras,
        'num_frames': num_frames,
        'skip_n': skip_n,
    }


def write_episode_video(frames: np.ndarray, video_path: Path, fps: int) -> None:
    """Write frames to mp4 video using torchvision (h264 codec)."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frames_tensor = torch.from_numpy(frames)
    torchvision.io.write_video(str(video_path), frames_tensor, fps, video_codec='h264')


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

    data = {
        'observation.state': list(state),
        'action': list(actions),
        'timestamp': np.arange(num_frames, dtype=np.float32) / FPS,
        'episode_index': np.full(num_frames, episode_index, dtype=np.int64),
        'index': np.arange(global_index_start, global_index_start + num_frames, dtype=np.int64),
        'frame_index': np.arange(num_frames, dtype=np.int64),
        'task_index': np.full(num_frames, task_index, dtype=np.int64),
    }

    df = pd.DataFrame(data)
    df.to_parquet(parquet_path)
    return parquet_path


# ============================================================
# Main
# ============================================================
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert HDF5 demonstration files to LeRobot v2.1 dataset format.'
    )
    parser.add_argument(
        '--input_dir', type=str, default='./data',
        help='Directory containing HDF5 files (default: ./data)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./lerobot_data',
        help='Output directory for LeRobot dataset (default: ./lerobot_data)',
    )
    parser.add_argument(
        '--task', type=str, default=DEFAULT_TASK,
        help='Task description for all episodes',
    )
    parser.add_argument(
        '--state_mode', type=str, default='joints', choices=['joints', 'ee'],
        help="State representation: 'joints' (prev action as state) or "
             "'ee' (end-effector pose). Default: joints",
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite existing output directory',
    )
    return parser.parse_args()


def main():
    """Convert HDF5 demonstrations to LeRobot v2.1 dataset."""
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    task = args.task
    state_mode = args.state_mode

    # --- Discover HDF5 files ---
    print(f'Scanning for HDF5 files in: {input_dir}')
    hdf5_files = find_hdf5_files(input_dir)
    if not hdf5_files:
        print('No HDF5 files found.')
        return

    print(f'Found {len(hdf5_files)} HDF5 files:')
    for f in hdf5_files:
        print(f'  - {f.name}')

    # --- Handle output directory ---
    if output_dir.exists():
        if args.force:
            print(f'\n--force: Removing existing output: {output_dir}')
            shutil.rmtree(output_dir)
        else:
            print(f'\nERROR: Output directory already exists: {output_dir}')
            print('  Use --force to overwrite, or specify a different --output_dir')
            return

    meta_dir = output_dir / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    # --- Analyze first file for dimensions ---
    first_data = extract_episode_data(hdf5_files[0], state_mode=state_mode)
    action_dim = first_data['actions'].shape[1]
    state_dim = first_data['state'].shape[1]
    available_cameras = list(first_data['cameras'].keys())
    cam0_shape = first_data['cameras'][available_cameras[0]].shape[1:] if available_cameras else (480, 640, 3)

    print('\nDataset configuration:')
    print(f'  FPS:          {FPS}')
    print(f'  State mode:   {state_mode}')
    print(f'  Action dim:   {action_dim}')
    print(f'  State dim:    {state_dim}')
    print(f'  Video shape:  {cam0_shape}')
    print(f'  Cameras:      {available_cameras}')
    print(f'  Task:         {task}')
    print(f'  Output:       {output_dir}')
    print('  Format:       LeRobot v2.1 (per-episode files)')

    # --- Process each HDF5 file as one episode ---
    episodes_info = []
    total_frames = 0
    video_meta_cache = {}

    for episode_index, hdf5_path in enumerate(hdf5_files):
        print(f'\n[Episode {episode_index}] {hdf5_path.name}')

        try:
            ep_data = extract_episode_data(hdf5_path, state_mode=state_mode)
        except Exception as e:
            print(f'  ERROR: {e}, skipping.')
            continue

        num_frames = ep_data['num_frames']
        skip_n = ep_data['skip_n']

        if num_frames <= 0:
            print('  WARNING: No valid frames, skipping.')
            continue

        actions = ep_data['actions']
        state = ep_data['state']
        cameras = ep_data['cameras']

        print(f'  Skipped initial frames: {skip_n}')
        print(f'  Usable frames: {num_frames}')
        print(f'  Action range: [{actions.min():.4f}, {actions.max():.4f}]')

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
            video_key = f'observation.images.{cam_key}'
            video_rel = VIDEO_PATH_TEMPLATE.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_index,
            )
            video_path = output_dir / video_rel
            write_episode_video(cam_frames, video_path, FPS)

            if video_key not in video_meta_cache:
                meta = get_video_metadata(str(video_path))
                if meta:
                    video_meta_cache[video_key] = meta

        episodes_info.append({
            'episode_index': episode_index,
            'tasks': [task],
            'length': num_frames,
        })
        total_frames += num_frames
        print(f'  Saved. (episode {episode_index})')

    # --- Write meta/tasks.jsonl ---
    tasks_path = meta_dir / 'tasks.jsonl'
    with open(tasks_path, 'w') as f:
        f.write(json.dumps({'task_index': 0, 'task': task}) + '\n')

    # --- Write meta/episodes.jsonl ---
    episodes_path = meta_dir / 'episodes.jsonl'
    with open(episodes_path, 'w') as f:
        for ep in episodes_info:
            f.write(json.dumps(ep) + '\n')

    # --- Build features dict (v2.1 compliant) ---
    features = {}
    for video_key, meta in video_meta_cache.items():
        features[video_key] = meta

    features['observation.state'] = {
        'dtype': 'float32',
        'shape': [state_dim],
        'names': None,
    }
    features['action'] = {
        'dtype': 'float32',
        'shape': [action_dim],
        'names': None,
    }
    features['timestamp'] = {'dtype': 'float32', 'shape': [1], 'names': None}
    features['frame_index'] = {'dtype': 'int64', 'shape': [1], 'names': None}
    features['episode_index'] = {'dtype': 'int64', 'shape': [1], 'names': None}
    features['index'] = {'dtype': 'int64', 'shape': [1], 'names': None}
    features['task_index'] = {'dtype': 'int64', 'shape': [1], 'names': None}

    # --- Write meta/info.json ---
    num_episodes = len(episodes_info)
    info = {
        'codebase_version': 'v2.1',
        'robot_type': None,
        'total_episodes': num_episodes,
        'total_frames': total_frames,
        'total_tasks': 1,
        'total_videos': num_episodes * len(available_cameras),
        'total_chunks': (num_episodes - 1) // CHUNKS_SIZE + 1 if num_episodes > 0 else 0,
        'chunks_size': CHUNKS_SIZE,
        'fps': FPS,
        'splits': {'train': f'0:{num_episodes}'},
        'data_path': DATA_PATH_TEMPLATE,
        'video_path': VIDEO_PATH_TEMPLATE if available_cameras else None,
        'features': features,
    }
    info_path = meta_dir / 'info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)

    print('\n' + '=' * 60)
    print('Conversion complete!')
    print('  Format:     LeRobot v2.1')
    print(f'  State mode: {state_mode}')
    print(f'  Output:     {output_dir}')
    print(f'  Episodes:   {num_episodes}')
    print(f'  Frames:     {total_frames}')
    print(f'  Action dim: {action_dim}')
    print(f'  State dim:  {state_dim}')
    print(f'  FPS:        {FPS}')
    print(f'  Cameras:    {available_cameras}')
    print('=' * 60)


if __name__ == '__main__':
    main()
