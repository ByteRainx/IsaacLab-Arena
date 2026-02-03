# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Replay demos from an HDF5 file and re-record with fresh images."""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import (
    get_isaaclab_arena_cli_parser,
)
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    '--select_episodes',
    type=int,
    nargs='+',
    default=[],
    help='Episode indices to replay. Empty = all.',
)
parser.add_argument(
    '--dataset_name',
    type=str,
    default='vr_episode0.hdf5',
    help='HDF5 filename under ./demos to process.',
)
parser.add_argument(
    '--replay_mode',
    type=str,
    choices=['action', 'state'],
    default='state',
    help='Replay mode: action (use recorded actions, may drift) or state (use recorded states, exact replay).',
)
parser.add_argument(
    '--enable_pinocchio',
    action='store_true',
    default=False,
    help='Enable Pinocchio.',
)

add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os

import gymnasium as gym
import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.managers import DatasetExportMode
from isaaclab.envs.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
)
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Record camera observations each step."""

    def record_pre_step(self):
        camera_data = {}

        if 'camera_obs' in self._env.obs_buf:
            return 'camera_obs', self._env.obs_buf['camera_obs']

        if 'policy' in self._env.obs_buf:
            policy_obs = self._env.obs_buf['policy']
            if isinstance(policy_obs, dict):
                for key in [
                    'left_wrist_cam',
                    'right_wrist_cam',
                    'head_cam',
                    'robot_pov_cam_rgb',
                ]:
                    if key in policy_obs:
                        camera_data[key] = policy_obs[key]

        if camera_data:
            return 'camera_obs', camera_data
        return None


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Config for camera observation recorder."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder manager with camera obs."""

    record_pre_step_flat_camera_observations = (
        PreStepFlatCameraObservationsRecorderCfg()
    )


def main() -> None:
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    datasets_dir = './demos'
    input_dataset_path = os.path.join(datasets_dir, args_cli.dataset_name)
    if not os.path.exists(input_dataset_path):
        print(f'Dataset not found: {input_dataset_path}')
        return
    if not input_dataset_path.endswith('.hdf5'):
        print(f'Invalid dataset file (must be .hdf5): {input_dataset_path}')
        return
    if input_dataset_path.endswith('_with_images.hdf5'):
        print(f'Skipping already processed dataset: {input_dataset_path}')
        return

    output_dataset_path = input_dataset_path[: -len('.hdf5')] + '_with_images.hdf5'

    dataset_in = HDF5DatasetFileHandler()
    dataset_in.open(input_dataset_path)
    episode_count = dataset_in.get_num_episodes()
    if episode_count == 0:
        print(f'No episodes found in input dataset: {input_dataset_path}')
        dataset_in.close()
        return

    episode_indices = args_cli.select_episodes
    if len(episode_indices) == 0:
        episode_indices = list(range(episode_count))

    env_cfg.recorders = (
        ArenaEnvRecorderManagerCfg()
        if args_cli.enable_cameras
        else ActionStateRecorderManagerCfg()
    )
    output_dir = os.path.dirname(output_dataset_path) or '.'
    output_name = os.path.splitext(os.path.basename(output_dataset_path))[0]
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.terminations = {}
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()

    episode_names = list(dataset_in.get_episode_names())
    num_envs = args_cli.num_envs

    env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}

    def export_env_episode(env_id: int) -> None:
        episode = env.recorder_manager.get_episode(env_id)
        if not episode.is_empty():
            env.recorder_manager.export_episodes([env_id])

    use_state_replay = args_cli.replay_mode == 'state'
    if use_state_replay:
        print('[INFO] Using STATE replay mode (exact reproduction)')
    else:
        print('[INFO] Using ACTION replay mode (may drift due to IK)')

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            actions = torch.zeros(env.action_space.shape, device=env.device)
            has_next_data = False

            for env_id in range(num_envs):
                # Get next action (needed for recording even in state mode)
                env_next_action = env_episode_data_map[env_id].get_next_action()

                if env_next_action is None:
                    export_env_episode(env_id)

                    next_episode_index = None
                    while episode_indices:
                        next_episode_index = episode_indices.pop(0)
                        if next_episode_index < episode_count:
                            break
                        next_episode_index = None

                    if next_episode_index is not None:
                        print(f'[Episode {next_episode_index}] Loading...')
                        episode_data = dataset_in.load_episode(
                            episode_names[next_episode_index], env.device
                        )
                        env_episode_data_map[env_id] = episode_data
                        initial_state = episode_data.get_initial_state()
                        env.recorder_manager.reset([env_id])
                        # Restore exact recorded state (robot pose, object positions, etc.)
                        env.reset_to(
                            initial_state,
                            torch.tensor([env_id], device=env.device),
                            is_relative=True,
                        )
                        env_next_action = env_episode_data_map[env_id].get_next_action()
                        has_next_data = True
                    else:
                        continue
                else:
                    has_next_data = True

                actions[env_id] = env_next_action

            if not has_next_data:
                break

            if use_state_replay:
                # State replay: set robot state directly from recorded data
                for env_id in range(num_envs):
                    next_state = env_episode_data_map[env_id].get_next_state()
                    if next_state is not None:
                        env.scene.reset_to(
                            next_state,
                            env_ids=torch.tensor([env_id], device=env.device),
                            is_relative=True,
                        )
                env.scene.write_data_to_sim()
                env.sim.step(render=True)
                # Still record the action for the output file
                env.recorder_manager.record_pre_step()
            else:
                # Action replay: use IK (may drift)
                env.step(actions)

    for env_id in range(num_envs):
        export_env_episode(env_id)

    print('Replay complete. Output:', output_dataset_path)
    env.close()
    dataset_in.close()


if __name__ == '__main__':
    main()
    simulation_app.close()
