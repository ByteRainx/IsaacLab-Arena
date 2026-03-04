# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Replay demonstrations from HDF5 in simulation only.

Replay modes:
  - action: use recorded `actions`
  - joint: use `obs/joint_pos` converted to 14D joint action
  - ee: use `obs/eef_state` (if present) or absolute EE pose fields
  - state: exact replay by applying recorded runtime states
"""

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to replay. Keep empty to replay all episodes in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to replay.")
parser.add_argument(
    "--replay_mode",
    type=str,
    choices=["action", "joint", "ee", "state"],
    default="action",
    help=(
        "Replay source: action (recorded actions), joint (obs/joint_pos), "
        "ee (obs/eef_state or eef pose), state (recorded runtime states)."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate dataset states against runtime states. Only valid if --num_envs is 1 and replay_mode=action."
    ),
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import h5py
import os
import torch
import gymnasium as gym

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_tasks  # noqa: F401

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def _map_action_dim(action: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if action.shape[-1] == expected_dim:
        return action
    if expected_dim < action.shape[-1]:
        return action[:expected_dim]
    mapped = torch.zeros((expected_dim,), device=action.device, dtype=action.dtype)
    mapped[: action.shape[-1]] = action
    return mapped


def _joint_obs_to_action(joint_pos_step: torch.Tensor) -> torch.Tensor:
    """Convert observed joint positions to 14D joint control action.

    Expected common layout (18D):
      [l_arm6, l_gripper, l_mimic2, r_arm6, r_gripper, r_mimic2]
    """
    if joint_pos_step.numel() >= 16:
        left_arm = joint_pos_step[0:6]
        left_gripper = joint_pos_step[6:7]
        right_arm = joint_pos_step[9:15]
        right_gripper = joint_pos_step[15:16]
        return torch.cat([left_arm, left_gripper, right_arm, right_gripper], dim=0)
    if joint_pos_step.numel() >= 14:
        return joint_pos_step[:14]
    raise ValueError(f"joint_pos dimension too small: {joint_pos_step.numel()}")


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> tuple[bool, str]:
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.dataset_file)
    episode_count = dataset_file_handler.get_num_episodes()
    if episode_count == 0:
        print("No episodes found in the dataset.")
        return

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))
    num_envs = args_cli.num_envs

    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # Configure action mode for non-state replay.
    if args_cli.replay_mode == "joint":
        from isaaclab_arena.embodiments.ex001arm.ex001arm import EX001ArmJointActionsCfg

        env_cfg.actions = EX001ArmJointActionsCfg()
        print("[INFO] replay_mode=joint: using EX001ArmJointActionsCfg")
    elif args_cli.replay_mode == "ee":
        for action_name in ("arm_action", "right_arm_action"):
            term = getattr(env_cfg.actions, action_name, None)
            if term is not None and hasattr(term, "controller"):
                term.controller.use_relative_mode = False
                term.scale = 1.0
        print("[INFO] replay_mode=ee: absolute EE replay")

    env_cfg.recorders = {}
    env_cfg.terminations = {}
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    expected_action_dim = (
        env.single_action_space.shape[0] if hasattr(env, "single_action_space") else env.action_space.shape[-1]
    )

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume replay.')

    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1 and args_cli.replay_mode == "action":
        state_validation_enabled = True
    elif args_cli.validate_states:
        print("Warning: state validation requires --num_envs=1 and replay_mode=action. Skipping.")

    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape, device=env.device)

    env.reset()
    teleop_interface.reset()

    episode_names = list(dataset_file_handler.get_episode_names())
    raw_h5 = h5py.File(args_cli.dataset_file, "r")
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            env_step_idx = {index: 0 for index in range(num_envs)}
            env_loaded_episode = {index: None for index in range(num_envs)}
            first_loop = True
            has_next = True
            while has_next:
                actions = idle_action.clone()
                has_next = False
                for env_id in range(num_envs):
                    episode_data = env_episode_data_map[env_id]
                    env_next_action = None
                    next_state = None

                    if args_cli.replay_mode == "action":
                        env_next_action = episode_data.get_next_action()
                    elif args_cli.replay_mode == "state":
                        next_state = episode_data.get_next_state()
                    else:
                        # joint/ee from raw obs arrays
                        if episode_data.is_empty():
                            env_next_action = None
                        else:
                            step = env_step_idx[env_id]
                            loaded_idx = env_loaded_episode[env_id]
                            if loaded_idx is None:
                                env_next_action = None
                                step = 0
                            else:
                                episode_name = episode_names[int(loaded_idx)]
                                obs_group = raw_h5["data"][episode_name]["obs"]
                                if args_cli.replay_mode == "joint":
                                    if step < obs_group["joint_pos"].shape[0]:
                                        joint_pos = torch.tensor(
                                            obs_group["joint_pos"][step], device=env.device, dtype=torch.float32
                                        )
                                        env_next_action = _joint_obs_to_action(joint_pos)
                                else:
                                    if "eef_state" in obs_group and step < obs_group["eef_state"].shape[0]:
                                        env_next_action = torch.tensor(
                                            obs_group["eef_state"][step], device=env.device, dtype=torch.float32
                                        )
                                    else:
                                        required = (
                                            "eef_pos",
                                            "eef_quat",
                                            "gripper_pos",
                                            "right_eef_pos",
                                            "right_eef_quat",
                                            "right_gripper_pos",
                                        )
                                        if all(k in obs_group for k in required) and step < obs_group["eef_pos"].shape[0]:
                                            left_pos = torch.tensor(
                                                obs_group["eef_pos"][step], device=env.device, dtype=torch.float32
                                            )
                                            left_quat = torch.tensor(
                                                obs_group["eef_quat"][step], device=env.device, dtype=torch.float32
                                            )
                                            left_gripper = torch.tensor(
                                                obs_group["gripper_pos"][step], device=env.device, dtype=torch.float32
                                            )
                                            right_pos = torch.tensor(
                                                obs_group["right_eef_pos"][step], device=env.device, dtype=torch.float32
                                            )
                                            right_quat = torch.tensor(
                                                obs_group["right_eef_quat"][step], device=env.device, dtype=torch.float32
                                            )
                                            right_gripper = torch.tensor(
                                                obs_group["right_gripper_pos"][step], device=env.device, dtype=torch.float32
                                            )
                                            env_next_action = torch.cat(
                                                [
                                                    left_pos,
                                                    left_quat,
                                                    left_gripper,
                                                    right_pos,
                                                    right_quat,
                                                    right_gripper,
                                                ],
                                                dim=0,
                                            )
                                if env_next_action is not None:
                                    env_step_idx[env_id] += 1

                    needs_episode = False
                    if args_cli.replay_mode in ("action", "joint", "ee") and env_next_action is None:
                        needs_episode = True
                    if args_cli.replay_mode == "state" and next_state is None:
                        needs_episode = True

                    if needs_episode:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is None:
                            continue

                        replayed_episode_count += 1
                        print(f"{replayed_episode_count:4}: Loading #{next_episode_index} episode to env_{env_id}")
                        episode_data = dataset_file_handler.load_episode(episode_names[next_episode_index], env.device)
                        env_episode_data_map[env_id] = episode_data
                        env_loaded_episode[env_id] = next_episode_index
                        initial_state = episode_data.get_initial_state()
                        env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)

                        if args_cli.replay_mode == "action":
                            env_next_action = episode_data.get_next_action()
                        elif args_cli.replay_mode == "state":
                            next_state = episode_data.get_next_state()
                        else:
                            # joint/ee: step cursor starts from 0
                            env_step_idx[env_id] = 0
                            obs_group = raw_h5["data"][episode_names[next_episode_index]]["obs"]
                            if args_cli.replay_mode == "joint":
                                joint_pos = torch.tensor(obs_group["joint_pos"][0], device=env.device, dtype=torch.float32)
                                env_next_action = _joint_obs_to_action(joint_pos)
                            else:
                                if "eef_state" in obs_group:
                                    env_next_action = torch.tensor(obs_group["eef_state"][0], device=env.device, dtype=torch.float32)
                                else:
                                    left_pos = torch.tensor(obs_group["eef_pos"][0], device=env.device, dtype=torch.float32)
                                    left_quat = torch.tensor(obs_group["eef_quat"][0], device=env.device, dtype=torch.float32)
                                    left_gripper = torch.tensor(
                                        obs_group["gripper_pos"][0], device=env.device, dtype=torch.float32
                                    )
                                    right_pos = torch.tensor(
                                        obs_group["right_eef_pos"][0], device=env.device, dtype=torch.float32
                                    )
                                    right_quat = torch.tensor(
                                        obs_group["right_eef_quat"][0], device=env.device, dtype=torch.float32
                                    )
                                    right_gripper = torch.tensor(
                                        obs_group["right_gripper_pos"][0], device=env.device, dtype=torch.float32
                                    )
                                    env_next_action = torch.cat(
                                        [left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper], dim=0
                                    )
                                env_step_idx[env_id] = 1
                        has_next = True
                    else:
                        has_next = True

                    if args_cli.replay_mode == "state":
                        if next_state is not None:
                            env.scene.reset_to(
                                next_state,
                                env_ids=torch.tensor([env_id], device=env.device),
                                is_relative=True,
                            )
                    elif env_next_action is not None:
                        actions[env_id] = _map_action_dim(env_next_action, expected_action_dim)

                if not has_next:
                    break
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue

                if args_cli.replay_mode == "state":
                    env.scene.write_data_to_sim()
                    env.sim.step(render=True)
                    env.scene.update(dt=env.physics_dt)
                else:
                    env.step(actions)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1:4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break

    raw_h5.close()
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
