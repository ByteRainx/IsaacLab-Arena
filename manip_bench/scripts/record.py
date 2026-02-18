# Copyright (c) 2025, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from manip_bench.scripts.cli import (
    add_manip_bench_cli_args as add_example_environments_cli_args,
    get_env_builder_from_cli as get_arena_builder_from_cli,
)


def _teleop_device_requires_xr(device_name: str | None) -> bool:
    if not device_name:
        return False
    name = device_name.lower()
    return any(token in name for token in ("handtracking", "openxr")) or name in {
        "avp_handtracking",
        "ex001arm_openxr_bimanual",
    }


parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--dataset_file",
    type=str,
    default="",
    help="File path or directory to export recorded demos. Defaults to ./demos",
)
parser.add_argument(
    "--reset_duration",
    type=float,
    default=10.0,
    help="Duration of reset trajectory in seconds. Default: 10.0",
)
DEFAULT_STEP_HZ = 30

add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

device_name = getattr(args_cli, "teleop_device", None)
if _teleop_device_requires_xr(device_name):
    app_launcher_args["xr"] = True
    setattr(args_cli, "xr", True)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import manip_bench.extensions  # noqa: F401  — register all ManipBench assets

import gymnasium as gym
import torch

import omni.log
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat


class RateLimiter:
    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


@dataclass
class AutoResetState:
    success_wait_start: float | None = None
    success_pending_reset: bool = False
    active: bool = False
    start_time: float | None = None
    done_pending_export: bool = False
    export_wait_steps: int = 0
    success_lock: bool = False
    home_left_ee_pos: torch.Tensor | None = None
    home_left_ee_quat: torch.Tensor | None = None
    home_right_ee_pos: torch.Tensor | None = None
    home_right_ee_quat: torch.Tensor | None = None

    def clear(self):
        self.success_wait_start = None
        self.success_pending_reset = False
        self.active = False
        self.start_time = None
        self.done_pending_export = False
        self.export_wait_steps = 0
        self.success_lock = False


def _get_expected_action_dim(env) -> int | None:
    if hasattr(env, "single_action_space") and hasattr(env.single_action_space, "shape"):
        return env.single_action_space.shape[0]
    if hasattr(env, "action_space") and hasattr(env.action_space, "shape"):
        return env.action_space.shape[-1]
    return None


def _map_action_dim(action: torch.Tensor, expected_dim: int) -> torch.Tensor:
    if action.shape[-1] == expected_dim:
        return action
    if expected_dim < action.shape[-1]:
        return action[:expected_dim]
    mapped = torch.zeros((expected_dim,), device=action.device, dtype=action.dtype)
    mapped[: action.shape[-1]] = action
    return mapped


def _check_success(success_term: Any | None, env: gym.Env) -> bool:
    if success_term is None:
        return False
    return bool(success_term.func(env, **success_term.params)[0])


def _get_ee_poses(env: gym.Env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    left_ee = env.scene["ee_frame"]
    right_ee = env.scene["right_ee_frame"]
    left_pos = left_ee.data.target_pos_w[0, 0, :].clone()
    left_quat = left_ee.data.target_quat_w[0, 0, :].clone()
    right_pos = right_ee.data.target_pos_w[0, 0, :].clone()
    right_quat = right_ee.data.target_quat_w[0, 0, :].clone()
    return left_pos, left_quat, right_pos, right_quat


def _compute_reset_action(
    env: gym.Env,
    auto_reset: AutoResetState,
    expected_action_dim: int,
    reset_duration: float,
    pos_gain: float = 0.3,
    rot_gain: float = 0.3,
) -> torch.Tensor:
    """Compute action to move EE towards home position."""
    action = torch.zeros(expected_action_dim, device=env.device)

    if auto_reset.home_left_ee_pos is None:
        return action

    elapsed = time.time() - auto_reset.start_time if auto_reset.start_time else 0.0
    alpha = min(elapsed / reset_duration, 1.0)
    blend_gain = 1.0 - alpha * 0.5

    left_pos, left_quat, right_pos, right_quat = _get_ee_poses(env)

    left_pos_delta = (auto_reset.home_left_ee_pos - left_pos) * pos_gain * blend_gain
    left_quat_error = quat_mul(auto_reset.home_left_ee_quat, quat_conjugate(left_quat))
    left_rot_delta = axis_angle_from_quat(left_quat_error) * rot_gain * blend_gain
    action[0:3] = left_pos_delta
    action[3:6] = left_rot_delta
    action[6] = -1.0  # Close left gripper during reset

    right_pos_delta = (auto_reset.home_right_ee_pos - right_pos) * pos_gain * blend_gain
    right_quat_error = quat_mul(auto_reset.home_right_ee_quat, quat_conjugate(right_quat))
    right_rot_delta = axis_angle_from_quat(right_quat_error) * rot_gain * blend_gain
    action[7:10] = right_pos_delta
    action[10:13] = right_rot_delta

    if expected_action_dim > 13:
        action[13] = -1.0  # Close right gripper during reset

    return action


def create_teleop_interface(env, env_cfg):
    device_name = getattr(args_cli, "teleop_device", "keyboard") or "keyboard"

    if _teleop_device_requires_xr(device_name):
        teleop_callbacks = {}
        if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                device_name,
                env_cfg.teleop_devices.devices,
                teleop_callbacks
            )
            print(f"[INFO] Using VR teleop device: {device_name}")
        else:
            omni.log.error(f"VR device '{device_name}' not found in environment config")
            exit(1)
        return teleop_interface
    else:
        from manip_bench.scripts.teleop_keyboard import (
            BimanualSe3Keyboard,
            BimanualSe3KeyboardCfg,
        )
        cfg = BimanualSe3KeyboardCfg(sim_device=env.device)
        teleop_interface = BimanualSe3Keyboard(cfg)
        print("[INFO] Using keyboard teleop device")
        return teleop_interface


def main() -> None:
    # Parse environment configuration
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Setup output directory
    dataset_path = args_cli.dataset_file or "./demos"
    dataset_ext = os.path.splitext(dataset_path)[1]
    is_dir_path = dataset_path.endswith(os.sep) or dataset_ext == ""

    if is_dir_path:
        output_dir = dataset_path.rstrip(os.sep) or "."
    else:
        output_dir = os.path.dirname(dataset_path) or "."

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    device_name = getattr(args_cli, "teleop_device", None)
    is_vr = _teleop_device_requires_xr(device_name)

    def _device_tag() -> str:
        return "vr" if is_vr else "keyboard"

    def _next_available_filename() -> str:
        prefix = _device_tag()
        base_prefix = f"{prefix}_episode"
        index = 0
        while True:
            candidate = f"{base_prefix}{index}"
            candidate_path = os.path.join(output_dir, f"{candidate}.hdf5")
            if not os.path.exists(candidate_path):
                return candidate
            index += 1

    # Get first filename
    current_output_file_name = _next_available_filename()
    current_output_path = os.path.join(output_dir, f"{current_output_file_name}.hdf5")

    # Configure recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    print("[INFO] Recording actions and states only (no images)")

    camera_obs_keys = ["left_wrist_cam", "right_wrist_cam", "head_cam", "robot_pov_cam_rgb"]
    for cam_key in camera_obs_keys:
        if hasattr(env_cfg.observations.policy, cam_key):
            delattr(env_cfg.observations.policy, cam_key)
            print(f"[INFO] Removed {cam_key} from observations")

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = current_output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)

    expected_action_dim = _get_expected_action_dim(env) or env.action_space.shape[-1]
    teleop_interface = create_teleop_interface(env, env_cfg)

    should_reset = False
    recorded_demos = 0
    auto_reset = AutoResetState()
    reset_duration = args_cli.reset_duration

    def _store_home_ee_poses() -> None:
        left_pos, left_quat, right_pos, right_quat = _get_ee_poses(env)
        auto_reset.home_left_ee_pos = left_pos
        auto_reset.home_left_ee_quat = left_quat
        auto_reset.home_right_ee_pos = right_pos
        auto_reset.home_right_ee_quat = right_quat

    # VR: wait for START; Keyboard: start immediately
    running_recording = not is_vr

    def export_and_prepare_next() -> None:
        nonlocal recorded_demos, current_output_file_name, current_output_path
        nonlocal running_recording

        # Export current episode
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.export_episodes([0])

        recorded_demos += 1
        print(f"[{recorded_demos}] Demo exported to: {current_output_path}")

        # Close existing dataset file handler to allow new file creation
        if hasattr(env.recorder_manager, '_dataset_file_handler') and env.recorder_manager._dataset_file_handler is not None:
            env.recorder_manager._dataset_file_handler.close()

        # Prepare new filename for next trajectory
        current_output_file_name = _next_available_filename()
        current_output_path = os.path.join(output_dir, f"{current_output_file_name}.hdf5")

        # Update recorder config and create new file handler
        env.recorder_manager.cfg.dataset_filename = current_output_file_name
        env.recorder_manager._dataset_file_handler = env.recorder_manager.cfg.dataset_file_handler_class_type()
        env.recorder_manager._dataset_file_handler.create(
            os.path.join(output_dir, current_output_file_name),
            env_name=getattr(env.cfg, "env_name", None)
        )
        env.recorder_manager.reset([0])

        # Reset environment
        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()

        # Stop recording, wait for user to click START
        running_recording = False
        print("=" * 60)
        print(f"[INFO] Ready for next trajectory: {current_output_file_name}.hdf5")
        print("[INFO] Click START in VR to begin recording next trajectory")
        print("=" * 60)

    def reset_only() -> None:
        env.sim.reset()
        env.recorder_manager.reset([0])
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()
        auto_reset.clear()
        print("[INFO] Environment reset (no export).")

    def request_reset() -> None:
        nonlocal should_reset
        should_reset = True

    def start_recording() -> None:
        nonlocal running_recording
        running_recording = True
        print(f"[INFO] Recording started → {current_output_file_name}.hdf5")

    def stop_recording() -> None:
        nonlocal running_recording
        running_recording = False
        print("[INFO] Recording paused")

    teleop_interface.add_callback("R", request_reset)
    teleop_interface.add_callback("RESET", request_reset)
    teleop_interface.add_callback("START", start_recording)
    teleop_interface.add_callback("STOP", stop_recording)

    rate_limiter = None if is_vr else RateLimiter(DEFAULT_STEP_HZ)

    # Initial setup
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    _store_home_ee_poses()

    if not is_vr:
        print(f"Using teleop device:\n{teleop_interface}")
    print("=" * 60)
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Reset duration: {reset_duration}s")
    print("[INFO] One trajectory per HDF5 file")
    if is_vr:
        print("[INFO] VR Mode: Click START to begin recording")
        print("[INFO]          Click STOP to pause, RESET to discard")
    print("[INFO] Press 'R' to reset (no export)")
    print("=" * 60)

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Compute action
            if auto_reset.active:
                device_action = _compute_reset_action(
                    env, auto_reset, expected_action_dim, reset_duration
                )
            elif auto_reset.done_pending_export:
                device_action = torch.zeros(expected_action_dim, device=env.device)
            else:
                action = teleop_interface.advance()
                device_action = _map_action_dim(action, expected_action_dim)

            # Step environment
            should_step = running_recording or auto_reset.active or auto_reset.done_pending_export
            if should_step:
                env.step(device_action.repeat(env.num_envs, 1))
            else:
                env.sim.render()

            # Detect success
            if (
                running_recording
                and not auto_reset.success_pending_reset
                and not auto_reset.active
                and not auto_reset.done_pending_export
                and not auto_reset.success_lock
                and _check_success(success_term, env)
            ):
                auto_reset.success_pending_reset = True
                auto_reset.success_wait_start = time.time()
                auto_reset.success_lock = True
                print(f"[INFO] Success! Waiting 2s then {reset_duration}s reset trajectory...")

            # Start auto reset after wait
            if auto_reset.success_pending_reset and auto_reset.success_wait_start is not None:
                if time.time() - auto_reset.success_wait_start >= 2.0:
                    auto_reset.active = True
                    auto_reset.start_time = time.time()
                    running_recording = True
                    print(f"[INFO] Reset trajectory started ({reset_duration}s)...")
                    auto_reset.success_pending_reset = False
                    auto_reset.success_wait_start = None

            # Check reset completion
            if auto_reset.active and auto_reset.start_time is not None:
                if time.time() - auto_reset.start_time >= reset_duration:
                    auto_reset.active = False
                    auto_reset.done_pending_export = True
                    auto_reset.export_wait_steps = 2
                    auto_reset.start_time = None

            # Export after wait steps
            if auto_reset.done_pending_export and not auto_reset.active:
                if should_step and auto_reset.export_wait_steps > 0:
                    auto_reset.export_wait_steps -= 1
                if auto_reset.export_wait_steps == 0:
                    export_and_prepare_next()
                    auto_reset.done_pending_export = False
                    auto_reset.success_lock = False

            # Handle manual reset
            if should_reset:
                reset_only()
                should_reset = False

            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    env.close()
    print(f"Recording session completed with {recorded_demos} demonstrations")


if __name__ == "__main__":
    main()
    simulation_app.close()
