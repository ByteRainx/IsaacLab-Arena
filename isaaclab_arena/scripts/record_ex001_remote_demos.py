# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Record demonstrations for EX001Arm using remote physical arms via WebSocket.

The remote machine runs ``tools/ros1_ws_bridge.py`` to forward ROS1 arm data
over WebSocket. This script receives the data and drives the simulation.

Supports two control modes:

* ``ee``    -- End-effector delta (DifferentialIK, default)
* ``joint`` -- Direct joint position (fastest, no IK)

Usage::

    # EE mode (default)
    python -m isaaclab_arena.scripts.record_ex001_remote_demos \\
        --remote_ip 10.100.21.249 ex001arm_cvpr_scene_put_blocks_to_color

    # Joint mode (recommended for same-model physical arms)
    python -m isaaclab_arena.scripts.record_ex001_remote_demos \\
        --control_mode joint --remote_ip 10.100.21.249 \\
        ex001arm_cvpr_scene_put_blocks_to_color
"""

# ── Pre-simulation imports & AppLauncher ──────────────────────────────────────

import contextlib
import os
import time
from dataclasses import dataclass
from typing import Any

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()

# ── Standard recording args ──────────────────────────────────────────────────
parser.add_argument(
    "--dataset_file", type=str, default="",
    help="File path or directory to export recorded demos. Defaults to ./demos",
)
parser.add_argument(
    "--reset_duration", type=float, default=10.0,
    help="Duration of reset trajectory in seconds. Default: 10.0",
)

# ── Remote (WebSocket) args ──────────────────────────────────────────────────
parser.add_argument(
    "--remote_ip", type=str, default="192.168.1.100",
    help="IP address of the robot PC running ros1_ws_bridge.py",
)
parser.add_argument(
    "--remote_port", type=int, default=5555,
    help="WebSocket port on the robot PC. Default: 5555",
)
parser.add_argument(
    "--pos_scale", type=float, default=1.0,
    help="Scale factor for position deltas. Increase if sim moves too little. Default: 1.0",
)
parser.add_argument(
    "--rot_scale", type=float, default=1.0,
    help="Scale factor for rotation deltas. Default: 1.0",
)
parser.add_argument(
    "--debug", action="store_true",
    help="Print gripper and delta values periodically for debugging.",
)
parser.add_argument(
    "--control_mode", type=str, default="ee", choices=["ee", "joint"],
    help="Control mode: 'ee' (end-effector delta, default) or 'joint' "
         "(direct joint positions, no IK, fastest).",
)

add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-simulation imports ───────────────────────────────────────────────────

import gymnasium as gym
import torch

import omni.log
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul


# ── Utilities ─────────────────────────────────────────────────────────────────

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


def _get_ee_poses(env: gym.Env):
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
    action[6] = -1.0

    right_pos_delta = (auto_reset.home_right_ee_pos - right_pos) * pos_gain * blend_gain
    right_quat_error = quat_mul(auto_reset.home_right_ee_quat, quat_conjugate(right_quat))
    right_rot_delta = axis_angle_from_quat(right_quat_error) * rot_gain * blend_gain
    action[7:10] = right_pos_delta
    action[10:13] = right_rot_delta
    if expected_action_dim > 13:
        action[13] = -1.0

    return action


# ── Teleop device creation ────────────────────────────────────────────────────

def create_teleop_interface(env):
    from isaaclab_arena.teleop_devices.ex001arm_ws_remote import (
        Ex001ArmWsRemoteCfg,
        Ex001ArmWsRemoteTeleop,
    )

    cfg = Ex001ArmWsRemoteCfg(
        remote_ip=args_cli.remote_ip,
        remote_port=args_cli.remote_port,
        sim_device=str(env.device),
        control_mode=args_cli.control_mode,
        pos_scale=args_cli.pos_scale,
        rot_scale=args_cli.rot_scale,
        debug=args_cli.debug,
    )
    teleop = Ex001ArmWsRemoteTeleop(cfg)
    mode_label = args_cli.control_mode.upper()
    print(f"[INFO] Using remote WebSocket teleop ({mode_label} mode): "
          f"ws://{args_cli.remote_ip}:{args_cli.remote_port}")
    return teleop


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # --- Select action config based on control mode -----------------
    if args_cli.control_mode == "joint":
        from isaaclab_arena.embodiments.ex001arm.ex001arm import EX001ArmJointActionsCfg
        env_cfg.actions = EX001ArmJointActionsCfg()
        print("[INFO] Using EX001ArmJointActionsCfg (direct joint position, no IK)")
    else:
        from isaaclab_arena.embodiments.ex001arm.ex001arm import EX001ArmPhysicalTeleopActionsCfg
        env_cfg.actions = EX001ArmPhysicalTeleopActionsCfg()
        print("[INFO] Using EX001ArmPhysicalTeleopActionsCfg (IK scale=1.0, 1:1 mapping)")

    # Success / termination handling
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # Output directory
    dataset_path = args_cli.dataset_file or "./demos"
    dataset_ext = os.path.splitext(dataset_path)[1]
    is_dir_path = dataset_path.endswith(os.sep) or dataset_ext == ""

    output_dir = (dataset_path.rstrip(os.sep) or ".") if is_dir_path else (os.path.dirname(dataset_path) or ".")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    device_tag = f"arx_remote_{args_cli.control_mode}"

    def _next_available_filename() -> str:
        base_prefix = f"{device_tag}_episode"
        index = 0
        while True:
            candidate = f"{base_prefix}{index}"
            if not os.path.exists(os.path.join(output_dir, f"{candidate}.hdf5")):
                return candidate
            index += 1

    current_output_file_name = _next_available_filename()
    current_output_path = os.path.join(output_dir, f"{current_output_file_name}.hdf5")

    # Recorder
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    print("[INFO] Recording actions and states only (no images)")

    for cam_key in ["left_wrist_cam", "right_wrist_cam", "head_cam", "robot_pov_cam_rgb"]:
        if hasattr(env_cfg.observations.policy, cam_key):
            delattr(env_cfg.observations.policy, cam_key)
            print(f"[INFO] Removed {cam_key} from observations")

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = current_output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    # Create environment
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)

    expected_action_dim = _get_expected_action_dim(env) or env.action_space.shape[-1]
    teleop_interface = create_teleop_interface(env)

    # State
    should_reset = False
    recorded_demos = 0
    auto_reset = AutoResetState()
    reset_duration = args_cli.reset_duration
    running_recording = True  # start recording immediately

    def _store_home_ee_poses():
        lp, lq, rp, rq = _get_ee_poses(env)
        auto_reset.home_left_ee_pos = lp
        auto_reset.home_left_ee_quat = lq
        auto_reset.home_right_ee_pos = rp
        auto_reset.home_right_ee_quat = rq

    def export_and_prepare_next():
        nonlocal recorded_demos, current_output_file_name, current_output_path
        nonlocal running_recording

        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.export_episodes([0])
        recorded_demos += 1
        print(f"[{recorded_demos}] Demo exported to: {current_output_path}")

        if hasattr(env.recorder_manager, "_dataset_file_handler") and env.recorder_manager._dataset_file_handler is not None:
            env.recorder_manager._dataset_file_handler.close()

        current_output_file_name = _next_available_filename()
        current_output_path = os.path.join(output_dir, f"{current_output_file_name}.hdf5")

        env.recorder_manager.cfg.dataset_filename = current_output_file_name
        env.recorder_manager._dataset_file_handler = env.recorder_manager.cfg.dataset_file_handler_class_type()
        env.recorder_manager._dataset_file_handler.create(
            os.path.join(output_dir, current_output_file_name),
            env_name=getattr(env.cfg, "env_name", None),
        )
        env.recorder_manager.reset([0])

        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()

        running_recording = True
        print("=" * 60)
        print(f"[INFO] Ready for next trajectory: {current_output_file_name}.hdf5")
        print("=" * 60)

    def reset_only():
        env.sim.reset()
        env.recorder_manager.reset([0])
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()
        auto_reset.clear()
        print("[INFO] Environment reset (no export).")

    def request_reset():
        nonlocal should_reset
        should_reset = True

    teleop_interface.add_callback("R", request_reset)

    # Initial setup
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    _store_home_ee_poses()

    print(f"Using teleop device:\n{teleop_interface}")
    print("=" * 60)
    print(f"[INFO] Output directory : {output_dir}")
    print(f"[INFO] Control mode     : {args_cli.control_mode}")
    print(f"[INFO] Reset duration   : {reset_duration}s")
    print(f"[INFO] Remote           : ws://{args_cli.remote_ip}:{args_cli.remote_port}")
    print("[INFO] One trajectory per HDF5 file")
    print("[INFO] Press 'R' to reset (no export)")
    print("=" * 60)

    # Main loop
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            if auto_reset.active:
                device_action = _compute_reset_action(env, auto_reset, expected_action_dim, reset_duration)
            elif auto_reset.done_pending_export:
                device_action = torch.zeros(expected_action_dim, device=env.device)
            else:
                action = teleop_interface.advance()
                device_action = _map_action_dim(action, expected_action_dim)

            should_step = running_recording or auto_reset.active or auto_reset.done_pending_export
            if should_step:
                env.step(device_action.repeat(env.num_envs, 1))
            else:
                env.sim.render()

            # Success detection
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

            if auto_reset.success_pending_reset and auto_reset.success_wait_start is not None:
                if time.time() - auto_reset.success_wait_start >= 2.0:
                    auto_reset.active = True
                    auto_reset.start_time = time.time()
                    running_recording = True
                    print(f"[INFO] Reset trajectory started ({reset_duration}s)...")
                    auto_reset.success_pending_reset = False
                    auto_reset.success_wait_start = None

            if auto_reset.active and auto_reset.start_time is not None:
                if time.time() - auto_reset.start_time >= reset_duration:
                    auto_reset.active = False
                    auto_reset.done_pending_export = True
                    auto_reset.export_wait_steps = 2
                    auto_reset.start_time = None

            if auto_reset.done_pending_export and not auto_reset.active:
                if should_step and auto_reset.export_wait_steps > 0:
                    auto_reset.export_wait_steps -= 1
                if auto_reset.export_wait_steps == 0:
                    export_and_prepare_next()
                    auto_reset.done_pending_export = False
                    auto_reset.success_lock = False

            if should_reset:
                reset_only()
                should_reset = False

            if env.sim.is_stopped():
                break

            # Render each frame (no artificial rate limiting -- run at max sim speed)
            env.sim.render()

    env.close()
    print(f"Recording session completed with {recorded_demos} demonstrations")


if __name__ == "__main__":
    main()
    simulation_app.close()
