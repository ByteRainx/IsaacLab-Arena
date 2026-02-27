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
    help="File path or directory to export recorded demos. Defaults to ./data",
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
parser.add_argument(
    "--joint_signs", type=str, default="1,1,-1,-1,-1,1",
    help="Per-joint sign multipliers (6 comma-separated values). "
         "Use -1 to flip a joint axis. "
         "Default: '1,1,-1,-1,-1,1' (joints 3-5 inverted for ARX X5, joint6 same).",
)
parser.add_argument(
    "--joint_offsets", type=str, default="0,0,0,0,0,0",
    help="Per-joint offsets in radians (6 comma-separated values). "
         "Applied after sign: sim = sign * phys + offset. "
         "Default: '0,0,0,0,0,0'",
)

add_example_environments_cli_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-simulation imports ───────────────────────────────────────────────────

import gymnasium as gym
import numpy as np
import torch

import omni.log
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul


# ── Camera observation recorder (same as replay_and_record_demos.py) ─────────

class _PreStepCameraRecorder(RecorderTerm):
    """Record camera observations from the policy obs buffer each step."""

    def record_pre_step(self):
        camera_data = {}
        if "camera_obs" in self._env.obs_buf:
            return "camera_obs", self._env.obs_buf["camera_obs"]
        if "policy" in self._env.obs_buf:
            policy_obs = self._env.obs_buf["policy"]
            if isinstance(policy_obs, dict):
                for key in ["left_wrist_cam", "right_wrist_cam", "head_cam"]:
                    if key in policy_obs:
                        camera_data[key] = policy_obs[key]
        if camera_data:
            return "camera_obs", camera_data
        return None


@configclass
class _PreStepCameraRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = _PreStepCameraRecorder


@configclass
class _ActionStateCameraRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder manager that records actions, states AND camera images."""
    record_pre_step_camera_observations = _PreStepCameraRecorderCfg()


# ── Joint diagnostic helpers ──────────────────────────────────────────────────

def _parse_float_list(s: str, expected_len: int, name: str) -> tuple[float, ...]:
    """Parse a comma-separated string of floats."""
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != expected_len:
        raise ValueError(
            f"--{name} expects {expected_len} values, got {len(parts)}: '{s}'"
        )
    return tuple(float(x) for x in parts)


def _dump_joint_info(env) -> None:
    """Print detailed joint information from the simulation articulation.

    Helps diagnose axis/limit mismatches between the physical arm and the
    simulation model.
    """
    robot = env.scene["robot"]
    joint_names = robot.joint_names

    # Joint positions and limits
    joint_pos = robot.data.joint_pos[0].cpu().numpy()
    joint_limits = robot.data.joint_limits[0].cpu().numpy() if hasattr(robot.data, "joint_limits") else None

    # Default joint positions from the articulation config
    default_pos = robot.data.default_joint_pos[0].cpu().numpy() if hasattr(robot.data, "default_joint_pos") else None

    print("\n" + "=" * 80)
    print("  SIMULATION JOINT DIAGNOSTIC (from USD articulation)")
    print("=" * 80)
    print(f"  Total joints: {len(joint_names)}")
    print(f"  {'Idx':<4} {'Joint Name':<35} {'Curr Pos':>10} {'Default':>10} {'Lower':>10} {'Upper':>10}")
    print("  " + "-" * 83)
    for i, name in enumerate(joint_names):
        curr = f"{joint_pos[i]:+.4f}" if i < len(joint_pos) else "N/A"
        defp = f"{default_pos[i]:+.4f}" if default_pos is not None and i < len(default_pos) else "N/A"
        lo = f"{joint_limits[i, 0]:+.4f}" if joint_limits is not None and i < len(joint_limits) else "N/A"
        hi = f"{joint_limits[i, 1]:+.4f}" if joint_limits is not None and i < len(joint_limits) else "N/A"
        # Highlight arm joints
        marker = " <<" if "arm_joint" in name else ""
        print(f"  {i:<4} {name:<35} {curr:>10} {defp:>10} {lo:>10} {hi:>10}{marker}")

    # Also show action term → joint mapping for JointPositionActionCfg
    if hasattr(env, "action_manager"):
        print("\n  ACTION TERM -> JOINT MAPPING:")
        for term_name, term in env.action_manager._terms.items():
            if hasattr(term, "_joint_ids"):
                jids = term._joint_ids
                jnames = [joint_names[j] for j in jids] if isinstance(jids, (list, tuple)) else "N/A"
                print(f"    {term_name}: joint_ids={list(jids)} -> {jnames}")
            elif hasattr(term, "joint_ids"):
                jids = term.joint_ids
                jnames = [joint_names[j] for j in jids] if isinstance(jids, (list, tuple)) else "N/A"
                print(f"    {term_name}: joint_ids={list(jids)} -> {jnames}")

    print("=" * 80 + "\n")


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

    # Parse joint mapping parameters
    joint_signs = _parse_float_list(args_cli.joint_signs, 6, "joint_signs")
    joint_offsets = _parse_float_list(args_cli.joint_offsets, 6, "joint_offsets")

    cfg = Ex001ArmWsRemoteCfg(
        remote_ip=args_cli.remote_ip,
        remote_port=args_cli.remote_port,
        sim_device=str(env.device),
        control_mode=args_cli.control_mode,
        pos_scale=args_cli.pos_scale,
        rot_scale=args_cli.rot_scale,
        debug=args_cli.debug,
        joint_signs=joint_signs,
        joint_offsets=joint_offsets,
    )
    teleop = Ex001ArmWsRemoteTeleop(cfg)
    mode_label = args_cli.control_mode.upper()
    print(f"[INFO] Using remote WebSocket teleop ({mode_label} mode): "
          f"ws://{args_cli.remote_ip}:{args_cli.remote_port}")
    if args_cli.control_mode == "joint":
        print(f"[INFO] Joint signs  : {joint_signs}")
        print(f"[INFO] Joint offsets: {joint_offsets}")
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

    # Output directory: ./data/<task_name>/ by default
    task_name = getattr(args_cli, "example_environment", "default")
    dataset_path = args_cli.dataset_file or os.path.join("./data", task_name)
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

    # Recorder — with or without camera images
    enable_cameras = getattr(args_cli, "enable_cameras", False)
    if enable_cameras:
        env_cfg.recorders = _ActionStateCameraRecorderManagerCfg()
        print("[INFO] Recording actions, states AND camera images")
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        print("[INFO] Recording actions and states only (no images)")
        for cam_key in ["left_wrist_cam", "right_wrist_cam", "head_cam", "robot_pov_cam_rgb"]:
            if hasattr(env_cfg.observations.policy, cam_key):
                delattr(env_cfg.observations.policy, cam_key)
                print(f"[INFO] Removed {cam_key} from observations")

    base_frame_keys = ["eef_pos_base", "eef_quat_base", "right_eef_pos_base",
                       "right_eef_quat_base", "gripper_pos_normalized", "right_gripper_pos_normalized"]
    present = [k for k in base_frame_keys if hasattr(env_cfg.observations.policy, k)]
    if present:
        print(f"[INFO] Base-frame observations included in recording: {present}")

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = current_output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    # ▸ CRITICAL: disable automatic export on every env.reset() —
    #   otherwise empty episodes are written to the HDF5 each time
    #   the environment resets (including the initial reset and R-key resets).
    env_cfg.recorders.export_in_record_pre_reset = False

    # Create environment
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)

    # The RecorderManager.__init__ eagerly creates an HDF5 file handler,
    # but we don't want to leave an empty file if the user resets before
    # the first successful demo.  Close and remove the empty file; we will
    # create file handlers on-demand when exporting real data.
    if (
        hasattr(env.recorder_manager, "_dataset_file_handler")
        and env.recorder_manager._dataset_file_handler is not None
    ):
        env.recorder_manager._dataset_file_handler.close()
        env.recorder_manager._dataset_file_handler = None
    _startup_empty = os.path.join(output_dir, f"{current_output_file_name}.hdf5")
    if os.path.exists(_startup_empty) and os.path.getsize(_startup_empty) < 8192:
        os.remove(_startup_empty)
        print(f"[INFO] Removed startup placeholder: {os.path.basename(_startup_empty)}")

    expected_action_dim = _get_expected_action_dim(env) or env.action_space.shape[-1]

    # Dump joint diagnostic information (always in joint mode or when --debug)
    if args_cli.control_mode == "joint" or args_cli.debug:
        _dump_joint_info(env)

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

    def _clear_task_state():
        """Clear task-specific persistent state (e.g. contact history) after env reset."""
        for attr in ("_buttons_contact_history_state",):
            if hasattr(env, attr):
                getattr(env, attr).zero_()

    def export_and_prepare_next():
        """Export the current episode to a NEW HDF5 file and prepare for the next recording.

        File handler is created on-demand here (not in advance), so reset-only
        operations never leave behind empty HDF5 files.
        """
        nonlocal recorded_demos, current_output_file_name, current_output_path
        nonlocal running_recording

        # 1. Determine the output filename for this demo
        current_output_file_name = _next_available_filename()
        current_output_path = os.path.join(output_dir, f"{current_output_file_name}.hdf5")

        # 2. Create a fresh file handler and export
        env.recorder_manager.cfg.dataset_filename = current_output_file_name
        fh = env.recorder_manager.cfg.dataset_file_handler_class_type()
        fh.create(
            os.path.join(output_dir, current_output_file_name),
            env_name=getattr(env.cfg, "env_name", None),
        )
        env.recorder_manager._dataset_file_handler = fh

        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.export_episodes([0])
        recorded_demos += 1
        print(f"[{recorded_demos}] Demo exported to: {current_output_path}")

        # 3. Close the file handler — no pre-creation for the next file
        fh.close()
        env.recorder_manager._dataset_file_handler = None

        # 4. Reset environment for the next recording
        env.recorder_manager.reset([0])
        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()
        _clear_task_state()

        running_recording = True
        print("=" * 60)
        print("[INFO] Episode saved.  Ready for next trajectory.")
        print("=" * 60)

    def reset_only():
        """Discard current recording and reset the environment.

        No data is exported; the recorder buffer is cleared BEFORE env.reset()
        so that record_pre_reset (even if it somehow fires) has nothing to write.
        """
        env.recorder_manager.reset([0])       # clear buffer first
        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        _store_home_ee_poses()
        _clear_task_state()
        auto_reset.clear()
        print("[INFO] Environment reset — recording discarded, no file written.")

    def request_reset():
        nonlocal should_reset
        should_reset = True

    teleop_interface.add_callback("R", request_reset)

    # Initial setup
    env.sim.reset()
    env.reset()
    teleop_interface.reset()
    _store_home_ee_poses()
    _clear_task_state()

    # Auto-create 4-viewport layout: Perspective + Left Wrist + Head + Right Wrist
    # Viewport 1 (default, Perspective) and Viewport 2 are created by Isaac Sim.
    # We create Viewport 3 and 4 for the remaining cameras.
    try:
        from omni.kit.viewport.window import ViewportWindow
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        # Ordered: Viewport 1=Perspective(default), 2=Left Wrist, 3=Head, 4=Right Wrist
        cam_viewport_map = [
            (2, "Left_Gripper_Camera", "Left Wrist"),
            (3, "Head_Camera", "Head"),
            (4, "Right_Gripper_Camera", "Right Wrist"),
        ]
        cam_paths = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            for _, cam_name, _ in cam_viewport_map:
                if name == cam_name:
                    cam_paths[cam_name] = str(prim.GetPath())

        for vp_num, cam_name, label in cam_viewport_map:
            path = cam_paths.get(cam_name)
            if path is None:
                continue
            if vp_num >= 3:
                vp = ViewportWindow(f"Viewport {vp_num}", width=640, height=480)
                vp.viewport_api.set_active_camera(path)
            print(f"[INFO] Viewport {vp_num} -> {label} ({path})")
    except Exception as e:
        print(f"[INFO] Auto-viewport setup skipped: {e}")

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
