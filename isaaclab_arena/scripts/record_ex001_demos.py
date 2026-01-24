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
"""
Script to record demonstrations with EX001Arm using bimanual keyboard or VR control.

This script allows users to record demonstrations with EX001Arm embodiment using
either bimanual keyboard teleoperation or VR hand tracking. The recorded demonstrations
are stored as episodes in an hdf5 file. Cameras in the USD scene are always loaded,
but image recording to HDF5 is optional.

required arguments:
    --dataset_file            File path to export recorded demos.

optional arguments:
    -h, --help                Show this help message and exit
    --step_hz                 Environment stepping rate in Hz. (default: 30, only for keyboard)
    --num_demos               Number of demonstrations to record. (default: 1, set to 0 for infinite)
    --sensitivity             Keyboard sensitivity multiplier. (default: 1.0, only for keyboard)
    --record_images           Record camera images to HDF5. (default: False, saves disk space)

Note: The --teleop_device argument is defined by the environment subparser.
      Use 'keyboard' for bimanual keyboard control or 'ex001arm_openxr_bimanual' for VR.
"""

"""Launch Isaac Sim Simulator first."""

# Standard library imports
import contextlib
from collections.abc import Callable
from dataclasses import dataclass

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)


def _teleop_device_requires_xr(device_name: str | None) -> bool:
    """Check if teleop device requires XR to be enabled."""
    if not device_name:
        return False
    name = device_name.lower()
    return any(token in name for token in ("handtracking", "openxr")) or name in {
        "avp_handtracking",
        "ex001arm_openxr_bimanual",
    }


# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--dataset_file", type=str, required=True, help="File path to export recorded demos.")
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=1, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--sensitivity",
    type=float,
    default=1.0,
    help="Keyboard sensitivity multiplier (only for keyboard control).",
)
parser.add_argument(
    "--record_images",
    action="store_true",
    default=False,
    help="Record camera images to HDF5 (increases file size significantly).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)

# Add the example environments CLI args
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# Auto-enable XR for VR devices
device_name = getattr(args_cli, "teleop_device", None)
if _teleop_device_requires_xr(device_name):
    app_launcher_args["xr"] = True
    setattr(args_cli, "xr", True)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# Third-party imports
import gymnasium as gym
import numpy as np
import os
import time
import torch
from scipy.spatial.transform import Rotation

import carb
import omni
import omni.log
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


# =======================================================================================
# Camera Recording Support
# =======================================================================================


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step.
    
    Supports both:
    - Separate camera_obs group
    - Camera obs in policy group (like EX001Arm with left_wrist_cam, right_wrist_cam)
    """

    def record_pre_step(self):
        camera_data = {}
        
        # Check if camera_obs exists as separate group
        if "camera_obs" in self._env.obs_buf:
            return "camera_obs", self._env.obs_buf["camera_obs"]
        
        # For EX001Arm and similar: camera obs are in policy group with concatenate_terms=False
        if "policy" in self._env.obs_buf:
            policy_obs = self._env.obs_buf["policy"]
            # Check if it's a dict (concatenate_terms=False)
            if isinstance(policy_obs, dict):
                for key in ["left_wrist_cam", "right_wrist_cam", "head_cam", "robot_pov_cam_rgb"]:
                    if key in policy_obs:
                        camera_data[key] = policy_obs[key]
        
        if camera_data:
            return "camera_obs", camera_data
        return None


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Recorder manager with camera observation recording."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()


# =======================================================================================
# Bimanual Keyboard Controller (copied from teleop_bimanual_keyboard.py to avoid import)
# =======================================================================================


@dataclass
class BimanualSe3KeyboardCfg:
    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.05
    sim_device: str | None = None


class BimanualSe3Keyboard:
    """Keyboard controller that outputs two independent SE(3)+gripper commands (14-dim).

    Left arm:
      - Toggle gripper: K
      - Move: W/S (x), A/D (y), Q/E (z)
      - Rotate: Z/X (rx), T/G (ry), C/V (rz)

    Right arm:
      - Toggle gripper: P
      - Move: I/F (x), J/L (y), U/O (z)
      - Rotate: N/M (rx), Y/B (ry), 1/2 (rz)

    Output layout:
      [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
       right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]
    """

    def __init__(self, cfg: BimanualSe3KeyboardCfg):
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self._sim_device = cfg.sim_device

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=self: obj._on_keyboard_event(event, *args),
        )

        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        self._left_close_gripper = False
        self._right_close_gripper = False
        self._left_delta_pos = np.zeros(3)
        self._left_delta_rot = np.zeros(3)
        self._right_delta_pos = np.zeros(3)
        self._right_delta_rot = np.zeros(3)

        self._create_key_bindings()

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        except Exception:
            pass
        self._keyboard_sub = None

    def __str__(self) -> str:
        msg = f"Bimanual Keyboard Controller (SE(3)+gripper): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLeft gripper toggle: K\n"
        msg += "\tLeft move: W/S (x), A/D (y), Q/E (z)\n"
        msg += "\tLeft rotate: Z/X (rx), T/G (ry), C/V (rz)\n"
        msg += "\tRight gripper toggle: P\n"
        msg += "\tRight move: I/F (x), J/L (y), U/O (z)\n"
        msg += "\tRight rotate: N/M (rx), Y/B (ry), 1/2 (rz)\n"
        msg += "\tReset device state: R\n"
        return msg

    def reset(self) -> None:
        self._left_close_gripper = False
        self._right_close_gripper = False
        self._left_delta_pos = np.zeros(3)
        self._left_delta_rot = np.zeros(3)
        self._right_delta_pos = np.zeros(3)
        self._right_delta_rot = np.zeros(3)

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        left_rot_vec = Rotation.from_euler("XYZ", self._left_delta_rot).as_rotvec()
        right_rot_vec = Rotation.from_euler("XYZ", self._right_delta_rot).as_rotvec()
        left_cmd = np.concatenate([self._left_delta_pos, left_rot_vec, [-1.0 if self._left_close_gripper else 1.0]])
        right_cmd = np.concatenate(
            [self._right_delta_pos, right_rot_vec, [-1.0 if self._right_close_gripper else 1.0]]
        )
        cmd = np.concatenate([left_cmd, right_cmd])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _create_key_bindings(self) -> None:
        ps = self.pos_sensitivity
        rs = self.rot_sensitivity

        self._LEFT_POS = {
            "W": np.asarray([1.0, 0.0, 0.0]) * ps,
            "S": np.asarray([-1.0, 0.0, 0.0]) * ps,
            "A": np.asarray([0.0, 1.0, 0.0]) * ps,
            "D": np.asarray([0.0, -1.0, 0.0]) * ps,
            "Q": np.asarray([0.0, 0.0, 1.0]) * ps,
            "E": np.asarray([0.0, 0.0, -1.0]) * ps,
        }
        self._LEFT_ROT = {
            "Z": np.asarray([1.0, 0.0, 0.0]) * rs,
            "X": np.asarray([-1.0, 0.0, 0.0]) * rs,
            "T": np.asarray([0.0, 1.0, 0.0]) * rs,
            "G": np.asarray([0.0, -1.0, 0.0]) * rs,
            "C": np.asarray([0.0, 0.0, 1.0]) * rs,
            "V": np.asarray([0.0, 0.0, -1.0]) * rs,
        }
        self._RIGHT_POS = {
            "I": np.asarray([1.0, 0.0, 0.0]) * ps,
            "F": np.asarray([-1.0, 0.0, 0.0]) * ps,
            "J": np.asarray([0.0, 1.0, 0.0]) * ps,
            "L": np.asarray([0.0, -1.0, 0.0]) * ps,
            "U": np.asarray([0.0, 0.0, 1.0]) * ps,
            "O": np.asarray([0.0, 0.0, -1.0]) * ps,
        }
        self._RIGHT_ROT = {
            "N": np.asarray([1.0, 0.0, 0.0]) * rs,
            "M": np.asarray([-1.0, 0.0, 0.0]) * rs,
            "Y": np.asarray([0.0, 1.0, 0.0]) * rs,
            "B": np.asarray([0.0, -1.0, 0.0]) * rs,
            "1": np.asarray([0.0, 0.0, 1.0]) * rs,
            "2": np.asarray([0.0, 0.0, -1.0]) * rs,
        }

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name
            if key == "R":
                self.reset()
            elif key == "K":
                self._left_close_gripper = not self._left_close_gripper
            elif key == "P":
                self._right_close_gripper = not self._right_close_gripper
            elif key in self._LEFT_POS:
                self._left_delta_pos += self._LEFT_POS[key]
            elif key in self._LEFT_ROT:
                self._left_delta_rot += self._LEFT_ROT[key]
            elif key in self._RIGHT_POS:
                self._right_delta_pos += self._RIGHT_POS[key]
            elif key in self._RIGHT_ROT:
                self._right_delta_rot += self._RIGHT_ROT[key]

            callback = self._additional_callbacks.get(key)
            if callback:
                callback()

        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            if key in self._LEFT_POS:
                self._left_delta_pos -= self._LEFT_POS[key]
            elif key in self._LEFT_ROT:
                self._left_delta_rot -= self._LEFT_ROT[key]
            elif key in self._RIGHT_POS:
                self._right_delta_pos -= self._RIGHT_POS[key]
            elif key in self._RIGHT_ROT:
                self._right_delta_rot -= self._RIGHT_ROT[key]

        return True


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


# =======================================================================================
# Recording Logic
# =======================================================================================


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment() -> tuple[gym.Env, int, object]:
    """Create and configure the environment for recording.

    Returns:
        tuple[gym.Env, int, object]: A tuple containing:
            - env: The configured environment
            - expected_action_dim: Expected action dimension
            - env_cfg: Environment configuration (needed for VR teleop)
    """
    # parse configuration
    try:
        arena_builder = get_arena_builder_from_cli(args_cli)
        env_name, env_cfg = arena_builder.build_registered()
    except Exception as e:
        omni.log.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # modify configuration for recording
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # configure recorder
    output_dir, output_file_name = setup_output_directories()
    
    # Setup recorders - use ArenaEnvRecorderManagerCfg when recording images
    if args_cli.record_images:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
        print("[INFO] Camera image recording enabled - HDF5 file size will be larger")
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
        print("[INFO] Recording actions and states only (no images)")
        
        # Remove camera observations from policy group to avoid recording images
        camera_obs_keys = ["left_wrist_cam", "right_wrist_cam", "head_cam", "robot_pov_cam_rgb"]
        for cam_key in camera_obs_keys:
            if hasattr(env_cfg.observations.policy, cam_key):
                delattr(env_cfg.observations.policy, cam_key)
                print(f"[INFO] Removed {cam_key} from observations")
    
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    # create environment
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
    except Exception as e:
        omni.log.error(f"Failed to create environment: {e}")
        exit(1)

    expected_action_dim = _get_expected_action_dim(env) or env.action_space.shape[-1]
    return env, expected_action_dim, env_cfg


def create_teleop_interface(env, env_cfg) -> object:
    """Create teleop interface based on device type."""
    device_name = getattr(args_cli, "teleop_device", "keyboard") or "keyboard"
    
    if _teleop_device_requires_xr(device_name):
        # VR device - use factory
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
        # Keyboard device
        sensitivity = float(args_cli.sensitivity)
        teleop_interface = BimanualSe3Keyboard(
            BimanualSe3KeyboardCfg(
                pos_sensitivity=0.05 * sensitivity,
                rot_sensitivity=0.05 * sensitivity,
                sim_device=env.device,
            )
        )
        print("[INFO] Using keyboard teleop device")
        return teleop_interface


def main() -> None:
    """Main function to record demonstrations with bimanual keyboard or VR control."""
    # Create environment
    env, expected_action_dim, env_cfg = create_environment()

    # Set up teleoperation interface (keyboard or VR)
    teleop_interface = create_teleop_interface(env, env_cfg)

    # State variables
    should_reset = False
    recorded_demos = 0
    device_name = getattr(args_cli, "teleop_device", None)
    is_vr = _teleop_device_requires_xr(device_name)
    
    # For VR: wait for START button. For keyboard: start immediately
    running_recording = not is_vr

    def reset_and_export() -> None:
        """Export current demo and reset environment."""
        nonlocal recorded_demos
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.export_episodes([0])
        env.recorder_manager.reset([0])
        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        recorded_demos += 1
        print(f"[{recorded_demos}/{args_cli.num_demos if args_cli.num_demos > 0 else '∞'}] Demo exported and environment reset")

    def request_reset() -> None:
        """Request reset on next loop iteration."""
        nonlocal should_reset
        should_reset = True

    def start_recording() -> None:
        """Start recording (for VR control)."""
        nonlocal running_recording
        running_recording = True
        print("[INFO] Recording started")

    def stop_recording() -> None:
        """Stop recording (for VR control)."""
        nonlocal running_recording
        running_recording = False
        print("[INFO] Recording paused")

    # Add callbacks
    teleop_interface.add_callback("R", request_reset)
    teleop_interface.add_callback("RESET", request_reset)
    teleop_interface.add_callback("START", start_recording)
    teleop_interface.add_callback("STOP", stop_recording)

    # Set up rate limiter (not needed for VR devices)
    rate_limiter = None if is_vr else RateLimiter(args_cli.step_hz)

    # Reset before starting
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    # Print control instructions
    if not is_vr:
        print(f"Using teleop device:\n{teleop_interface}")
    print("=" * 60)
    if is_vr:
        print("VR Mode: Click START button in VR to begin recording")
        print("         Click STOP to pause, RESET to discard and restart")
    print("Press 'R' to export current demo and reset for next recording.")
    print(f"Target: {args_cli.num_demos if args_cli.num_demos > 0 else '∞ (infinite)'} demonstrations")
    print("=" * 60)

    # Main recording loop
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running():
            # Get action from teleop interface
            action = teleop_interface.advance()
            device_action = _map_action_dim(action, expected_action_dim)

            # Step environment only if recording is active
            if running_recording:
                env.step(device_action.repeat(env.num_envs, 1))
            else:
                # Just render if not recording (VR waiting for START)
                env.sim.render()

            # Handle reset request
            if should_reset:
                reset_and_export()
                should_reset = False

            # Check if target number of demos reached
            if args_cli.num_demos > 0 and recorded_demos >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting.")
                break

            # Check if simulation is stopped
            if env.sim.is_stopped():
                break

            # Rate limiting (only for keyboard mode)
            if rate_limiter:
                rate_limiter.sleep(env)

    # Clean up
    env.close()
    print(f"Recording session completed with {recorded_demos} demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
