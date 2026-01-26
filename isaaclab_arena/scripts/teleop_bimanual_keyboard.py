# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to run a bimanual keyboard teleoperation for EX001Arm-style control."""

"""Launch Isaac Sim Simulator first."""

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task name to expose in logs and teleop logic (e.g. pick_and_place).",
)

# Add the example environments CLI args
# NOTE: This has to be added last, because of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# ensure task exists (fallback to selected example env)
if not hasattr(args_cli, "task") or args_cli.task is None:
    args_cli.task = getattr(args_cli, "example_environment", None) or getattr(args_cli, "environment", "") or ""

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import omni.log
import torch
from scipy.spatial.transform import Rotation

import carb
import omni


@dataclass
class BimanualSe3KeyboardCfg:
    pos_sensitivity: float = 0.05
    rot_sensitivity: float = 0.5 
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


def main() -> None:
    """
    Run bimanual keyboard teleoperation with Isaac Lab environments.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.
    """
    # parse configuration
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    # modify configuration
    env_cfg.terminations.time_out = None

    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
    except Exception as exc:
        omni.log.error(f"Failed to create environment: {exc}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset = False
    teleoperation_active = True

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Callback handlers
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_env,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_env,
    }

    sensitivity = float(args_cli.sensitivity)
    cfg = BimanualSe3KeyboardCfg()
    teleop_interface = BimanualSe3Keyboard(
        BimanualSe3KeyboardCfg(
            pos_sensitivity=cfg.pos_sensitivity * sensitivity,
            rot_sensitivity=cfg.rot_sensitivity * sensitivity
        )
    )
    for key, callback in teleoperation_callbacks.items():
        teleop_interface.add_callback(key, callback)

    print(f"Using teleop device:\n{teleop_interface}")

    expected_action_dim = _get_expected_action_dim(env)
    # reset environment
    env.reset()
    teleop_interface.reset()

    print("Teleoperation started. Press 'R' to reset the environment.")
    warned_action_mismatch = False

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                action = teleop_interface.advance()
                if teleoperation_active:
                    device_action = action
                    if expected_action_dim is not None:
                        device_action = _map_action_dim(device_action, expected_action_dim)
                        if device_action.shape[-1] != action.shape[-1] and not warned_action_mismatch:
                            omni.log.warn(
                                f"Teleop action dim ({action.shape[-1]}) != env action dim ({expected_action_dim}). "
                                "Auto-mapping enabled."
                            )
                            warned_action_mismatch = True
                    # apply actions
                    env.step(device_action.repeat(env.num_envs, 1))
                else:
                    env.sim.render()

                if should_reset:
                    env.reset()
                    should_reset = False
                    print("Environment reset complete")
        except Exception as exc:
            omni.log.error(f"Error during simulation step: {exc}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
