# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote WebSocket teleoperation device for EX001Arm (EE + Joint modes).

Receives arm states from a remote machine (robot PC) over WebSocket and
outputs action tensors for IsaacLab-Arena.

**Two control modes:**

* ``ee``    -- End-effector delta mode.  Computes frame-to-frame delta
  poses from ``PosCmd`` data and outputs 14D tensors for
  ``DifferentialInverseKinematicsActionCfg``.
* ``joint`` -- Joint position direct mode.  Passes through absolute joint
  positions from ``JointInformation`` data and outputs 14D tensors for
  ``JointPositionActionCfg``.

The remote machine runs ``tools/ros1_ws_bridge.py`` which subscribes to
ROS1 topics and forwards the data via WebSocket.

Requirements (simulation PC only):
    pip install websocket-client
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# PosCmd gripper range on the physical master arm
_MASTER_GRIPPER_MAX = 4.5   # fully open
_MASTER_GRIPPER_MIN = 0.0   # fully closed


@dataclass
class Ex001ArmWsRemoteCfg:
    """Configuration for :class:`Ex001ArmWsRemoteTeleop`."""

    remote_ip: str = "192.168.1.100"
    """IP address of the robot PC running ``ros1_ws_bridge.py``."""

    remote_port: int = 5555
    """WebSocket port on the robot PC."""

    sim_device: str | None = None
    """Torch device string for the output tensor (e.g. ``"cuda:0"``)."""

    control_mode: str = "ee"
    """Control mode: ``"ee"`` (end-effector delta) or ``"joint"`` (joint
    position direct).  Must match the bridge ``--mode`` setting."""

    pos_scale: float = 1.0
    """(EE mode only) Scale factor for position deltas."""

    rot_scale: float = 1.0
    """(EE mode only) Scale factor for rotation deltas."""

    gripper_max: float = _MASTER_GRIPPER_MAX
    """Physical master gripper value when fully open."""

    gripper_min: float = _MASTER_GRIPPER_MIN
    """Physical master gripper value when fully closed."""

    reconnect_interval: float = 2.0
    """Seconds between reconnection attempts when the link is down."""

    debug: bool = False
    """Print incoming data periodically for debugging."""

    # -- Joint mode mapping --------------------------------------------------

    joint_signs: tuple[float, ...] = (1.0, 1.0, -1.0, -1.0, -1.0, -1.0)
    """Per-joint sign multipliers (6 values, one per arm joint).
    Use ``-1.0`` to invert a joint's direction when the physical arm
    and the simulation model have opposite axis conventions.

    Default ``(1, 1, -1, -1, -1, -1)`` matches the ARX X5 physical arm
    against the ex001arm simulation USD model (joints 3-6 are inverted
    due to URDF origin rotations during USD conversion).
    """

    joint_offsets: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    """Per-joint offset in radians added *after* sign multiplication.
    Useful when the simulation zero and the physical zero differ.

    ``sim_joint = sign * physical_joint + offset``
    """


class Ex001ArmWsRemoteTeleop:
    """Remote bimanual teleop device that receives arm states via WebSocket.

    Implements the same interface used by other teleop devices:

    * ``advance() -> torch.Tensor``  (14-dim action)
    * ``reset() -> None``
    * ``add_callback(key, func) -> None``

    **EE mode output layout (14D)**::

        [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
         right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]

    **Joint mode output layout (14D)**::

        [left_j1, left_j2, left_j3, left_j4, left_j5, left_j6, left_grip,
         right_j1, right_j2, right_j3, right_j4, right_j5, right_j6, right_grip]
    """

    def __init__(self, cfg: Ex001ArmWsRemoteCfg):
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._control_mode = cfg.control_mode
        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        # Latest state received from WebSocket (written by bg thread)
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._connected = False

        # Previous EE poses for delta computation (EE mode only)
        self._prev_left: np.ndarray | None = None   # [x,y,z,roll,pitch,yaw]
        self._prev_right: np.ndarray | None = None

        # Joint mapping arrays (joint mode)
        self._joint_signs = np.array(cfg.joint_signs, dtype=np.float64)
        self._joint_offsets = np.array(cfg.joint_offsets, dtype=np.float64)

        # Debug counter
        self._debug_counter = 0

        # Start background receiver thread
        self._ws = None
        self._should_run = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        # Start keyboard listener for callbacks (uses carb.input, same as other teleop devices)
        self._keyboard_sub = None
        self._setup_keyboard_listener()

        logger.info(
            "WS remote teleop (%s mode): connecting to ws://%s:%d",
            cfg.control_mode, cfg.remote_ip, cfg.remote_port,
        )

    # ------------------------------------------------------------------
    # Background WebSocket receiver
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """Connect to the bridge and continuously receive state updates."""
        import websocket  # websocket-client (sync)

        url = f"ws://{self.cfg.remote_ip}:{self.cfg.remote_port}"
        while self._should_run:
            try:
                ws = websocket.create_connection(url, timeout=5)
                self._connected = True
                print(f"[WS Teleop] Connected to {url}")
                while self._should_run:
                    raw = ws.recv()
                    data = json.loads(raw)
                    with self._lock:
                        self._latest = data
            except Exception as e:
                if self._connected:
                    print(f"[WS Teleop] Disconnected: {e}")
                self._connected = False
                time.sleep(self.cfg.reconnect_interval)

    # ------------------------------------------------------------------
    # Keyboard listener (carb.input — Isaac Sim built-in, works reliably)
    # ------------------------------------------------------------------

    def _setup_keyboard_listener(self) -> None:
        """Register a keyboard listener via Isaac Sim's carb.input system.

        This is the same mechanism used by ``Se3Keyboard`` and
        ``BimanualSe3Keyboard`` and works whenever the Isaac Sim viewport
        window has focus (unlike pynput which is blocked by carb).
        """
        try:
            import carb.input
            import omni.appwindow

            appwindow = omni.appwindow.get_default_app_window()
            input_iface = carb.input.acquire_input_interface()
            keyboard = appwindow.get_keyboard()

            def _on_keyboard_event(event, *args, **kwargs) -> bool:
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    key_name = event.input.name  # e.g. "R", "ESCAPE", …
                    cb = self._additional_callbacks.get(key_name)
                    if cb is not None:
                        cb()
                return True

            self._keyboard_sub = input_iface.subscribe_to_keyboard_events(
                keyboard, _on_keyboard_event
            )
        except Exception as e:
            logger.warning("carb.input keyboard listener unavailable: %s", e)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        self._should_run = False

    def __str__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        mode_label = self._control_mode.upper()
        lines = [
            f"WS Remote Teleop ({mode_label} mode): {self.__class__.__name__}",
            f"\tRemote    : ws://{self.cfg.remote_ip}:{self.cfg.remote_port}",
            f"\tStatus    : {status}",
            "\t----------------------------------------------",
            "\tR : Reset environment",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset internal state; next advance() will use current pose as baseline."""
        self._prev_left = None
        self._prev_right = None

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Read latest remote state and return 14D action tensor."""
        if self._control_mode == "joint":
            return self._advance_joint()
        return self._advance_ee()

    # ------------------------------------------------------------------
    # EE mode
    # ------------------------------------------------------------------

    def _advance_ee(self) -> torch.Tensor:
        """Compute 14D EE delta action from PosCmd data."""
        with self._lock:
            state = self._latest

        # No data yet -> zero action
        if state is None or state.get("left") is None or state.get("right") is None:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        left = state["left"]
        right = state["right"]

        # Check that EE fields are present
        if "x" not in left or "x" not in right:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        # Extract current poses [x, y, z, roll, pitch, yaw]
        left_pose = np.array(
            [left["x"], left["y"], left["z"],
             left["roll"], left["pitch"], left["yaw"]],
            dtype=np.float64,
        )
        right_pose = np.array(
            [right["x"], right["y"], right["z"],
             right["roll"], right["pitch"], right["yaw"]],
            dtype=np.float64,
        )

        # First frame -> initialise, return zeros
        if self._prev_left is None:
            self._prev_left = left_pose.copy()
        if self._prev_right is None:
            self._prev_right = right_pose.copy()

        # Compute deltas and apply scale
        left_delta = self._compute_pose_delta(self._prev_left, left_pose)
        right_delta = self._compute_pose_delta(self._prev_right, right_pose)

        left_delta[:3] *= self.cfg.pos_scale
        left_delta[3:] *= self.cfg.rot_scale
        right_delta[:3] *= self.cfg.pos_scale
        right_delta[3:] *= self.cfg.rot_scale

        self._prev_left = left_pose.copy()
        self._prev_right = right_pose.copy()

        # Gripper: 0 (close) ~ 4.5 (open) -> -1 (close) ~ +1 (open)
        left_grip = self._normalize_gripper(left.get("gripper", 0.0))
        right_grip = self._normalize_gripper(right.get("gripper", 0.0))

        # Debug
        if self.cfg.debug:
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:
                print(
                    f"[EE Debug] L_grip_raw={left.get('gripper', 0):.2f} -> {left_grip:.2f}  "
                    f"R_grip_raw={right.get('gripper', 0):.2f} -> {right_grip:.2f}  "
                    f"L_dpos={np.linalg.norm(left_delta[:3]):.5f}  "
                    f"R_dpos={np.linalg.norm(right_delta[:3]):.5f}"
                )

        cmd = np.concatenate([left_delta, [left_grip], right_delta, [right_grip]])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    # ------------------------------------------------------------------
    # Joint mode
    # ------------------------------------------------------------------

    def _advance_joint(self) -> torch.Tensor:
        """Return 14D absolute joint position action from JointControl data.

        Layout: [left_j1..j6, left_gripper, right_j1..j6, right_gripper]

        ``joint_pos`` from the bridge is a 7-element array where indices
        0-5 are joint angles and index 6 is the gripper joint position.

        Each joint value is transformed as::

            sim_joint[i] = joint_signs[i] * physical_joint[i] + joint_offsets[i]

        This handles axis direction differences and zero-offset mismatches
        between the physical arm and the simulation model.
        """
        with self._lock:
            state = self._latest

        # No data yet -> zero action
        if state is None or state.get("left") is None or state.get("right") is None:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        left = state["left"]
        right = state["right"]

        # Check that joint fields are present
        if "joint_pos" not in left or "joint_pos" not in right:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        left_jp = np.array(left["joint_pos"], dtype=np.float64)
        right_jp = np.array(right["joint_pos"], dtype=np.float64)

        # joint_pos[0:6] = joint angles, joint_pos[6] = gripper
        left_joints_raw = left_jp[:6]
        left_grip = left_jp[6] if len(left_jp) > 6 else 0.0
        right_joints_raw = right_jp[:6]
        right_grip = right_jp[6] if len(right_jp) > 6 else 0.0

        # Apply per-joint sign and offset mapping
        # sim_joint = sign * physical_joint + offset
        left_joints = self._joint_signs * left_joints_raw + self._joint_offsets
        right_joints = self._joint_signs * right_joints_raw + self._joint_offsets

        # Debug -- print every 50 frames: raw input, mapped output, and per-joint diff
        if self.cfg.debug:
            self._debug_counter += 1
            if self._debug_counter % 50 == 0:
                print(
                    f"[Joint Debug #{self._debug_counter}] "
                    f"signs={self._joint_signs.tolist()}  "
                    f"offsets={np.round(self._joint_offsets, 4).tolist()}"
                )
                for side, raw, mapped, grip in [
                    ("L", left_joints_raw, left_joints, left_grip),
                    ("R", right_joints_raw, right_joints, right_grip),
                ]:
                    print(
                        f"  {side}_raw   = [{', '.join(f'{v:+.4f}' for v in raw)}]  "
                        f"grip={grip:.3f}"
                    )
                    print(
                        f"  {side}_mapped= [{', '.join(f'{v:+.4f}' for v in mapped)}]"
                    )

        cmd = np.concatenate([left_joints, [left_grip], right_joints, [right_grip]])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pose_delta(
        prev_pose: np.ndarray, curr_pose: np.ndarray
    ) -> np.ndarray:
        """Compute 6D delta [dx, dy, dz, drx, dry, drz] between two
        [x, y, z, roll, pitch, yaw] poses.

        Position delta is a simple subtraction.
        Rotation delta is computed as R_curr * R_prev^-1, output as a rotation vector
        matching what DifferentialIK (use_relative_mode=True) expects.
        """
        delta_pos = curr_pose[:3] - prev_pose[:3]

        R_prev = Rotation.from_euler("xyz", prev_pose[3:6])
        R_curr = Rotation.from_euler("xyz", curr_pose[3:6])
        delta_rot = (R_curr * R_prev.inv()).as_rotvec()

        return np.concatenate([delta_pos, delta_rot])

    def _normalize_gripper(self, gripper_val: float) -> float:
        """Map physical gripper value to -1 (close) / +1 (open)."""
        g_min = self.cfg.gripper_min
        g_max = self.cfg.gripper_max
        ratio = float(np.clip((gripper_val - g_min) / (g_max - g_min), 0.0, 1.0))
        return ratio * 2.0 - 1.0
