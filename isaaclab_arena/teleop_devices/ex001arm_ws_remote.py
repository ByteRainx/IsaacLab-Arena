# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Remote WebSocket teleoperation device for EX001Arm.

Receives end-effector states from a remote machine (robot PC) over WebSocket,
computes frame-to-frame delta poses, and outputs 14D action tensors compatible
with IsaacLab-Arena's ``EX001ArmActionsCfg`` (DifferentialIK).

The remote machine runs ``tools/ros1_ws_bridge.py`` which subscribes to
ROS1 ``PosCmd`` topics and forwards the data via WebSocket.

Requirements (simulation PC only):
    pip install websocket-client
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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

    pos_scale: float = 1.0
    """Scale factor for position deltas. Increase if sim arm moves too little,
    decrease if it moves too much."""

    rot_scale: float = 1.0
    """Scale factor for rotation deltas."""

    gripper_max: float = _MASTER_GRIPPER_MAX
    """Physical master gripper value when fully open."""

    gripper_min: float = _MASTER_GRIPPER_MIN
    """Physical master gripper value when fully closed."""

    reconnect_interval: float = 2.0
    """Seconds between reconnection attempts when the link is down."""

    debug: bool = False
    """Print incoming data periodically for debugging."""


class Ex001ArmWsRemoteTeleop:
    """Remote bimanual teleop device that receives EE states via WebSocket.

    Implements the same interface as
    :class:`~isaaclab_arena.scripts.teleop_bimanual_keyboard.BimanualSe3Keyboard`:

    * ``advance() -> torch.Tensor``  (14-dim action)
    * ``reset() -> None``
    * ``add_callback(key, func) -> None``

    **Output layout (14D)**::

        [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
         right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]

    * Position delta: difference between consecutive EE positions (metres)
    * Rotation delta: rotation vector from consecutive Euler-angle frames
    * Gripper: ``-1`` (closed) to ``+1`` (open)
    """

    def __init__(self, cfg: Ex001ArmWsRemoteCfg):
        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        # Latest state received from WebSocket (written by bg thread)
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._connected = False

        # Previous EE poses for delta computation
        self._prev_left: np.ndarray | None = None   # [x,y,z,roll,pitch,yaw]
        self._prev_right: np.ndarray | None = None

        # Start background receiver thread
        self._ws = None
        self._should_run = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        # Start keyboard listener for callbacks
        self._listener = None
        self._setup_keyboard_listener()

        logger.info(
            "WS remote teleop: connecting to ws://%s:%d",
            cfg.remote_ip,
            cfg.remote_port,
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
    # Keyboard listener
    # ------------------------------------------------------------------

    def _setup_keyboard_listener(self) -> None:
        try:
            from pynput import keyboard as pynput_keyboard

            def _on_press(key):
                try:
                    ch = key.char.upper() if hasattr(key, "char") and key.char else None
                except AttributeError:
                    ch = None
                if ch and ch in self._additional_callbacks:
                    self._additional_callbacks[ch]()

            self._listener = pynput_keyboard.Listener(on_press=_on_press)
            self._listener.daemon = True
            self._listener.start()
        except ImportError:
            logger.warning("pynput not available; keyboard callbacks disabled.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        self._should_run = False
        if getattr(self, "_listener", None) is not None:
            self._listener.stop()

    def __str__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        lines = [
            f"WS Remote Teleop (EE mode): {self.__class__.__name__}",
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
        """Read latest remote state and return 14D EE delta action."""
        with self._lock:
            state = self._latest

        # No data yet -> zero action
        if state is None or state.get("left") is None or state.get("right") is None:
            return torch.zeros(14, dtype=torch.float32, device=self._sim_device)

        left = state["left"]
        right = state["right"]

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
        left_grip = self._normalize_gripper(left["gripper"])
        right_grip = self._normalize_gripper(right["gripper"])

        # Debug: print data periodically
        if self.cfg.debug:
            self._debug_counter = getattr(self, "_debug_counter", 0) + 1
            if self._debug_counter % 50 == 0:  # every ~1s at 50Hz
                print(
                    f"[WS Debug] L_grip_raw={left['gripper']:.2f} -> {left_grip:.2f}  "
                    f"R_grip_raw={right['gripper']:.2f} -> {right_grip:.2f}  "
                    f"L_delta_pos={np.linalg.norm(left_delta[:3]):.5f}"
                )

        cmd = np.concatenate([left_delta, [left_grip], right_delta, [right_grip]])
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
