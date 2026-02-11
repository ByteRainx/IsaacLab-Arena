# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""ARX X5 bimanual teleoperation device for EX001Arm.

Reads joint/EEF states from two physical ARX arms via CAN bus using arx5-sdk,
and outputs 14D action tensors compatible with IsaacLab-Arena's ex001arm embodiment.

Supports two control modes:

- ``ee``: End-effector (relative delta pose) -- use with ``EX001ArmActionsCfg`` (DifferentialIK)
- ``joint``: Absolute joint positions -- use with ``EX001ArmJointActionsCfg``

Requirements:
    - arx5-sdk Python bindings on ``PYTHONPATH``
    - CAN interfaces activated, e.g.::

        sudo ip link set up can0 type can bitrate 1000000
        sudo ip link set up can1 type can bitrate 1000000
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# arx5-sdk imports (runtime dependency -- graceful fallback when unavailable)
# ---------------------------------------------------------------------------
try:
    from arx5_interface import (
        Arx5CartesianController,
        Arx5JointController,
    )

    _ARX5_AVAILABLE = True
except ImportError:
    _ARX5_AVAILABLE = False
    logger.warning(
        "arx5_interface not found. Install arx5-sdk and add to PYTHONPATH "
        "to use ARX teleoperation."
    )

# ─── Constants ────────────────────────────────────────────────────────────────
# Fallback ARX X5 gripper width in metres (fully open).
# Overridden at runtime via ``RobotConfig.gripper_width`` when available.
_DEFAULT_GRIPPER_WIDTH = 0.088
# EX001Arm USD gripper joint range: 0.0 (close) → 5.0 (open)
_SIM_GRIPPER_OPEN = 5.0


# ─── Configuration dataclass ─────────────────────────────────────────────────
@dataclass
class Ex001ArmArxBimanualCfg:
    """Configuration for :class:`Ex001ArmArxBimanualTeleop`."""

    model: str = "X5"
    """ARX arm model identifier (``X5``, ``L5``, ``X7``, …)."""

    left_interface: str = "can0"
    """CAN bus interface name for the **left** arm."""

    right_interface: str = "can1"
    """CAN bus interface name for the **right** arm."""

    control_mode: str = "ee"
    """Control mode: ``"ee"`` for end-effector delta pose, ``"joint"`` for absolute
    joint positions."""

    sim_device: str | None = None
    """Torch device string for the output tensor (e.g. ``"cuda:0"``)."""

    damping_scale: float = 0.1
    """Scale factor applied to the default ``kd`` gains when entering passive
    teaching mode.  Smaller → easier to drag by hand."""


# ─── Main teleop class ───────────────────────────────────────────────────────
class Ex001ArmArxBimanualTeleop:
    """Bimanual ARX arm teleop device that reads physical arm states via CAN.

    The class follows the same interface as
    :class:`~isaaclab_arena.scripts.teleop_bimanual_keyboard.BimanualSe3Keyboard`:

    * ``advance() → torch.Tensor``  (14-dim action)
    * ``reset() → None``
    * ``add_callback(key, func) → None``

    **EE mode** output layout (14D)::

        [left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
         right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]

    **Joint mode** output layout (14D)::

        [left_j1..j6, left_gripper, right_j1..j6, right_gripper]
    """

    def __init__(self, cfg: Ex001ArmArxBimanualCfg):
        if not _ARX5_AVAILABLE:
            raise RuntimeError(
                "arx5_interface is not available.  "
                "Install arx5-sdk and add its Python directory to PYTHONPATH."
            )

        self.cfg = cfg
        self._sim_device = cfg.sim_device
        self._control_mode = cfg.control_mode
        self._additional_callbacks: dict[str, Callable[[], None]] = {}

        # --- Create arm controllers based on control mode ----------------
        if cfg.control_mode == "ee":
            self._left_ctrl = Arx5CartesianController(cfg.model, cfg.left_interface)
            self._right_ctrl = Arx5CartesianController(cfg.model, cfg.right_interface)
        else:
            self._left_ctrl = Arx5JointController(cfg.model, cfg.left_interface)
            self._right_ctrl = Arx5JointController(cfg.model, cfg.right_interface)

        # --- Resolve gripper width from the robot config ------------------
        robot_config = self._left_ctrl.get_robot_config()
        self._gripper_width: float = (
            robot_config.gripper_width
            if robot_config.gripper_width > 0
            else _DEFAULT_GRIPPER_WIDTH
        )

        # --- Enter passive teaching mode (damping only) -------------------
        self._left_ctrl.set_to_damping()
        self._right_ctrl.set_to_damping()

        # Reduce damping so the operator can drag arms more easily
        for ctrl in (self._left_ctrl, self._right_ctrl):
            gain = ctrl.get_gain()
            gain.kd()[:] *= cfg.damping_scale
            ctrl.set_gain(gain)

        # --- EE-mode bookkeeping: previous pose for delta computation -----
        self._prev_left_pose: np.ndarray | None = None
        self._prev_right_pose: np.ndarray | None = None

        # --- Background keyboard listener for callbacks -------------------
        self._listener = None
        self._setup_keyboard_listener()

        logger.info(
            "ARX bimanual teleop initialised: model=%s  left=%s  right=%s  mode=%s",
            cfg.model,
            cfg.left_interface,
            cfg.right_interface,
            cfg.control_mode,
        )

    # ------------------------------------------------------------------
    # Keyboard listener (pynput)
    # ------------------------------------------------------------------

    def _setup_keyboard_listener(self) -> None:
        """Start a daemon thread that listens for key presses."""
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

    def __del__(self) -> None:
        if getattr(self, "_listener", None) is not None:
            self._listener.stop()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # noqa: D105
        lines = [
            f"ARX Bimanual Teleop ({self.cfg.model}): {self.__class__.__name__}",
            f"\tLeft interface : {self.cfg.left_interface}",
            f"\tRight interface: {self.cfg.right_interface}",
            f"\tControl mode   : {self._control_mode}",
            "\t----------------------------------------------",
            "\tR : Reset environment",
        ]
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset internal state.  Re-reads current poses as the baseline for EE deltas."""
        if self._control_mode == "ee":
            left_eef = self._left_ctrl.get_eef_state()
            right_eef = self._right_ctrl.get_eef_state()
            self._prev_left_pose = left_eef.pose_6d().copy()
            self._prev_right_pose = right_eef.pose_6d().copy()

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        """Register a callback triggered when *key* is pressed."""
        self._additional_callbacks[key] = func

    def advance(self) -> torch.Tensor:
        """Read arm states and return a 14-D action tensor."""
        if self._control_mode == "ee":
            return self._advance_ee()
        return self._advance_joint()

    # ------------------------------------------------------------------
    # EE (end-effector) mode
    # ------------------------------------------------------------------

    def _advance_ee(self) -> torch.Tensor:
        left_eef = self._left_ctrl.get_eef_state()
        right_eef = self._right_ctrl.get_eef_state()

        left_pose = left_eef.pose_6d().copy()
        right_pose = right_eef.pose_6d().copy()

        # First call: initialise reference poses
        if self._prev_left_pose is None:
            self._prev_left_pose = left_pose.copy()
        if self._prev_right_pose is None:
            self._prev_right_pose = right_pose.copy()

        left_delta = self._compute_pose_delta(self._prev_left_pose, left_pose)
        right_delta = self._compute_pose_delta(self._prev_right_pose, right_pose)

        self._prev_left_pose = left_pose
        self._prev_right_pose = right_pose

        left_grip = self._normalize_gripper_ee(left_eef.gripper_pos)
        right_grip = self._normalize_gripper_ee(right_eef.gripper_pos)

        cmd = np.concatenate([left_delta, [left_grip], right_delta, [right_grip]])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    @staticmethod
    def _compute_pose_delta(
        prev_pose_6d: np.ndarray, curr_pose_6d: np.ndarray
    ) -> np.ndarray:
        """Compute a 6-D delta ``[dx, dy, dz, drx, dry, drz]``.

        ``pose_6d`` from arx5-sdk is ``[x, y, z, rx, ry, rz]`` where the
        rotation part is a rotation vector (axis-angle in radians).  The output
        rotation delta is also a rotation vector, matching what
        ``DifferentialIKController`` with ``use_relative_mode=True`` expects.
        """
        delta_pos = curr_pose_6d[:3] - prev_pose_6d[:3]

        R_prev = Rotation.from_rotvec(prev_pose_6d[3:6])
        R_curr = Rotation.from_rotvec(curr_pose_6d[3:6])
        delta_rot = (R_curr * R_prev.inv()).as_rotvec()

        return np.concatenate([delta_pos, delta_rot])

    def _normalize_gripper_ee(self, gripper_pos: float) -> float:
        """Map gripper position (metres) → ``-1`` (close) / ``+1`` (open)."""
        ratio = float(np.clip(gripper_pos / self._gripper_width, 0.0, 1.0))
        return ratio * 2.0 - 1.0

    # ------------------------------------------------------------------
    # Joint mode
    # ------------------------------------------------------------------

    def _advance_joint(self) -> torch.Tensor:
        left_state = self._left_ctrl.get_joint_state()
        right_state = self._right_ctrl.get_joint_state()

        left_pos = left_state.pos().copy()   # radians, shape (6,)
        right_pos = right_state.pos().copy()

        left_grip = self._map_gripper_joint(left_state.gripper_pos)
        right_grip = self._map_gripper_joint(right_state.gripper_pos)

        cmd = np.concatenate([left_pos, [left_grip], right_pos, [right_grip]])
        return torch.tensor(cmd, dtype=torch.float32, device=self._sim_device)

    def _map_gripper_joint(self, gripper_pos: float) -> float:
        """Map gripper position (metres) → simulation joint value (``0`` … ``5.0``)."""
        ratio = float(np.clip(gripper_pos / self._gripper_width, 0.0, 1.0))
        return ratio * _SIM_GRIPPER_OPEN
