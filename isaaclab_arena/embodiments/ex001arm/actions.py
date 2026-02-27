# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gripper action terms for EX001Arm.

* ``ThreeStateGripperAction`` – Simple three-state gripper (open / proportional / grasp-clamp).
  No contact sensors required.
* ``ContactLimitedGripperAction`` – (Legacy) Contact-force based gripper.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ══════════════════════════════════════════════════════════════════════════════
# Three-State Gripper (state-machine with contact-force detection)
# ══════════════════════════════════════════════════════════════════════════════

# Internal state codes (integers on the same device as env tensors)
_STATE_OPEN = 0
_STATE_CLOSE = 1
_STATE_GRASP = 2


class ThreeStateGripperAction(ActionTerm):
    """State-machine gripper with three discrete states: Open / Close / Grasp.

    **State diagram**::

        OPEN ──(cmd=close)──→ CLOSE ──(contact force)──→ GRASP
          ↑                     │                          │
          └──(cmd=open)─────────┘                          │
          └──(cmd=open)────────────────────────────────────┘
          dead-zone: stay in current state

    *   The **input** is interpreted as a binary open/close *command*
        (not a proportional target).  A dead-zone prevents spurious
        transitions when the input is ambiguous.
    *   **CLOSE → GRASP** is triggered *automatically* when the
        :class:`ContactSensor` on the gripper fingers reports force
        above ``force_threshold`` on **both** fingers simultaneously.
    *   **GRASP** is a sticky state — the gripper holds the object at
        ``grasp_pos`` and ignores further "close" commands.  Only an
        explicit "open" command exits GRASP.

    **Absolute input** (``absolute_input = True``, joint teleop):
      - input ≥ ``open_threshold``  → *cmd = open*
      - input ≤ ``close_threshold`` → *cmd = close*
      - between thresholds          → *no command* (dead-zone, keep state)

    **Binary input** (``absolute_input = False``, EE / VR teleop):
      - input > 0  → *cmd = open*
      - input < 0  → *cmd = close*
      - input == 0 → *no command* (keep state)
    """

    cfg: "ThreeStateGripperActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "ThreeStateGripperActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"ThreeStateGripperAction resolved joints: {self._joint_names} [{self._joint_ids}]"
        )

        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Parse the three fixed target positions
        self._open_pos = self._parse_command(self.cfg.open_command_expr)
        self._grasp_pos = self._parse_command(self.cfg.grasp_command_expr)
        self._close_pos = self._parse_command(self.cfg.close_command_expr)

        # ── State machine bookkeeping ──────────────────────────────────
        # Per-env state: 0=OPEN, 1=CLOSE, 2=GRASP
        self._state = torch.full(
            (self.num_envs,), _STATE_OPEN, dtype=torch.long, device=self.device
        )

        # Contact sensor (lazy-initialised on first use)
        self._contact_sensor: ContactSensor | None = None
        self._left_finger_ids: list[int] | None = None
        self._right_finger_ids: list[int] | None = None

    # ── helpers ────────────────────────────────────────────────────────

    def _parse_command(self, expr: dict[str, float]) -> torch.Tensor:
        cmd = torch.zeros(self._num_joints, device=self.device)
        idx, _, vals = string_utils.resolve_matching_names_values(expr, self._joint_names)
        cmd[idx] = torch.tensor(vals, device=self.device)
        return cmd

    def _get_contact_sensor(self) -> ContactSensor | None:
        """Lazy-load the contact sensor from the scene."""
        if self._contact_sensor is None and self.cfg.contact_sensor_name:
            try:
                self._contact_sensor = self._env.scene[self.cfg.contact_sensor_name]
                omni.log.info(
                    f"ThreeStateGripperAction: using contact sensor "
                    f"'{self.cfg.contact_sensor_name}'"
                )
            except KeyError:
                omni.log.warn(
                    f"Contact sensor '{self.cfg.contact_sensor_name}' not found — "
                    f"CLOSE→GRASP transition will never fire."
                )
        return self._contact_sensor

    def _resolve_finger_body_ids(self, sensor: ContactSensor) -> None:
        if self._left_finger_ids is None:
            self._left_finger_ids, _ = sensor.find_bodies(
                self.cfg.left_finger_body_regex, preserve_order=True
            )
        if self._right_finger_ids is None:
            self._right_finger_ids, _ = sensor.find_bodies(
                self.cfg.right_finger_body_regex, preserve_order=True
            )

    def _both_fingers_in_contact(self) -> torch.Tensor:
        """Return a (num_envs,) bool tensor — True where both fingers exceed
        the force threshold simultaneously."""
        sensor = self._get_contact_sensor()
        if sensor is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._resolve_finger_body_ids(sensor)
        if not self._left_finger_ids or not self._right_finger_ids:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        net_forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
        force_mag = torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)
        left_max = force_mag[:, self._left_finger_ids].max(dim=1).values
        right_max = force_mag[:, self._right_finger_ids].max(dim=1).values
        return (left_max > self.cfg.force_threshold) & (right_max > self.cfg.force_threshold)

    # ── ActionTerm interface ──────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        val = actions.squeeze(-1)  # (num_envs,)

        # ── 1. Decode input into cmd_open / cmd_close / dead-zone ─────
        if self.cfg.absolute_input:
            cmd_open = val >= self.cfg.open_threshold
            cmd_close = val <= self.cfg.close_threshold
            # between thresholds → neither flag set → dead-zone
        else:
            cmd_open = val > 0
            cmd_close = val < 0

        # ── 2. State transitions ──────────────────────────────────────
        #
        #   OPEN  + cmd_close          → CLOSE
        #   CLOSE + cmd_open           → OPEN
        #   CLOSE + contact_force      → GRASP   (sticky)
        #   GRASP + cmd_open           → OPEN
        #   everything else            → keep current state
        #
        is_open = self._state == _STATE_OPEN
        is_close = self._state == _STATE_CLOSE
        is_grasp = self._state == _STATE_GRASP

        # OPEN → CLOSE  (only on explicit close command)
        self._state[is_open & cmd_close] = _STATE_CLOSE

        # CLOSE → OPEN  (explicit open command)
        self._state[is_close & cmd_open] = _STATE_OPEN

        # CLOSE → GRASP (contact force detected, regardless of input)
        # Re-evaluate is_close after possible OPEN→CLOSE transition above
        is_close = self._state == _STATE_CLOSE
        force_contact = self._both_fingers_in_contact()
        self._state[is_close & force_contact] = _STATE_GRASP

        # GRASP → OPEN  (explicit open command)
        self._state[is_grasp & cmd_open] = _STATE_OPEN

        # ── 3. Map state → target joint position ─────────────────────
        open_t = self._open_pos.unsqueeze(0).expand(self.num_envs, -1)
        grasp_t = self._grasp_pos.unsqueeze(0).expand(self.num_envs, -1)
        close_t = self._close_pos.unsqueeze(0).expand(self.num_envs, -1)

        # Start from close, layer grasp, then open
        target = close_t.clone()
        target[self._state == _STATE_GRASP] = grasp_t[self._state == _STATE_GRASP]
        target[self._state == _STATE_OPEN] = open_t[self._state == _STATE_OPEN]

        self._processed_actions[:] = target

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._raw_actions[env_ids] = 0.0
        self._state[env_ids] = _STATE_OPEN


@configclass
class ThreeStateGripperActionCfg(ActionTermCfg):
    """Configuration for the three-state gripper state machine.

    The gripper uses a simple state machine (OPEN → CLOSE → GRASP)
    with contact-force detection for the CLOSE→GRASP transition.
    """

    class_type: type[ActionTerm] = ThreeStateGripperAction

    joint_names: list[str] = MISSING
    """Gripper joint name(s)."""

    open_command_expr: dict[str, float] = MISSING
    """Joint target for **OPEN** state (e.g. ``{"left_arm_gripper": 5.0}``)."""

    grasp_command_expr: dict[str, float] = MISSING
    """Joint target for **GRASP** state (e.g. ``{"left_arm_gripper": 1.7}``)."""

    close_command_expr: dict[str, float] = MISSING
    """Joint target for **CLOSE** state (e.g. ``{"left_arm_gripper": 0.0}``)."""

    absolute_input: bool = False
    """When True the action input is an absolute gripper joint position
    (joint teleop).  When False (default) the input is binary
    open/close (EE / VR teleop)."""

    open_threshold: float = 3.3
    """(absolute_input only) Input ≥ this → *open command*."""

    close_threshold: float = 1.0
    """(absolute_input only) Input ≤ this → *close command*.
    Between ``close_threshold`` and ``open_threshold`` → dead-zone."""

    # ── Contact sensor (for CLOSE → GRASP transition) ────────────────

    contact_sensor_name: str | None = None
    """Name of the :class:`ContactSensorCfg` in the scene that tracks this
    gripper's finger links.  If ``None``, the CLOSE→GRASP transition
    never fires and the gripper simply closes fully."""

    force_threshold: float = 5.0
    """Contact force (N) that must be exceeded on **both** fingers
    simultaneously to trigger CLOSE → GRASP.  Lower = more sensitive."""

    left_finger_body_regex: str = ".*_gripper_left_link"
    """Regex matching the **left** finger body in the contact sensor."""

    right_finger_body_regex: str = ".*_gripper_right_link"
    """Regex matching the **right** finger body in the contact sensor."""


# ══════════════════════════════════════════════════════════════════════════════
# Contact-Limited Gripper (legacy, requires ContactSensor)
# ══════════════════════════════════════════════════════════════════════════════


class ContactLimitedGripperAction(ActionTerm):
    """Gripper action with a grasp target when contact force exceeds threshold.

    This action term prevents gripper penetration by monitoring contact forces on the
    gripper finger links using a ContactSensor. When the contact force exceeds the
    threshold, the gripper switches to a configured grasp target position.

    The action follows the same convention as BinaryJointAction:
    - Positive values (or 1): Open gripper
    - Negative values (or 0): Close gripper

    IMPORTANT: You must add a ContactSensor to the scene config with the name specified
    in `contact_sensor_name` that tracks the gripper finger links.
    """

    cfg: "ContactLimitedGripperActionCfg"
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: "ContactLimitedGripperActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # Resolve the joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"ContactLimitedGripperAction resolved joints: {self._joint_names} [{self._joint_ids}]"
        )

        # Create tensors for actions
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # Parse open command
        self._open_command = torch.zeros(self._num_joints, device=self.device)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.open_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(f"Could not resolve all joints. Missing: {set(self._joint_names) - set(name_list)}")
        self._open_command[index_list] = torch.tensor(value_list, device=self.device)

        # Parse close command
        self._close_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.close_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(f"Could not resolve all joints. Missing: {set(self._joint_names) - set(name_list)}")
        self._close_command[index_list] = torch.tensor(value_list, device=self.device)

        # Parse grasp command
        self._grasp_command = torch.zeros_like(self._open_command)
        index_list, name_list, value_list = string_utils.resolve_matching_names_values(
            self.cfg.grasp_command_expr, self._joint_names
        )
        if len(index_list) != self._num_joints:
            raise ValueError(f"Could not resolve all joints. Missing: {set(self._joint_names) - set(name_list)}")
        self._grasp_command[index_list] = torch.tensor(value_list, device=self.device)

        # Track grasp state (when force limit is reached)
        self._is_grasping = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Track previous close command state
        self._prev_is_closing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Contact sensor reference (will be set in first process_actions call)
        self._contact_sensor: ContactSensor | None = None
        self._left_finger_ids: list[int] | None = None
        self._right_finger_ids: list[int] | None = None

    def _get_contact_sensor(self) -> ContactSensor | None:
        """Get the contact sensor from the scene."""
        if self._contact_sensor is None and self.cfg.contact_sensor_name:
            try:
                self._contact_sensor = self._env.scene[self.cfg.contact_sensor_name]
                omni.log.info(f"ContactLimitedGripperAction using contact sensor: {self.cfg.contact_sensor_name}")
            except KeyError:
                omni.log.warn(
                    f"Contact sensor '{self.cfg.contact_sensor_name}' not found in scene. "
                    "Falling back to joint effort based detection."
                )
        return self._contact_sensor

    def _resolve_finger_body_ids(self, sensor: ContactSensor) -> None:
        if self._left_finger_ids is None:
            self._left_finger_ids, _ = sensor.find_bodies(
                self.cfg.left_finger_body_regex, preserve_order=True
            )
        if self._right_finger_ids is None:
            self._right_finger_ids, _ = sensor.find_bodies(
                self.cfg.right_finger_body_regex, preserve_order=True
            )

    def _compute_force_exceeded(self) -> torch.Tensor:
        sensor = self._get_contact_sensor()
        if sensor is not None:
            self._resolve_finger_body_ids(sensor)
            net_forces = sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
            force_mag = torch.norm(net_forces, dim=-1)  # (num_envs, num_bodies)
            if not self._left_finger_ids or not self._right_finger_ids:
                return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            left_force = force_mag[:, self._left_finger_ids].max(dim=1).values
            right_force = force_mag[:, self._right_finger_ids].max(dim=1).values
            return (left_force > self.cfg.force_threshold) & (
                right_force > self.cfg.force_threshold
            )
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        # Determine if closing (negative action = close)
        is_closing = actions.squeeze(-1) < 0

        # Check if transitioning from open to close
        just_started_closing = is_closing & ~self._prev_is_closing

        # Reset grasp state when transitioning to close
        self._is_grasping[just_started_closing] = False

        # Reset grasp state when opening
        is_opening = ~is_closing
        self._is_grasping[is_opening] = False

        # Check contact force threshold (requires both fingers in contact)
        force_exceeded = self._compute_force_exceeded()

        # When closing and force exceeded, switch to grasp target
        should_grasp = is_closing & force_exceeded & ~self._is_grasping
        self._is_grasping[should_grasp] = True

        # Compute target positions
        # Default: open or close command
        target_pos = torch.where(
            is_closing.unsqueeze(-1),
            self._close_command.unsqueeze(0).expand(self.num_envs, -1),
            self._open_command.unsqueeze(0).expand(self.num_envs, -1),
        )

        # Override with grasp target when grasping
        target_pos = torch.where(
            self._is_grasping.unsqueeze(-1),
            self._grasp_command.unsqueeze(0).expand(self.num_envs, -1),
            target_pos,
        )

        self._processed_actions[:] = target_pos
        self._prev_is_closing[:] = is_closing

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions,
            joint_ids=self._joint_ids,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._raw_actions[env_ids] = 0.0
        self._is_grasping[env_ids] = False
        self._prev_is_closing[env_ids] = False


@configclass
class ContactLimitedGripperActionCfg(ActionTermCfg):
    """Configuration for contact-limited gripper action."""

    class_type: type[ActionTerm] = ContactLimitedGripperAction

    joint_names: list[str] = MISSING
    """List of joint names for the gripper."""

    open_command_expr: dict[str, float] = MISSING
    """Joint command for open configuration."""

    close_command_expr: dict[str, float] = MISSING
    """Joint command for close configuration."""

    grasp_command_expr: dict[str, float] = MISSING
    """Joint command for grasp configuration (used on contact)."""

    contact_sensor_name: str | None = None
    """Name of the ContactSensor in the scene that tracks gripper finger contacts.
    If None, falls back to joint effort based detection (less accurate)."""

    force_threshold: float = 5.0
    """Contact force threshold (in N) above which the gripper stops closing and holds position.
    Lower values = more sensitive. Typical values: 1.0 - 10.0 N."""

    left_finger_body_regex: str = ".*_gripper_left_link"
    """Regex for left finger body names in the contact sensor."""

    right_finger_body_regex: str = ".*_gripper_right_link"
    """Regex for right finger body names in the contact sensor."""
