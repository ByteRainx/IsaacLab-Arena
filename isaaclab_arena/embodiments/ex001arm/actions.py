# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Custom gripper action with contact force limiting for EX001Arm."""

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
