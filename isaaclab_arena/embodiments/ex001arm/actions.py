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
    """Gripper action that stops closing when contact force on finger links exceeds threshold.

    This action term prevents gripper penetration by monitoring contact forces on the
    gripper finger links using a ContactSensor. When the contact force exceeds the
    threshold, the gripper stops closing and holds its current position.

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

        # Track held positions (when force limit is reached)
        self._held_positions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._is_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Track previous close command state
        self._prev_is_closing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Contact sensor reference (will be set in first process_actions call)
        self._contact_sensor: ContactSensor | None = None

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

    def _get_contact_force(self) -> torch.Tensor:
        """Get the contact force magnitude for each environment.

        Returns:
            Tensor of shape (num_envs,) with contact force magnitude.
        """
        sensor = self._get_contact_sensor()

        if sensor is not None:
            # Use contact sensor data
            # net_forces_w shape: (num_envs, num_bodies, 3)
            net_forces = sensor.data.net_forces_w
            # Sum forces across all bodies and compute magnitude
            total_force = net_forces.sum(dim=1)  # (num_envs, 3)
            force_magnitude = torch.norm(total_force, dim=-1)  # (num_envs,)
            return force_magnitude
        else:
            # Fallback: use joint effort (less accurate)
            current_effort = self._asset.data.applied_torque[:, self._joint_ids]
            return torch.abs(current_effort).sum(dim=-1)

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

        # Reset holding state when transitioning to close
        self._is_holding[just_started_closing] = False

        # Reset holding state when opening
        is_opening = ~is_closing
        self._is_holding[is_opening] = False

        # Get current joint positions
        current_pos = self._asset.data.joint_pos[:, self._joint_ids]

        # Check contact force threshold
        contact_force = self._get_contact_force()
        force_exceeded = contact_force > self.cfg.force_threshold

        # When closing and force exceeded, start holding
        should_hold = is_closing & force_exceeded & ~self._is_holding
        self._held_positions[should_hold] = current_pos[should_hold]
        self._is_holding[should_hold] = True

        # Compute target positions
        # Default: open or close command
        target_pos = torch.where(
            is_closing.unsqueeze(-1),
            self._close_command.unsqueeze(0).expand(self.num_envs, -1),
            self._open_command.unsqueeze(0).expand(self.num_envs, -1),
        )

        # Override with held position when holding
        target_pos = torch.where(
            self._is_holding.unsqueeze(-1),
            self._held_positions,
            target_pos,
        )

        self._processed_actions[:] = target_pos
        self._prev_is_closing[:] = is_closing

    def apply_actions(self) -> None:
        # For environments that are holding, directly write joint position for maximum stability
        if self._is_holding.any():
            holding_envs = self._is_holding.nonzero(as_tuple=True)[0]
            # Directly set joint position (not target) for holding envs - this locks the joint
            self._asset.write_joint_state_to_sim(
                position=self._held_positions[holding_envs],
                velocity=torch.zeros_like(self._held_positions[holding_envs]),
                joint_ids=self._joint_ids,
                env_ids=holding_envs,
            )

        # For non-holding environments, use normal position target
        non_holding_envs = (~self._is_holding).nonzero(as_tuple=True)[0]
        if len(non_holding_envs) > 0:
            self._asset.set_joint_position_target(
                self._processed_actions[non_holding_envs],
                joint_ids=self._joint_ids,
                env_ids=non_holding_envs,
            )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._raw_actions[env_ids] = 0.0
        self._is_holding[env_ids] = False
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

    contact_sensor_name: str | None = None
    """Name of the ContactSensor in the scene that tracks gripper finger contacts.
    If None, falls back to joint effort based detection (less accurate)."""

    force_threshold: float = 5.0
    """Contact force threshold (in N) above which the gripper stops closing and holds position.
    Lower values = more sensitive. Typical values: 1.0 - 10.0 N."""
