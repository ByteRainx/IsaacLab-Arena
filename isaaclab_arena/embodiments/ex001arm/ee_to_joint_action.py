# Copyright (c) 2025, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
#
# TEST-ONLY: EE input → IK solve → joint position control.
# Delete this file after testing. Do NOT commit.

"""EE-to-Joint action: policy outputs EE pose, we solve IK and apply joint position control."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EEToJointAction(ActionTerm):
    """Single action term: receives EE pose (x,y,z,rx,ry,rz,gripper) per arm,
    solves IK to get joint positions, then applies joint position control.

    Action format (14 dims): [left_pose_6, left_gripper_1, right_pose_6, right_gripper_1]
    Pose is in arm base (root) frame. Euler angles in xyz convention (rad).
    Gripper: [0,1] where 0=closed, 1=open.
    """

    cfg: "EEToJointActionCfg"
    _asset: Articulation
    _left_joint_ids: list | slice
    _right_joint_ids: list | slice
    _left_gripper_ids: list
    _right_gripper_ids: list
    _left_body_idx: int
    _right_body_idx: int
    _left_ik: DifferentialIKController
    _right_ik: DifferentialIKController
    _left_jacobi_body_idx: int
    _right_jacobi_body_idx: int
    _left_jacobi_joint_ids: list
    _right_jacobi_joint_ids: list
    _joint_pos_target: torch.Tensor

    def __init__(self, cfg: "EEToJointActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Resolve joints
        left_arm_ids, _ = self._asset.find_joints(cfg.left_arm_joint_names)
        right_arm_ids, _ = self._asset.find_joints(cfg.right_arm_joint_names)
        left_gripper_ids, _ = self._asset.find_joints(cfg.left_gripper_joint_names)
        right_gripper_ids, _ = self._asset.find_joints(cfg.right_gripper_joint_names)

        self._left_joint_ids = left_arm_ids if len(left_arm_ids) < self._asset.num_joints else slice(None)
        self._right_joint_ids = right_arm_ids if len(right_arm_ids) < self._asset.num_joints else slice(None)
        self._left_gripper_ids = left_gripper_ids
        self._right_gripper_ids = right_gripper_ids

        # Body indices for EE
        left_body_ids, _ = self._asset.find_bodies([cfg.left_ee_body_name])
        right_body_ids, _ = self._asset.find_bodies([cfg.right_ee_body_name])
        self._left_body_idx = left_body_ids[0]
        self._right_body_idx = right_body_ids[0]

        if self._asset.is_fixed_base:
            self._left_jacobi_body_idx = self._left_body_idx - 1
            self._right_jacobi_body_idx = self._right_body_idx - 1
            self._left_jacobi_joint_ids = left_arm_ids
            self._right_jacobi_joint_ids = right_arm_ids
        else:
            self._left_jacobi_body_idx = self._left_body_idx
            self._right_jacobi_body_idx = self._right_body_idx
            self._left_jacobi_joint_ids = [i + 6 for i in left_arm_ids]
            self._right_jacobi_joint_ids = [i + 6 for i in right_arm_ids]

        self._left_ik = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )
        self._right_ik = DifferentialIKController(
            cfg=self.cfg.ik_controller_cfg, num_envs=self.num_envs, device=self.device
        )

        num_joints = len(left_arm_ids) + len(left_gripper_ids) + len(right_arm_ids) + len(right_gripper_ids)
        self._joint_pos_target = torch.zeros(self.num_envs, num_joints, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 14, device=self.device)

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._joint_pos_target

    def _get_jacobian_b(self, body_idx: int, joint_ids: list | slice) -> torch.Tensor:
        jac_w = self._asset.root_physx_view.get_jacobians()[:, body_idx, :, joint_ids]
        base_rot = self._asset.data.root_quat_w
        base_rot_inv = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jac_b = jac_w.clone()
        jac_b[:, :3, :] = torch.bmm(base_rot_inv, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot_inv, jac_w[:, 3:, :])
        return jac_b

    def _get_ee_pose_root_frame(self, body_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        return ee_pos_b, ee_quat_b

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        # Left arm: action[0:6] = (x,y,z,rx,ry,rz), action[6] = gripper
        left_pos = actions[:, 0:3]
        left_euler = actions[:, 3:6]
        left_gripper = actions[:, 6]

        # Right arm: action[7:13], action[13]
        right_pos = actions[:, 7:10]
        right_euler = actions[:, 10:13]
        right_gripper = actions[:, 13]

        # Euler (xyz) -> quat (wxyz)
        left_quat = math_utils.quat_from_euler_xyz(left_euler[:, 0], left_euler[:, 1], left_euler[:, 2])
        right_quat = math_utils.quat_from_euler_xyz(right_euler[:, 0], right_euler[:, 1], right_euler[:, 2])

        # Set IK commands (absolute pose)
        left_cmd = torch.cat([left_pos, left_quat], dim=-1)
        right_cmd = torch.cat([right_pos, right_quat], dim=-1)
        self._left_ik.set_command(left_cmd)
        self._right_ik.set_command(right_cmd)

        # Solve IK
        left_ee_pos, left_ee_quat = self._get_ee_pose_root_frame(self._left_body_idx)
        right_ee_pos, right_ee_quat = self._get_ee_pose_root_frame(self._right_body_idx)

        left_jac = self._get_jacobian_b(self._left_jacobi_body_idx, self._left_jacobi_joint_ids)
        right_jac = self._get_jacobian_b(self._right_jacobi_body_idx, self._right_jacobi_joint_ids)

        left_joint_pos = self._asset.data.joint_pos[:, self._left_joint_ids]
        right_joint_pos = self._asset.data.joint_pos[:, self._right_joint_ids]

        left_joint_des = self._left_ik.compute(left_ee_pos, left_ee_quat, left_jac, left_joint_pos)
        right_joint_des = self._right_ik.compute(right_ee_pos, right_ee_quat, right_jac, right_joint_pos)

        # Gripper: [0,1] -> [0, 4.5] (closed to open)
        left_gripper_des = left_gripper.unsqueeze(-1) * self.cfg.gripper_open_value
        right_gripper_des = right_gripper.unsqueeze(-1) * self.cfg.gripper_open_value

        self._joint_pos_target[:, 0:6] = left_joint_des
        self._joint_pos_target[:, 6] = left_gripper_des.squeeze(-1)
        self._joint_pos_target[:, 7:13] = right_joint_des
        self._joint_pos_target[:, 13] = right_gripper_des.squeeze(-1)

    def apply_actions(self) -> None:
        # Build full joint target: left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)
        all_joint_ids = list(self._left_joint_ids) + self._left_gripper_ids + list(self._right_joint_ids) + self._right_gripper_ids
        self._asset.set_joint_position_target(self._joint_pos_target, joint_ids=all_joint_ids)

    def reset(self, env_ids: torch.Tensor) -> None:
        self._raw_actions[env_ids] = 0.0


@configclass
class EEToJointActionCfg(ActionTermCfg):
    """Config for EE-to-Joint action term (test only)."""

    class_type: type[ActionTerm] = EEToJointAction

    asset_name: str = "robot"
    left_arm_joint_names: list[str] = ["left_arm_joint[1-6]"]
    right_arm_joint_names: list[str] = ["right_arm_joint[1-6]"]
    left_gripper_joint_names: list[str] = ["left_arm_gripper"]
    right_gripper_joint_names: list[str] = ["right_arm_gripper"]
    left_ee_body_name: str = "left_arm_gripper_base_link"
    right_ee_body_name: str = "right_arm_gripper_base_link"
    gripper_open_value: float = 4.5

    ik_controller_cfg: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.01},
    )


@configclass
class EX001ArmEEToJointTestActionsCfg:
    """Actions config with single EE-to-Joint term (test only). Use with --ee_to_joint."""

    ee_to_joint: EEToJointActionCfg = EEToJointActionCfg()
