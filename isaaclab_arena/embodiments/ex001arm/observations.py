# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def openarm_left_gripper_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    all_joint_pos = robot.data.joint_pos
    
    # 获取左右手指关节
    left_finger_ids, _ = robot.find_joints(["left_arm_gripper_left_joint"])
    right_finger_ids, _ = robot.find_joints(["left_arm_gripper_right_joint"])
    
    # 计算夹爪开合度
    gripper_open = all_joint_pos[:, left_finger_ids[0]] - all_joint_pos[:, right_finger_ids[0]]
    
    return gripper_open.unsqueeze(-1)

