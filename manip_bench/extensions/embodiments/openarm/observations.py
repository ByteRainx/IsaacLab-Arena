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
    gripper_joint_ids, _ = robot.find_joints(["openarm_left_finger_joint.*"])
    gripper_open = all_joint_pos[:, gripper_joint_ids[0]] - all_joint_pos[:, gripper_joint_ids[1]]
    return gripper_open.unsqueeze(-1)

