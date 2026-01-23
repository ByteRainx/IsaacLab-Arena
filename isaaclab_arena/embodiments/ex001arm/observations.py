# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def ex001arm_left_gripper_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Left gripper opening width from mimic finger joints."""
    robot: Articulation = env.scene[asset_cfg.name]
    left_finger_ids, _ = robot.find_joints(["left_arm_gripper_left_joint"])
    right_finger_ids, _ = robot.find_joints(["left_arm_gripper_right_joint"])
    left_finger_pos = robot.data.joint_pos[:, left_finger_ids]
    right_finger_pos = robot.data.joint_pos[:, right_finger_ids]
    return left_finger_pos - right_finger_pos


def ex001arm_right_gripper_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Right gripper opening width from mimic joints, with a single-joint fallback."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_names = set(robot.data.joint_names)
    mimic_names = ("right_arm_gripper_left_joint", "right_arm_gripper_right_joint")
    if all(name in joint_names for name in mimic_names):
        left_finger_ids, _ = robot.find_joints([mimic_names[0]])
        right_finger_ids, _ = robot.find_joints([mimic_names[1]])
        left_finger_pos = robot.data.joint_pos[:, left_finger_ids]
        right_finger_pos = robot.data.joint_pos[:, right_finger_ids]
        return left_finger_pos - right_finger_pos
    gripper_ids, _ = robot.find_joints(["right_arm_gripper"])
    return robot.data.joint_pos[:, gripper_ids]
