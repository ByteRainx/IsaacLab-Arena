# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

# Max gripper joint value on the real robot (robocontrol config: [-0.1, 4.5])
_GRIPPER_MAX = 4.5

_body_names_printed = False


def _print_body_names_once(robot: Articulation) -> None:
    """Print articulation body names once for diagnostic purposes."""
    global _body_names_printed
    if not _body_names_printed:
        _body_names_printed = True
        print(f"[ex001arm observations] body_names ({len(robot.data.body_names)}): {robot.data.body_names}")


def _ee_pose_in_root_frame(
    env,
    ee_frame_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute EE position and quaternion (wxyz) in the articulation root frame.

    Uses the FrameTransformer output (which includes any configured offset) and
    converts from world frame to the articulation root frame.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    _print_body_names_once(robot)

    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # T_ee_in_root = inv(T_root_world) * T_ee_world
    ee_pos_root, ee_quat_root = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w,
    )
    return ee_pos_root, ee_quat_root


# ---------------------------------------------------------------------------
# Original observations (world frame, kept for backward compatibility)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Base-frame EE observations (arm root frame, matches real robot FK output)
# ---------------------------------------------------------------------------

def ex001arm_left_eef_pos_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Left arm EE position in the articulation root frame (matches real robot FK)."""
    pos, _ = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    return pos


def ex001arm_left_eef_quat_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Left arm EE quaternion in xyzw order in the articulation root frame.

    Isaac Lab internally uses wxyz. This function converts to xyzw to match
    the ROS / scipy / real-robot convention.
    """
    _, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    # wxyz -> xyzw
    return quat_wxyz[:, [1, 2, 3, 0]]


def ex001arm_right_eef_pos_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Right arm EE position in the articulation root frame."""
    pos, _ = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    return pos


def ex001arm_right_eef_quat_base(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Right arm EE quaternion in xyzw order in the articulation root frame."""
    _, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    return quat_wxyz[:, [1, 2, 3, 0]]


# ---------------------------------------------------------------------------
# Episode-relative EE observations (origin at episode start, initial ~= zero)
# ---------------------------------------------------------------------------

def _get_episode_eef_origin(
    env,
    origin_pos_attr: str,
    origin_quat_attr: str,
    curr_pos: torch.Tensor,
    curr_quat_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (or lazily initialize) per-env EE origin at episode start."""
    if not hasattr(env, origin_pos_attr) or not hasattr(env, origin_quat_attr):
        setattr(env, origin_pos_attr, curr_pos.clone())
        setattr(env, origin_quat_attr, curr_quat_wxyz.clone())

    origin_pos: torch.Tensor = getattr(env, origin_pos_attr)
    origin_quat: torch.Tensor = getattr(env, origin_quat_attr)

    # Refresh origin on reset frames (episode_length_buf == 0).
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        if torch.any(reset_mask):
            origin_pos[reset_mask] = curr_pos[reset_mask]
            origin_quat[reset_mask] = curr_quat_wxyz[reset_mask]

    return origin_pos, origin_quat


def ex001arm_left_eef_pos_rel0(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Left EE position in root frame relative to episode-start EE origin."""
    pos, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    origin_pos, _ = _get_episode_eef_origin(
        env,
        "_ex001arm_left_eef_origin_pos",
        "_ex001arm_left_eef_origin_quat_wxyz",
        pos,
        quat_wxyz,
    )
    return pos - origin_pos


def ex001arm_left_eef_quat_rel0(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Left EE quaternion relative to episode-start origin, in xyzw."""
    pos, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    _, origin_quat_wxyz = _get_episode_eef_origin(
        env,
        "_ex001arm_left_eef_origin_pos",
        "_ex001arm_left_eef_origin_quat_wxyz",
        pos,
        quat_wxyz,
    )
    rel_quat_wxyz = math_utils.quat_mul(quat_wxyz, math_utils.quat_inv(origin_quat_wxyz))
    return rel_quat_wxyz[:, [1, 2, 3, 0]]


def ex001arm_right_eef_pos_rel0(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Right EE position in root frame relative to episode-start EE origin."""
    pos, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    origin_pos, _ = _get_episode_eef_origin(
        env,
        "_ex001arm_right_eef_origin_pos",
        "_ex001arm_right_eef_origin_quat_wxyz",
        pos,
        quat_wxyz,
    )
    return pos - origin_pos


def ex001arm_right_eef_quat_rel0(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("right_ee_frame"),
) -> torch.Tensor:
    """Right EE quaternion relative to episode-start origin, in xyzw."""
    pos, quat_wxyz = _ee_pose_in_root_frame(env, ee_frame_cfg, asset_cfg)
    _, origin_quat_wxyz = _get_episode_eef_origin(
        env,
        "_ex001arm_right_eef_origin_pos",
        "_ex001arm_right_eef_origin_quat_wxyz",
        pos,
        quat_wxyz,
    )
    rel_quat_wxyz = math_utils.quat_mul(quat_wxyz, math_utils.quat_inv(origin_quat_wxyz))
    return rel_quat_wxyz[:, [1, 2, 3, 0]]


# ---------------------------------------------------------------------------
# Normalized gripper observations [0, 1] (matches real robot training data)
# ---------------------------------------------------------------------------

def ex001arm_left_gripper_normalized(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Left gripper joint value normalized to [0, 1]. 0=closed, 1=open."""
    robot: Articulation = env.scene[asset_cfg.name]
    gripper_ids, _ = robot.find_joints(["left_arm_gripper"])
    raw = robot.data.joint_pos[:, gripper_ids]
    return (raw / _GRIPPER_MAX).clamp(0.0, 1.0)


def ex001arm_right_gripper_normalized(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Right gripper joint value normalized to [0, 1]. 0=closed, 1=open."""
    robot: Articulation = env.scene[asset_cfg.name]
    gripper_ids, _ = robot.find_joints(["right_arm_gripper"])
    raw = robot.data.joint_pos[:, gripper_ids]
    return (raw / _GRIPPER_MAX).clamp(0.0, 1.0)
