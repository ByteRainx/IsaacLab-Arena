# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import MISSING
from pathlib import Path
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.pose import Pose


def _default_ex001arm_usd_path() -> str:
    """
    Resolve the ex001Arm USD path.
    """
    arena_root = Path(__file__).resolve().parents[3]
    cvpr_assets_path = arena_root / "cvpr_assets" / "ex001_arm.usd"
    return cvpr_assets_path.as_posix()


def ex001arm_left_gripper_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """ex001Arm left gripper position observation using mimic joints.
    
    Returns 1-dim observation: gripper opening width
    """
    from isaaclab.assets import Articulation
    import torch
    
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 获取两个 mimic 手指关节
    left_finger_ids, _ = robot.find_joints(["left_arm_gripper_left_joint"])
    right_finger_ids, _ = robot.find_joints(["left_arm_gripper_right_joint"])
    
    left_finger_pos = robot.data.joint_pos[:, left_finger_ids]
    right_finger_pos = robot.data.joint_pos[:, right_finger_ids]
    
    # 夹爪开合度
    gripper_opening = left_finger_pos - right_finger_pos
    
    return gripper_opening


@register_asset
class EX001ArmEmbodiment(EmbodimentBase):
    """Embodiment for the EX001Arm bimanual robot (left arm controllable by default)."""

    name = "ex001arm"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        usd_path: str | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = EX001ArmSceneCfg()
        resolved_usd_path = (
            usd_path
            or os.environ.get("ISAACLAB_ARENA_EX001ARM_USD_PATH")
            or _default_ex001arm_usd_path()
        )
        self.scene_config.robot = _make_ex001arm_articulation_cfg(usd_path=resolved_usd_path).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.action_config = EX001ArmActionsCfg()
        self.observation_config = EX001ArmObservationsCfg()

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        return super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)


@configclass
class EX001ArmSceneCfg:
    """Additions to the scene configuration coming from the EX001Arm embodiment."""

    robot: ArticulationCfg = MISSING

    # 左臂末端执行器 FrameTransformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_link6",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_left_link",
                name="tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_right_link",
                name="tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )

    # 右臂末端执行器 FrameTransformer
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_link6",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
                name="right_end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_left_link",
                name="right_tool_leftfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_right_link",
                name="right_tool_rightfinger",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            ),
        ],
    )


def _make_ex001arm_articulation_cfg(usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.5, 0.0, -0.4),
        ),
        actuators={
            # 左臂执行器 (6个关节)
            "left_arm_acts": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[1-6]"],
                stiffness=1500.0,
                damping=150.0,
            ),
            # 右臂执行器 (6个关节)
            "right_arm_acts": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_joint[1-6]"],
                stiffness=1500.0,
                damping=150.0,
            ),
            # 左夹爪执行器
            "left_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_gripper"],
                effort_limit_sim=10.0,
                stiffness=0.4,
                damping=2.0,
            ),
            # 右夹爪执行器
            "right_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_gripper"],
                effort_limit_sim=10.0,
                stiffness=0.4,
                damping=2.0,
            ),
        },
    )


@configclass
class EX001ArmActionsCfg:
    """Action specifications for the MDP (left arm + left gripper)."""

    # 左臂 IK 动作
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint[1-6]"],
        body_name="left_arm_gripper_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )

    # 左夹爪二值动作
    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_arm_gripper"],
        open_command_expr={"left_arm_gripper": 5.0},
        close_command_expr={"left_arm_gripper": 0.0},
    )


@configclass
class EX001ArmObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(func=ex001arm_left_gripper_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
