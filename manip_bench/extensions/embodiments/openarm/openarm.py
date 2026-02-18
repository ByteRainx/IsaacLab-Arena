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

from manip_bench.extensions.embodiments.openarm.observations import openarm_left_gripper_pos
from manip_bench.utils.paths import get_robot_usd_path


def _openarm_usd_path() -> str:
    """Resolve the OpenArm USD path from manip-bench assets."""
    return get_robot_usd_path("openarm_bimanual/openarm_bimanual.usd")


@register_asset
class OpenArmEmbodiment(EmbodimentBase):
    """Embodiment for the OpenArm bimanual robot (left arm controllable by default)."""

    name = "openarm"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        usd_path: str | None = None,
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = OpenArmSceneCfg()
        resolved_usd_path = (
            usd_path
            or os.environ.get("ISAACLAB_ARENA_OPENARM_USD_PATH")
            or os.environ.get("OPENARM_USD_PATH")
            or _openarm_usd_path()
        )
        self.scene_config.robot = (
            _make_openarm_articulation_cfg(usd_path=resolved_usd_path)
            .replace(prim_path="{ENV_REGEX_NS}/Robot")
        )
        self.action_config = OpenArmActionsCfg()
        self.observation_config = OpenArmObservationsCfg()

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        return super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)


@configclass
class OpenArmSceneCfg:
    """Additions to the scene configuration coming from the OpenArm embodiment."""

    robot: ArticulationCfg = MISSING

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_left_link7",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_left_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_right_link7",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_right_hand",
                name="right_end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )


def _make_openarm_articulation_cfg(usd_path: str) -> ArticulationCfg:
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
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.55, 0.0, -0.35),
        ),
        actuators={
            "left_arm_acts_1_4": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-4]"],
                stiffness=1500.0,
                damping=150.0,
            ),
            "left_arm_acts_5_7": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[5-7]"],
                stiffness=1000.0,
                damping=100.0,
            ),
            "right_arm_acts_1_4": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-4]"],
                stiffness=1500.0,
                damping=150.0,
            ),
            "right_arm_acts_5_7": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[5-7]"],
                stiffness=1000.0,
                damping=100.0,
            ),
            "left_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint.*"],
                stiffness=1000.0,
                damping=100.0,
            ),
            "right_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_finger_joint.*"],
                stiffness=1000.0,
                damping=100.0,
            ),
        },
    )


@configclass
class OpenArmActionsCfg:
    """Action specifications for the MDP (left arm + left gripper)."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_joint.*"],
        body_name="openarm_left_hand",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_finger_joint.*"],
        open_command_expr={"openarm_left_finger_joint.*": 1.0},
        close_command_expr={"openarm_left_finger_joint.*": 0.0},
    )


@configclass
class OpenArmObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(
            func=mdp_isaac_lab.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp_isaac_lab.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        eef_pos = ObsTerm(func=ee_frame_pos)
        eef_quat = ObsTerm(func=ee_frame_quat)
        gripper_pos = ObsTerm(
            func=openarm_left_gripper_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
