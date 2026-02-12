# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
from dataclasses import MISSING
from pathlib import Path
from typing import Any, Callable

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.stack.mdp.observations import ee_frame_pos, ee_frame_quat

import isaaclab.sim as sim_utils

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.ex001arm.actions import ContactLimitedGripperActionCfg
from isaaclab_arena.embodiments.ex001arm.observations import ex001arm_left_gripper_pos, ex001arm_right_gripper_pos
from isaaclab_arena.utils.pose import Pose


def _ex001arm_usd_path() -> str:
    """
    Resolve the ex001Arm USD path.
    """
    arena_root = Path(__file__).resolve().parents[3]
    cvpr_assets_path = arena_root / "assets" / "ex001arm_bimanual" / "ex001_arm.usd"
    return cvpr_assets_path.as_posix()


_FLOAT_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_SPLIT_PATTERN = re.compile(r"[,\s]+")


def _parse_env_tuple(env_value: str | None, expected_len: int) -> tuple[float, ...] | None:
    """Parse a comma/space separated list of floats from an env var."""
    if not env_value:
        return None
    parts = [p for p in _SPLIT_PATTERN.split(env_value.strip()) if p]
    if len(parts) != expected_len:
        return None
    if not all(_FLOAT_PATTERN.fullmatch(p) for p in parts):
        return None
    return tuple(float(p) for p in parts)


def update_opencv_fisheye_camera(
    prim_path: str,
    cfg: "OpenCVFisheyeCameraCfg",
    translation=None,
    orientation=None,
):
    """Update fisheye camera clipping range only."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid() and cfg.clipping_range:
        camera = UsdGeom.Camera(prim)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(*cfg.clipping_range))
    return prim


@configclass
class OpenCVFisheyeCameraCfg(PinholeCameraCfg):
    """Configuration for updating an existing OpenCV fisheye camera (clipping range only)."""
    func: Callable = update_opencv_fisheye_camera
    copy_from_source: bool = False
    clipping_range: tuple[float, float] | None = None


@register_asset
class EX001ArmEmbodiment(EmbodimentBase):
    """Embodiment for the EX001Arm bimanual robot (both arms controllable)."""

    name = "ex001arm"

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        usd_path: str | None = None,
        xr_anchor_pos: tuple[float, float, float] | None = None,
        xr_anchor_rot: tuple[float, float, float, float] | None = None,
        action_mode: str = "ee",
    ):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = EX001ArmSceneCfg()
        resolved_usd_path = _ex001arm_usd_path()
        self.scene_config.robot = _make_ex001arm_articulation_cfg(usd_path=resolved_usd_path).replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        if action_mode == "joint":
            self.action_config = EX001ArmJointActionsCfg()
        else:
            self.action_config = EX001ArmActionsCfg()
        self.observation_config = EX001ArmObservationsCfg()
        env_anchor_pos = _parse_env_tuple(os.environ.get("ISAACLAB_ARENA_EX001ARM_XR_ANCHOR_POS"), 3)
        env_anchor_rot = _parse_env_tuple(os.environ.get("ISAACLAB_ARENA_EX001ARM_XR_ANCHOR_ROT"), 4)
        self.xr = XrCfg(
            anchor_pos=xr_anchor_pos or env_anchor_pos or (0.0, 0.0, 0.0),
            anchor_rot=xr_anchor_rot or env_anchor_rot or (1.0, 0.0, 0.0, 0.0),
        )

    def _update_scene_cfg_with_robot_initial_pose(self, scene_config: Any, pose: Pose) -> Any:
        return super()._update_scene_cfg_with_robot_initial_pose(scene_config, pose)


@configclass
class EX001ArmSceneCfg:
    """Additions to the scene configuration coming from the EX001Arm embodiment."""

    robot: ArticulationCfg = MISSING

    # Left-arm end-effector frame transformer
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_base_link",
                name="end_effector",
                offset=OffsetCfg(pos=[0.154, 0.0, 0.0]),
            ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_left_link",
            #     name="tool_leftfinger",
            #     offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            # ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_right_link",
            #     name="tool_rightfinger",
            #     offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            # ),
        ],
    )

    # Right-arm end-effector frame transformer
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_base_link",
                name="right_end_effector",
                offset=OffsetCfg(pos=[0.154, 0.0, 0.0]),
            ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_left_link",
            #     name="right_tool_leftfinger",
            #     offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            # ),
            # FrameTransformerCfg.FrameCfg(
            #     prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_right_link",
            #     name="right_tool_rightfinger",
            #     offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
            # ),
        ],
    )

    # Left wrist camera (fisheye OpenCV)
    left_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_camera_color_frame/Left_Gripper_Camera",
        update_period=0.0333,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=OpenCVFisheyeCameraCfg(
            clipping_range=(0.1, 1.0e5),
        ),
    )

    # Right wrist camera (fisheye OpenCV)
    right_wrist_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_camera_color_frame/Right_Gripper_Camera",
        update_period=0.0333,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=OpenCVFisheyeCameraCfg(
            clipping_range=(0.1, 1.0e5),
        ),
    )

    # Head camera (pinhole)
    head_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Head_Camera",
        update_period=0.0333,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=PinholeCameraCfg(
            focal_length=14.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.31528, 0.0, 0.79201),
            rot=(0.66611, 0.23726, -0.23726, -0.66611),
            convention="opengl",
        ),
    )

    # Left gripper contact sensor (tracks finger link contacts)
    left_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_gripper_.*_link",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )

    # Right gripper contact sensor (tracks finger link contacts)
    right_gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_gripper_.*_link",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )


def _make_ex001arm_articulation_cfg(usd_path: str) -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,  # Enable for gripper contact detection
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=16,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-0.51676, -0.25918, -0.58061),
            rot=(0.04894564807667012, 0.0, 0.0, 0.9988014434983336),
        ),
        actuators={
            # Left arm actuators (6 joints)
            "left_arm_acts": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_joint[1-6]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=8.0,
            ),
            # Right arm actuators (6 joints)
            "right_arm_acts": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_joint[1-6]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=8.0,
            ),
            # Left gripper actuator (high stiffness for snappy response)
            "left_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_gripper"],
                effort_limit_sim=200.0,
                stiffness=200.0,
                damping=8.0,
            ),
            # Right gripper actuator (high stiffness for snappy response)
            "right_gripper_acts": ImplicitActuatorCfg(
                joint_names_expr=["right_arm_gripper"],
                effort_limit_sim=200.0,
                stiffness=200.0,
                damping=8.0,
            ),
        },
    )


@configclass
class EX001ArmActionsCfg:
    """Action specifications for the MDP (bimanual: left+right arms and grippers)."""

    # Left arm IK action
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint[1-6]"],
        body_name="left_arm_gripper_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )

    # Left gripper with contact force limiting
    gripper_action: ActionTermCfg = ContactLimitedGripperActionCfg(
        asset_name="robot",
        joint_names=["left_arm_gripper"],
        open_command_expr={"left_arm_gripper": 5.0},
        close_command_expr={"left_arm_gripper": 0.0},
        grasp_command_expr={"left_arm_gripper": 1.7},
        contact_sensor_name="left_gripper_contact",
        force_threshold=15.0,  # Contact force threshold in N
    )

    # Right arm IK action
    right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint[1-6]"],
        body_name="right_arm_gripper_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )

    # Right gripper with contact force limiting
    right_gripper_action: ActionTermCfg = ContactLimitedGripperActionCfg(
        asset_name="robot",
        joint_names=["right_arm_gripper"],
        open_command_expr={"right_arm_gripper": 5.0},
        close_command_expr={"right_arm_gripper": 0.0},
        grasp_command_expr={"right_arm_gripper": 1.7},
        contact_sensor_name="right_gripper_contact",
        force_threshold=15.0,  # Contact force threshold in N
    )


@configclass
class EX001ArmPhysicalTeleopActionsCfg:
    """Action config for physical arm teleoperation (1:1 scale).

    Same as ``EX001ArmActionsCfg`` but with ``scale=1.0`` on the IK actions,
    so that delta poses from a same-model physical arm map 1:1 to simulation.
    """

    # Left arm IK action -- scale=1.0 for 1:1 physical mapping
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint[1-6]"],
        body_name="left_arm_gripper_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
    )

    # Left gripper with contact force limiting
    gripper_action: ActionTermCfg = ContactLimitedGripperActionCfg(
        asset_name="robot",
        joint_names=["left_arm_gripper"],
        open_command_expr={"left_arm_gripper": 5.0},
        close_command_expr={"left_arm_gripper": 0.0},
        grasp_command_expr={"left_arm_gripper": 1.7},
        contact_sensor_name="left_gripper_contact",
        force_threshold=15.0,
    )

    # Right arm IK action -- scale=1.0 for 1:1 physical mapping
    right_arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint[1-6]"],
        body_name="right_arm_gripper_base_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=1.0,
    )

    # Right gripper with contact force limiting
    right_gripper_action: ActionTermCfg = ContactLimitedGripperActionCfg(
        asset_name="robot",
        joint_names=["right_arm_gripper"],
        open_command_expr={"right_arm_gripper": 5.0},
        close_command_expr={"right_arm_gripper": 0.0},
        grasp_command_expr={"right_arm_gripper": 1.7},
        contact_sensor_name="right_gripper_contact",
        force_threshold=15.0,
    )


@configclass
class EX001ArmJointActionsCfg:
    """Absolute joint position action specifications for ARX teleoperation.

    Used with ``Ex001ArmArxBimanualTeleop`` in **joint** control mode.
    Each arm's 6 joint positions are set directly, and the gripper receives
    an absolute joint position value (``0.0`` = close, ``5.0`` = open).
    """

    # Left arm -- absolute joint position
    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_arm_joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )

    # Left gripper -- absolute joint position (0.0 close → 5.0 open)
    gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_arm_gripper"],
        scale=1.0,
        use_default_offset=False,
    )

    # Right arm -- absolute joint position
    right_arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_arm_joint[1-6]"],
        scale=1.0,
        use_default_offset=False,
    )

    # Right gripper -- absolute joint position (0.0 close → 5.0 open)
    right_gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["right_arm_gripper"],
        scale=1.0,
        use_default_offset=False,
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
        right_eef_pos = ObsTerm(func=ee_frame_pos, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")})
        right_eef_quat = ObsTerm(func=ee_frame_quat, params={"ee_frame_cfg": SceneEntityCfg("right_ee_frame")})
        right_gripper_pos = ObsTerm(func=ex001arm_right_gripper_pos, params={"asset_cfg": SceneEntityCfg("robot")})

        # Left wrist camera observation
        left_wrist_cam = ObsTerm(
            func=mdp_isaac_lab.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False}
        )
        # Right wrist camera observation
        right_wrist_cam = ObsTerm(
            func=mdp_isaac_lab.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False}
        )
        # Head camera observation
        head_cam = ObsTerm(
            func=mdp_isaac_lab.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
