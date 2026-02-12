# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Put blocks to matching color destinations task."""

import numpy as np
import torch
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.assets import RigidObject
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


def all_objects_on_destinations(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    destination_cfgs: list[SceneEntityCfg],
    x_threshold: float = 0.12,
    y_threshold: float = 0.16,
    z_threshold: float = 0.06,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Check if all objects are placed on their corresponding destinations using proximity.

    Args:
        env: The environment instance.
        object_cfgs: List of scene entity configs for each object (blocks).
        destination_cfgs: List of scene entity configs for each destination (papers).
        x_threshold: Maximum X distance to consider object placed.
        y_threshold: Maximum Y distance to consider object placed.
        z_threshold: Maximum Z distance to consider object placed.
        velocity_threshold: Maximum velocity to consider object stationary.

    Returns:
        Boolean tensor indicating success for each environment.
    """
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for object_cfg, dest_cfg in zip(object_cfgs, destination_cfgs):
        obj: RigidObject = env.scene[object_cfg.name]
        dest: RigidObject = env.scene[dest_cfg.name]

        # Get positions relative to environment origin
        obj_pos = obj.data.root_pos_w - env.scene.env_origins
        dest_pos = dest.data.root_pos_w - env.scene.env_origins

        # Check X distance
        x_distance = torch.abs(obj_pos[:, 0] - dest_pos[:, 0])
        x_close = x_distance < x_threshold

        # Check Y distance
        y_distance = torch.abs(obj_pos[:, 1] - dest_pos[:, 1])
        y_close = y_distance < y_threshold

        # Check Z distance (block should be on top or very close)
        z_distance = torch.abs(obj_pos[:, 2] - dest_pos[:, 2])
        z_close = z_distance < z_threshold

        # Check velocity
        velocity_w = obj.data.root_lin_vel_w
        velocity_w_norm = torch.norm(velocity_w, dim=-1)
        velocity_below_threshold = velocity_w_norm < velocity_threshold

        # Object is placed if all conditions met
        object_placed = torch.logical_and(x_close, y_close)
        object_placed = torch.logical_and(object_placed, z_close)
        object_placed = torch.logical_and(object_placed, velocity_below_threshold)
        all_placed = torch.logical_and(all_placed, object_placed)

    return all_placed


def any_object_dropped(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    minimum_height: float,
) -> torch.Tensor:
    """Check if any object has been dropped below minimum height.

    Args:
        env: The environment instance.
        object_cfgs: List of scene entity configs for each object.
        minimum_height: Minimum height threshold.

    Returns:
        Boolean tensor indicating if any object dropped for each environment.
    """
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        object_pos = obj.data.root_pos_w - env.scene.env_origins
        dropped = object_pos[:, 2] < minimum_height
        any_dropped = torch.logical_or(any_dropped, dropped)

    return any_dropped


class PutBlocksToColorTask(TaskBase):
    """Task for placing multiple blocks to their matching color destinations.

    This task involves picking up multiple colored blocks and placing them
    on their corresponding colored destinations.
    """

    def __init__(
        self,
        blocks: list[ObjectBase],
        destinations: list[ObjectBase],
        background_scene: Asset,
        episode_length_s: float | None = None,
    ):
        """Initialize the put blocks to color task.

        Args:
            blocks: List of block assets (should be 3 blocks).
            destinations: List of destination assets (should be 3 destinations).
            background_scene: Background scene asset.
            episode_length_s: Episode length in seconds.
        """
        super().__init__(episode_length_s=episode_length_s)

        if len(blocks) != len(destinations):
            raise ValueError(
                f"Number of blocks ({len(blocks)}) must match "
                f"number of destinations ({len(destinations)})"
            )

        self.blocks = blocks
        self.destinations = destinations
        self.background_scene = background_scene
        self.num_blocks = len(blocks)

        # Create scene config with contact sensors for each block
        self.scene_config = self._create_scene_cfg()
        self.events_cfg = self._create_events_cfg()
        self.termination_cfg = self._create_termination_cfg()

    def _create_scene_cfg(self) -> Any:
        """Create scene configuration for multi-block task.

        Note: We use proximity-based detection instead of contact sensors,
        so no additional scene config is needed.
        """
        return None

    def _create_events_cfg(self) -> Any:
        """Create events configuration for resetting all blocks."""
        # Build fields list for make_configclass
        fields = []
        for i, block in enumerate(self.blocks):
            initial_pose = block.get_initial_pose()
            if initial_pose is not None:
                event_cfg = EventTermCfg(
                    func=set_object_pose,
                    mode="reset",
                    params={
                        "pose": initial_pose,
                        "asset_cfg": SceneEntityCfg(block.name),
                    },
                )
                fields.append((f"reset_block_{i}_pose", EventTermCfg, event_cfg))
            else:
                print(
                    f"Block {block.name} has no initial pose. "
                    f"Not setting reset pose event."
                )

        # Create configclass with proper fields
        EventsCfgClass = make_configclass("MultiBlockEventsCfg", fields)
        events_cfg = EventsCfgClass()
        return events_cfg

    def _create_termination_cfg(self) -> "MultiBlockTerminationsCfg":
        """Create termination configuration for success and failure conditions."""
        # Create object configs for blocks and destinations
        object_cfgs = [SceneEntityCfg(block.name) for block in self.blocks]
        destination_cfgs = [SceneEntityCfg(dest.name) for dest in self.destinations]

        success = TerminationTermCfg(
            func=all_objects_on_destinations,
            params={
                "object_cfgs": object_cfgs,
                "destination_cfgs": destination_cfgs,
                "x_threshold": 0.08,
                "y_threshold": 0.12,
                "z_threshold": 0.04,
                "velocity_threshold": 0.5,
            },
        )

        any_block_dropped = TerminationTermCfg(
            func=any_object_dropped,
            params={
                "object_cfgs": object_cfgs,
                "minimum_height": self.background_scene.object_min_z,
            },
        )

        terminations_cfg = MultiBlockTerminationsCfg()
        terminations_cfg.success = success
        terminations_cfg.any_block_dropped = any_block_dropped
        return terminations_cfg

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self) -> str:
        return "Pick up the colored blocks and place them on their matching colored destinations."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return PutBlocksToColorMimicEnvCfg(
            embodiment_name=embodiment_name,
            block_names=[block.name for block in self.blocks],
            destination_names=[dest.name for dest in self.destinations],
        )

    def get_metrics(self) -> list[MetricBase]:
        # NOTE: Only track the first block's movement to avoid duplicate field names.
        # ObjectMovedRateMetric uses a hardcoded name, so we can only have one instance.
        metrics: list[MetricBase] = [
            SuccessRateMetric(),
            ObjectMovedRateMetric(self.blocks[0]),
        ]
        return metrics

    def get_viewer_cfg(self) -> ViewerCfg:
        # Look at the first block by default
        return get_viewer_cfg_look_at_object(
            lookat_object=self.blocks[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


class ThreeBlocksToColorTask(PutBlocksToColorTask):
    """Convenience class for exactly 3 blocks and 3 destinations.

    This is the standard configuration for the put-blocks-to-color task.
    """

    def __init__(
        self,
        block_1: ObjectBase,
        block_2: ObjectBase,
        block_3: ObjectBase,
        destination_1: ObjectBase,
        destination_2: ObjectBase,
        destination_3: ObjectBase,
        background_scene: Asset,
        episode_length_s: float | None = None,
    ):
        """Initialize with exactly 3 blocks and 3 destinations.

        Args:
            block_1: First block asset.
            block_2: Second block asset.
            block_3: Third block asset.
            destination_1: First destination asset (for block_1).
            destination_2: Second destination asset (for block_2).
            destination_3: Third destination asset (for block_3).
            background_scene: Background scene asset.
            episode_length_s: Episode length in seconds.
        """
        super().__init__(
            blocks=[block_1, block_2, block_3],
            destinations=[destination_1, destination_2, destination_3],
            background_scene=background_scene,
            episode_length_s=episode_length_s,
        )


@configclass
class MultiBlockTerminationsCfg:
    """Termination terms for multi-block tasks."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    any_block_dropped: TerminationTermCfg = MISSING


@configclass
class PutBlocksToColorMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic environment config for put blocks to color task."""

    embodiment_name: str = "franka"
    block_names: list[str] | None = None
    destination_names: list[str] | None = None

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_put_blocks_to_color_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        # Ensure block_names and destination_names are not None
        if self.block_names is None:
            self.block_names = []
        if self.destination_names is None:
            self.destination_names = []

        # Create subtask configs for each block
        subtask_configs = []

        for i, (block_name, dest_name) in enumerate(
            zip(self.block_names, self.destination_names)
        ):
            # Pick subtask for this block
            subtask_configs.append(
                SubTaskConfig(
                    object_ref=block_name,
                    subtask_term_signal=f"grasp_{i + 1}",
                    subtask_term_offset_range=(10, 20),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            )
            # Place subtask for this block (except for the last one)
            if i < len(self.block_names) - 1:
                subtask_configs.append(
                    SubTaskConfig(
                        object_ref=dest_name,
                        subtask_term_signal=f"place_{i + 1}",
                        subtask_term_offset_range=(5, 10),
                        selection_strategy="nearest_neighbor_object",
                        selection_strategy_kwargs={"nn_k": 3},
                        action_noise=0.005,
                        num_interpolation_steps=5,
                        num_fixed_steps=0,
                        apply_noise_during_interpolation=False,
                    )
                )

        # Final place subtask (no term signal needed)
        if self.destination_names:
            subtask_configs.append(
                SubTaskConfig(
                    object_ref=self.destination_names[-1],
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=5,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            )

        if self.embodiment_name == "franka":
            self.subtask_configs["robot"] = subtask_configs
        elif self.embodiment_name == "gr1_pink":
            self.subtask_configs["right"] = subtask_configs
            # Static left arm config
            left_subtask_configs = [
                SubTaskConfig(
                    object_ref=self.block_names[0] if self.block_names else "block_0",
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ]
            self.subtask_configs["left"] = left_subtask_configs
        elif self.embodiment_name == "ex001arm":
            # EX001Arm bimanual: right arm handles primary manipulation subtasks
            self.subtask_configs["right"] = subtask_configs
            # Left arm stays static (holds position)
            left_subtask_configs = [
                SubTaskConfig(
                    object_ref=self.block_names[0] if self.block_names else "block_0",
                    subtask_term_signal=None,
                    subtask_term_offset_range=(0, 0),
                    selection_strategy="nearest_neighbor_object",
                    selection_strategy_kwargs={"nn_k": 3},
                    action_noise=0.005,
                    num_interpolation_steps=0,
                    num_fixed_steps=0,
                    apply_noise_during_interpolation=False,
                )
            ]
            self.subtask_configs["left"] = left_subtask_configs
        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
