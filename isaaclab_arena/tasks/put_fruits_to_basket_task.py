# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Put selected fruits into a basket-like destination task."""

import numpy as np
import torch
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
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


def all_objects_in_basket(
    env: ManagerBasedRLEnv,
    object_cfgs: list[SceneEntityCfg],
    basket_cfg: SceneEntityCfg,
    x_threshold: float = 0.05,
    y_threshold: float = 0.08,
    z_threshold: float = 0.1,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Check if all target objects are in proximity of the basket object."""
    basket: RigidObject = env.scene[basket_cfg.name]
    basket_pos = basket.data.root_pos_w - env.scene.env_origins
    all_placed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        obj_pos = obj.data.root_pos_w - env.scene.env_origins

        x_close = torch.abs(obj_pos[:, 0] - basket_pos[:, 0]) < x_threshold
        y_close = torch.abs(obj_pos[:, 1] - basket_pos[:, 1]) < y_threshold
        z_close = torch.abs(obj_pos[:, 2] - basket_pos[:, 2]) < z_threshold

        velocity_w_norm = torch.norm(obj.data.root_lin_vel_w, dim=-1)
        velocity_below_threshold = velocity_w_norm < velocity_threshold

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
    """Check if any monitored object is dropped below the minimum height."""
    any_dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for object_cfg in object_cfgs:
        obj: RigidObject = env.scene[object_cfg.name]
        object_pos = obj.data.root_pos_w - env.scene.env_origins
        dropped = object_pos[:, 2] < minimum_height
        any_dropped = torch.logical_or(any_dropped, dropped)
    return any_dropped


class PutFruitsToBasketTask(TaskBase):
    """Task for placing apple, grape and banana into a basket-like destination."""

    def __init__(
        self,
        target_objects: list[ObjectBase],
        all_objects: list[ObjectBase],
        basket: ObjectBase,
        background_scene: Asset,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.target_objects = target_objects
        self.all_objects = all_objects
        self.basket = basket
        self.background_scene = background_scene

        self.scene_config = self._create_scene_cfg()
        self.events_cfg = self._create_events_cfg()
        self.termination_cfg = self._create_termination_cfg()

    def _create_scene_cfg(self) -> Any:
        return None

    def _create_events_cfg(self) -> Any:
        fields = []
        for i, obj in enumerate(self.all_objects):
            initial_pose = obj.get_initial_pose()
            if initial_pose is None:
                continue
            event_cfg = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(obj.name),
                },
            )
            fields.append((f"reset_object_{i}_pose", EventTermCfg, event_cfg))
        EventsCfgClass = make_configclass("PutFruitsEventsCfg", fields)
        return EventsCfgClass()

    def _create_termination_cfg(self) -> "PutFruitsToBasketTerminationsCfg":
        target_cfgs = [SceneEntityCfg(obj.name) for obj in self.target_objects]
        all_cfgs = [SceneEntityCfg(obj.name) for obj in self.all_objects]

        success = TerminationTermCfg(
            func=all_objects_in_basket,
            params={
                "object_cfgs": target_cfgs,
                "basket_cfg": SceneEntityCfg(self.basket.name),
                "x_threshold": 0.10,
                "y_threshold": 0.12,
                "z_threshold": 0.08,
                "velocity_threshold": 0.5,
            },
        )
        any_obj_dropped = TerminationTermCfg(
            func=any_object_dropped,
            params={
                "object_cfgs": all_cfgs,
                "minimum_height": self.background_scene.object_min_z,
            },
        )
        return PutFruitsToBasketTerminationsCfg(success=success, any_object_dropped=any_obj_dropped)

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self) -> str:
        return "Pick up apple, grape, and banana and put them into the pink basket platform."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return PutFruitsToBasketMimicEnvCfg(
            embodiment_name=embodiment_name,
            target_object_names=[obj.name for obj in self.target_objects],
            basket_name=self.basket.name,
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.target_objects[0])]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.target_objects[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class PutFruitsToBasketTerminationsCfg:
    """Termination terms for put-fruits-to-basket task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    any_object_dropped: TerminationTermCfg = MISSING


@configclass
class PutFruitsToBasketMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic config for put-fruits-to-basket task."""

    embodiment_name: str = "franka"
    target_object_names: list[str] | None = None
    basket_name: str = "platform_pink"

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_put_fruits_to_basket_D0"
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

        if self.target_object_names is None:
            self.target_object_names = []

        subtask_configs = []
        for i, object_name in enumerate(self.target_object_names):
            subtask_configs.append(
                SubTaskConfig(
                    object_ref=object_name,
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
            if i < len(self.target_object_names) - 1:
                subtask_configs.append(
                    SubTaskConfig(
                        object_ref=self.basket_name,
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

        if self.target_object_names:
            subtask_configs.append(
                SubTaskConfig(
                    object_ref=self.basket_name,
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
            self.subtask_configs["left"] = [
                SubTaskConfig(
                    object_ref=self.target_object_names[0] if self.target_object_names else "apple",
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
        else:
            raise ValueError(f"Embodiment name {self.embodiment_name} not supported")
