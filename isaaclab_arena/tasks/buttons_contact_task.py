# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Task: contact three buttons in any order (not necessarily simultaneously)."""

import numpy as np
import torch
from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.terms.events import set_object_pose
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass


def contacts_all_buttons_once(
    env: ManagerBasedRLEnv,
    contact_sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Return success when each button has been contacted at least once."""
    num_envs = env.num_envs
    num_buttons = len(contact_sensor_cfgs)
    history_attr = "_buttons_contact_history_state"

    if (not hasattr(env, history_attr)) or (getattr(env, history_attr).shape != (num_envs, num_buttons)):
        setattr(
            env,
            history_attr,
            torch.zeros((num_envs, num_buttons), dtype=torch.bool, device=env.device),
        )

    contact_history: torch.Tensor = getattr(env, history_attr)

    # Reset history at the beginning of each episode.
    reset_mask = torch.zeros((num_envs,), dtype=torch.bool, device=env.device)
    if hasattr(env, "episode_length_buf"):
        reset_mask = torch.logical_or(reset_mask, env.episode_length_buf == 0)
    if hasattr(env, "reset_buf"):
        reset_mask = torch.logical_or(reset_mask, env.reset_buf > 0)
    if torch.any(reset_mask):
        contact_history[reset_mask] = False

    for idx, sensor_cfg in enumerate(contact_sensor_cfgs):
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        # force_matrix_w shape is usually (N, B, M, 3)
        force_norm = torch.norm(sensor.data.force_matrix_w, dim=-1)
        force_per_env = torch.amax(force_norm, dim=(1, 2))
        in_contact = force_per_env > force_threshold
        contact_history[:, idx] = torch.logical_or(contact_history[:, idx], in_contact)

    return torch.all(contact_history, dim=1)


class ThreeButtonsContactTask(TaskBase):
    """Task for touching three buttons with robot grippers in any order."""

    def __init__(
        self,
        button_1: ObjectBase,
        button_2: ObjectBase,
        button_3: ObjectBase,
        episode_length_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.buttons = [button_1, button_2, button_3]
        self.scene_config = self._create_scene_cfg()
        self.events_cfg = self._create_events_cfg()
        self.termination_cfg = self._create_termination_cfg()

    def _create_scene_cfg(self) -> Any:
        fields = []
        for idx, button in enumerate(self.buttons):
            # Use gripper-mounted contact sensors filtered to each button.
            # This is more robust for assets whose rigid body is under child prims.
            sensor_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_arm_gripper_.*_link",
                filter_prim_paths_expr=[
                    button.get_prim_path(),
                    f"{button.get_prim_path()}/.*",
                ],
            )
            fields.append((f"button_{idx}_contact_sensor", ContactSensorCfg, sensor_cfg))
        SceneCfgClass = make_configclass("ThreeButtonsContactSceneCfg", fields)
        return SceneCfgClass()

    def _create_events_cfg(self) -> Any:
        fields = []
        for idx, button in enumerate(self.buttons):
            initial_pose = button.get_initial_pose()
            if initial_pose is None:
                continue
            event_cfg = EventTermCfg(
                func=set_object_pose,
                mode="reset",
                params={
                    "pose": initial_pose,
                    "asset_cfg": SceneEntityCfg(button.name),
                },
            )
            fields.append((f"reset_button_{idx}_pose", EventTermCfg, event_cfg))
        EventsCfgClass = make_configclass("ThreeButtonsContactEventsCfg", fields)
        return EventsCfgClass()

    def _create_termination_cfg(self) -> "ThreeButtonsContactTerminationsCfg":
        success = TerminationTermCfg(
            func=contacts_all_buttons_once,
            params={
                "contact_sensor_cfgs": [
                    SceneEntityCfg("button_0_contact_sensor"),
                    SceneEntityCfg("button_1_contact_sensor"),
                    SceneEntityCfg("button_2_contact_sensor"),
                ],
                "force_threshold": 1.0,
            },
        )
        return ThreeButtonsContactTerminationsCfg(success=success)

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self) -> str:
        return "Use the robot arms to touch green, pink, and blue buttons in any order."

    def get_mimic_env_cfg(self, embodiment_name: str):
        return ButtonsContactMimicEnvCfg(
            embodiment_name=embodiment_name,
            button_names=[button.name for button in self.buttons],
        )

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ObjectMovedRateMetric(self.buttons[0])]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.buttons[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@configclass
class ThreeButtonsContactTerminationsCfg:
    """Termination terms for three-buttons contact task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class ButtonsContactMimicEnvCfg(MimicEnvCfg):
    """Isaac Lab Mimic config for button-contact task."""

    embodiment_name: str = "franka"
    button_names: list[str] | None = None

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_buttons_contact_D0"
        if self.button_names is None:
            self.button_names = []

        subtask_configs = []
        for i, button_name in enumerate(self.button_names):
            is_last = i == len(self.button_names) - 1
            subtask_configs.append(
                SubTaskConfig(
                    object_ref=button_name,
                    subtask_term_signal=None if is_last else f"touch_{i + 1}",
                    subtask_term_offset_range=(0, 0) if is_last else (3, 8),
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
                    object_ref=self.button_names[0] if self.button_names else "button_green",
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
