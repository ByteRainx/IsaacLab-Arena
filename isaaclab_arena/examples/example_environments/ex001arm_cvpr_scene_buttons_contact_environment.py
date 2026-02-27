# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""EX001Arm buttons-contact task with cvpr_assets background."""

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class Ex001ArmCvprSceneButtonsContactEnvironment(ExampleEnvironmentBase):
    """Touch green, pink, blue buttons (in any order) with cvpr_assets background."""

    name: str = "ex001arm_cvpr_scene_buttons_contact"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        # NOTE: This method is called after the simulation app is started.
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab.sim.spawners.lights import DomeLightCfg
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.buttons_contact_task import ThreeButtonsContactTask
        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose

        class _CvprSceneButtonsContactTask(ThreeButtonsContactTask):
            """Buttons-contact task with render knobs for heavy 3DGS backgrounds."""

            def __init__(self, x_range, y_range, z_val, **kwargs):
                self._button_x_range = x_range
                self._button_y_range = y_range
                self._button_z_val = z_val
                super().__init__(**kwargs)

            def modify_env_cfg(self, env_cfg):
                if hasattr(env_cfg, "wait_for_textures"):
                    env_cfg.wait_for_textures = True
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
                    env_cfg.sim.render_interval = 1
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
                    env_cfg.sim.physx.enable_ccd = True

                env_cfg.scene.dome_light = AssetBaseCfg(
                    prim_path="/World/DomeLight",
                    spawn=DomeLightCfg(
                        color=(1.0, 1.0, 1.0),
                        intensity=1000.0,
                    ),
                )
                return env_cfg

            def _create_events_cfg(self):
                """Override to randomize three buttons independently on reset."""
                fields = []
                asset_cfgs = [SceneEntityCfg(button.name) for button in self.buttons]

                event_cfg = EventTermCfg(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": self._button_x_range,
                            "y": self._button_y_range,
                            "z": (self._button_z_val, self._button_z_val),
                            "pitch": (3.1415926, 3.1415926),
                        },
                        "asset_cfgs": asset_cfgs,
                    },
                )
                fields.append(("reset_buttons_pose", EventTermCfg, event_cfg))

                EventsCfgClass = make_configclass("RandomButtonsEventsCfg", fields)
                return EventsCfgClass()

        background = self.asset_registry.get_asset_by_name("cvpr_background")()
        background.set_initial_pose(
            Pose(
                position_xyz=(-1.0, -0.562, 0.0),
                rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
            )
        )

        robot_initial_pos = (-0.51676, -0.25918, -0.58061)
        robot_initial_rot = (1.0, 0.0, 0.0, 0.0)

        embodiment = self.asset_registry.get_asset_by_name("ex001arm")(
            enable_cameras=args_cli.enable_cameras,
            xr_anchor_pos=robot_initial_pos,
            xr_anchor_rot=robot_initial_rot,
        )
        embodiment.set_initial_pose(
            Pose(
                position_xyz=robot_initial_pos,
                rotation_wxyz=robot_initial_rot,
            )
        )

        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)()
            if args_cli.teleop_device
            else None
        )

        button_z = -0.17
        button_x_range = (0.0, 0.3)
        button_y_range = (-0.7, 0.15)

        button_green = self.asset_registry.get_asset_by_name("button_green")()
        button_green.set_initial_pose(
            Pose(
                position_xyz=(0.05, -0.15, button_z),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        button_pink = self.asset_registry.get_asset_by_name("button_pink")()
        button_pink.set_initial_pose(
            Pose(
                position_xyz=(0.05, 0.0, button_z),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        button_blue = self.asset_registry.get_asset_by_name("button_blue")()
        button_blue.set_initial_pose(
            Pose(
                position_xyz=(0.05, 0.15, button_z),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(assets=[background, button_green, button_pink, button_blue])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=_CvprSceneButtonsContactTask(
                x_range=button_x_range,
                y_range=button_y_range,
                z_val=button_z,
                button_1=button_green,
                button_2=button_pink,
                button_3=button_blue,
            ),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None)
