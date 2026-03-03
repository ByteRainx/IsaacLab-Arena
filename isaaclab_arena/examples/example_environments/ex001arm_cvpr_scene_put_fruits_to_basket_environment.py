# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""EX001Arm put-fruits-to-basket task with cvpr_assets background."""

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class Ex001ArmCvprScenePutFruitsToBasketEnvironment(ExampleEnvironmentBase):
    """Put apple, fruit and banana into pink platform basket with cvpr_assets background."""

    name: str = "ex001arm_cvpr_scene_put_fruits_to_basket"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        # NOTE: This method is called after the simulation app is started.
        # Avoid importing IsaacLab modules that depend on Kit/Omni at module import time.
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab.sim.spawners.lights import DistantLightCfg
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.put_fruits_to_basket_task import PutFruitsToBasketTask
        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose

        class _CvprScenePutFruitsToBasketTask(PutFruitsToBasketTask):
            """Put-fruits-to-basket task with render knobs and randomized object positions."""

            def __init__(self, x_range, y_range, z_val, **kwargs):
                self._obj_x_range = x_range
                self._obj_y_range = y_range
                self._obj_z_val = z_val
                super().__init__(**kwargs)

            def modify_env_cfg(self, env_cfg):
                if hasattr(env_cfg, "wait_for_textures"):
                    env_cfg.wait_for_textures = True
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
                    env_cfg.sim.render_interval = 1

                env_cfg.scene.distant_light = AssetBaseCfg(
                    prim_path="/World/DistantLight",
                    spawn=DistantLightCfg(
                        color=(1.0, 1.0, 1.0),
                        intensity=1000.0,
                        angle=0.53,
                    ),
                )
                return env_cfg

            def _create_events_cfg(self):
                """Override to randomize four objects independently on reset."""
                fields = []
                asset_cfgs = [SceneEntityCfg(obj.name) for obj in self.all_objects]

                event_cfg = EventTermCfg(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": self._obj_x_range,
                            "y": self._obj_y_range,
                            "z": (self._obj_z_val, self._obj_z_val),
                            "yaw": (-0.5, 0.5),
                        },
                        "asset_cfgs": asset_cfgs,
                    },
                )
                fields.append(("reset_objects_pose", EventTermCfg, event_cfg))

                EventsCfgClass = make_configclass("RandomFruitObjectsEventsCfg", fields)
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

        obj_z = -0.15
        obj_x_range = (0.0, 0.1)
        obj_y_range = (-0.7, 0.15)

        bread = self.asset_registry.get_asset_by_name("bread")()
        bread.set_initial_pose(Pose(position_xyz=(0.05, -0.30, obj_z), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        fruit = self.asset_registry.get_asset_by_name("fruit")()
        fruit.set_initial_pose(Pose(position_xyz=(0.05, -0.10, obj_z), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        apple = self.asset_registry.get_asset_by_name("apple")()
        apple.set_initial_pose(Pose(position_xyz=(0.05, 0.10, obj_z), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        banana = self.asset_registry.get_asset_by_name("banana")()
        banana.set_initial_pose(Pose(position_xyz=(0.05, 0.30, obj_z), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        platform_pink = self.asset_registry.get_asset_by_name("platform_pink")()
        platform_pink.set_initial_pose(
            Pose(
                position_xyz=(0.26181, -0.19364, -0.22564),
                rotation_wxyz=(0.0, 0.70711, 0.70711, 0.0),
            )
        )

        scene = Scene(assets=[background, bread, fruit, apple, banana, platform_pink])
        task = _CvprScenePutFruitsToBasketTask(
            x_range=obj_x_range,
            y_range=obj_y_range,
            z_val=obj_z,
            target_objects=[apple, fruit, banana],
            all_objects=[bread, fruit, apple, banana],
            basket=platform_pink,
            background_scene=background,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--teleop_device", type=str, default=None)
