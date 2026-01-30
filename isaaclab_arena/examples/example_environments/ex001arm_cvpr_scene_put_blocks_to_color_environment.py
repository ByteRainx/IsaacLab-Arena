# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""EX001Arm put-blocks-to-color task with cvpr_assets background."""

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class Ex001ArmCvprScenePutBlocksToColorEnvironment(ExampleEnvironmentBase):
    """Put blocks to matching color destinations task with cvpr_assets background for EX001Arm."""

    name: str = "ex001arm_cvpr_scene_put_blocks_to_color"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        # NOTE: This method is called after the simulation app is started.
        # Avoid importing IsaacLab modules that depend on Kit/Omni at module import time.
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim.spawners.lights import DistantLightCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.put_blocks_to_color_task import ThreeBlocksToColorTask
        from isaaclab_arena.utils.pose import Pose

        class _CvprScenePutBlocksToColorTask(ThreeBlocksToColorTask):
            """Put-blocks-to-color task with render knobs for heavy 3DGS backgrounds."""

            def modify_env_cfg(self, env_cfg):
                if hasattr(env_cfg, "wait_for_textures"):
                    env_cfg.wait_for_textures = True
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
                    env_cfg.sim.render_interval = 1

                # Add distant light to the scene.
                env_cfg.scene.distant_light = AssetBaseCfg(
                    prim_path="/World/DistantLight",
                    spawn=DistantLightCfg(
                        color=(1.0, 1.0, 1.0),
                        intensity=1000.0,
                        angle=0.53,
                    ),
                )
                return env_cfg

        # Background (fixed)
        background = self.asset_registry.get_asset_by_name("cvpr_background")()
        background.set_initial_pose(
            Pose(
                position_xyz=(-1.0, -0.562, 0.0),
                rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
            )
        )

        # Robot initial pose (fixed)
        robot_initial_pos = (-0.51676, -0.25918, -0.58061)
        robot_initial_rot = (1.0, 0.0, 0.0, 0.0)  # wxyz quaternion

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

        # Three blocks (fixed)
        yellow_brick = self.asset_registry.get_asset_by_name("yellow_brick")()
        yellow_brick.set_initial_pose(
            Pose(
                position_xyz=(0.1, -0.15, -0.15),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        green_brick = self.asset_registry.get_asset_by_name("green_brick")()
        green_brick.set_initial_pose(
            Pose(
                position_xyz=(0.1, 0.0, -0.15),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        red_brick = self.asset_registry.get_asset_by_name("red_brick")()
        red_brick.set_initial_pose(
            Pose(
                position_xyz=(0.1, 0.15, -0.15),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Three destination papers (fixed)
        yellow_paper = self.asset_registry.get_asset_by_name("yellow_paper")()
        yellow_paper.set_initial_pose(
            Pose(
                position_xyz=(0.23565, 0.02566, -0.22092),
                rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711),
            )
        )

        green_paper = self.asset_registry.get_asset_by_name("green_paper")()
        green_paper.set_initial_pose(
            Pose(
                position_xyz=(0.23825, -0.29192, -0.22277),
                rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711),
            )
        )

        pink_paper = self.asset_registry.get_asset_by_name("pink_paper")()
        pink_paper.set_initial_pose(
            Pose(
                position_xyz=(0.24031, -0.60498, -0.22564),
                rotation_wxyz=(0.70711, 0.0, 0.0, 0.70711),
            )
        )

        # Create scene with all assets
        scene = Scene(
            assets=[
                background,
                yellow_brick,
                green_brick,
                red_brick,
                yellow_paper,
                green_paper,
                pink_paper,
            ]
        )

        # Create task with three blocks and three destinations
        task = _CvprScenePutBlocksToColorTask(
            block_1=yellow_brick,
            block_2=green_brick,
            block_3=red_brick,
            destination_1=yellow_paper,
            destination_2=green_paper,
            destination_3=pink_paper,
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
