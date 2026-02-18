# SPDX-License-Identifier: Apache-2.0

"""EX001Arm pick-and-place task with CVPR 3DGS background.

This environment composition uses ManipBench extensions (registered via
``manip_bench.extensions``) together with Isaac Lab Arena's composable
environment builder.
"""

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class Ex001ArmCvprScenePickAndPlaceEnvironment(ExampleEnvironmentBase):
    """Pick-and-place task setup with the `cvpr_assets` background for EX001Arm."""

    name: str = "ex001arm_cvpr_scene_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import manip_bench.extensions  # noqa: F401  — register custom assets

        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim.spawners.lights import DistantLightCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        class _CvprScenePickAndPlaceTask(PickAndPlaceTask):
            """Pick-and-place task with render knobs for heavy 3DGS backgrounds."""

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

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        # Set background initial pose (position and rotation)
        background.set_initial_pose(
            Pose(
                position_xyz=(-1.0, -0.562, 0.0),
                rotation_wxyz=(0.04457, 0.0, 0.0, -0.999),
            )
        )
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

        # Robot initial pose
        robot_initial_pos = (-0.51676, -0.25918, -0.58061)
        robot_initial_rot = (1.0, 0.0, 0.0, 0.0)  # wxyz quaternion - no rotation for correct VR orientation

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras,
            xr_anchor_pos=robot_initial_pos,
            xr_anchor_rot=robot_initial_rot,
        )
        teleop_device = (
            self.device_registry.get_device_by_name(args_cli.teleop_device)() if args_cli.teleop_device else None
        )

        embodiment.set_initial_pose(
            Pose(
                position_xyz=robot_initial_pos,
                rotation_wxyz=robot_initial_rot,
            )
        )

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.3, 0.0, -0.2),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        destination_location = self.asset_registry.get_asset_by_name("plate")()
        destination_location.set_initial_pose(
            Pose(
                position_xyz=(0.312, -0.286, -0.232),
                rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=_CvprScenePickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--background", type=str, default="cvpr_background")
        parser.add_argument("--object", type=str, default="cracker_box")
        parser.add_argument("--embodiment", type=str, default="ex001arm")
        parser.add_argument("--teleop_device", type=str, default=None)
