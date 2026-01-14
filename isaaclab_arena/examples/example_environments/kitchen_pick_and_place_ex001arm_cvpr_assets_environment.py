# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class KitchenPickAndPlaceEx001ArmCvprAssetsEnvironment(ExampleEnvironmentBase):
    """Kitchen pick-and-place task setup, but with the `cvpr_assets` background."""

    name: str = "kitchen_pick_and_place_ex001arm_cvpr_assets"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        # NOTE: This method is called after the simulation app is started.
        # Avoid importing IsaacLab modules that depend on Kit/Omni at module import time.
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
        from isaaclab.sim.spawners.lights import DistantLightCfg

        from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        class _DestinationMarker(ObjectBase):
            """A simple static cuboid used as a pick-and-place goal location."""

            def __init__(
                self,
                name: str = "destination_location",
                prim_path: str | None = None,
                initial_pose: Pose | None = None,
                size_xyz: tuple[float, float, float] = (0.10, 0.10, 0.02),
            ):
                super().__init__(name=name, prim_path=prim_path, object_type=ObjectType.RIGID, tags=["object"])
                self.initial_pose = initial_pose or Pose(
                    position_xyz=(0.9, 0.0, 0.05),
                    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
                self.size_xyz = size_xyz
                self.object_cfg = self._init_object_cfg()

            def set_initial_pose(self, pose: Pose) -> None:
                self.initial_pose = pose
                self.object_cfg = self._init_object_cfg()

            def get_initial_pose(self) -> Pose | None:
                return self.initial_pose

            def _generate_rigid_cfg(self) -> RigidObjectCfg:
                object_cfg = RigidObjectCfg(
                    prim_path=self.prim_path,
                    spawn=sim_utils.CuboidCfg(
                        size=self.size_xyz,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                        activate_contact_sensors=False,
                    ),
                )
                object_cfg.init_state.pos = self.initial_pose.position_xyz
                object_cfg.init_state.rot = self.initial_pose.rotation_wxyz
                return object_cfg

            def _generate_articulation_cfg(self) -> ArticulationCfg:
                raise NotImplementedError

            def _generate_base_cfg(self) -> AssetBaseCfg:
                raise NotImplementedError

        class _CvprAssetsPickAndPlaceTask(PickAndPlaceTask):
            """Pick-and-place task with render knobs suitable for heavy 3DGS backgrounds."""

            def modify_env_cfg(self, env_cfg):
                if hasattr(env_cfg, "wait_for_textures"):
                    env_cfg.wait_for_textures = True
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
                    env_cfg.sim.render_interval = 1
                if hasattr(env_cfg, "decimation"):
                    env_cfg.decimation = 1
                
                # Add distant light to scene
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
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=tuple(args_cli.object_xyz),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        destination_location = _DestinationMarker(
            initial_pose=Pose(
                position_xyz=tuple(args_cli.destination_xyz),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=_CvprAssetsPickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--background", type=str, default="cvpr_assets")
        parser.add_argument("--object", type=str, default="cracker_box")
        parser.add_argument("--embodiment", type=str, default="ex001arm")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument(
            "--object_xyz",
            type=float,
            nargs=3,
            default=(-1.41099, -0.17982, -0.10477),
            help="Initial XYZ of the pick-up object in the background frame.",
        )
        parser.add_argument(
            "--destination_xyz",
            type=float,
            nargs=3,
            default=(0.9, 0.0, 0.05),
            help="Initial XYZ of the destination marker in the background frame.",
        )
