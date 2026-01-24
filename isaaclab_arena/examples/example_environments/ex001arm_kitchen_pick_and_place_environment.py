# Copyright (c) 2025, The Isaac Lab Arena Project Developers
# (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import (
    ExampleEnvironmentBase,
)


class Ex001ArmKitchenPickAndPlaceEnvironment(ExampleEnvironmentBase):
    """Kitchen pick-and-place task configured for EX001Arm."""

    name: str = "ex001arm_kitchen_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(
            args_cli.embodiment
        )(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(
                args_cli.teleop_device
            )()
        else:
            teleop_device = None

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.15242, 0.14933, -0.31697),
                rotation_wxyz=(0.99998, 0.0, 0.0, -0.00648),
            )
        )

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.4, 0.0, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(pick_up_object, destination_location, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="cracker_box")
        parser.add_argument("--embodiment", type=str, default="ex001arm")
        parser.add_argument("--teleop_device", type=str, default=None)
