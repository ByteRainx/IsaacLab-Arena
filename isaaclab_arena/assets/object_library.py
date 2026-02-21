# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_arena.affordances.openable import Openable
from isaaclab_arena.affordances.pressable import Pressable
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


class LibraryObject(Object):
    """
    Base class for objects in the library which are defined in this file.
    These objects have class attributes (rather than instance attributes).
    """

    name: str
    tags: list[str]
    usd_path: str
    object_type: ObjectType = ObjectType.RIGID
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            prim_path=prim_path,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=self.scale,
            initial_pose=initial_pose,
            **kwargs,
        )


# TODO(peterd, 2025.11.05): Update all OV drive paths to use {ISAACLAB_NUCLEUS_DIR}
# alias prior to public release once assets are synced to S3
@register_asset
class CrackerBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "cracker_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class MustardBottle(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "mustard_bottle"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class SugarBox(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "sugar_box"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class TomatoSoupCan(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "tomato_soup_can"
    tags = ["object"]
    usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PowerDrill(LibraryObject):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    name = "power_drill"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/power_drill_physics/power_drill_physics.usd"

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Microwave(LibraryObject, Openable):
    """A microwave oven."""

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "microwave"
    tags = ["object", "openable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", file_name="Microwave039", file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    openable_joint_name = "microjoint"
    openable_open_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_open_threshold=self.openable_open_threshold,
        )


@register_asset
class CoffeeMachine(LibraryObject, Pressable):
    """
    Encapsulates the pick-up object config for a pick-and-place environment.
    """

    # Only required when using Lightwheel SDK
    from lightwheel_sdk.loader import object_loader

    name = "coffee_machine"
    tags = ["object", "pressable"]
    file_path, object_name, metadata = object_loader.acquire_by_registry(
        registry_type="fixtures", registry_name=["coffee_machine"], file_type="USD"
    )
    usd_path = file_path
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    pressable_joint_name = "CoffeeMachine108_Button002_joint"
    pressedness_threshold = 0.5

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(
            prim_path=prim_path,
            initial_pose=initial_pose,
            pressable_joint_name=self.pressable_joint_name,
            pressedness_threshold=self.pressedness_threshold,
        )


@register_asset
class OfficeTable(LibraryObject):
    """
    A basic office table.
    """

    name = "office_table"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/table.usd"
    default_prim_path = "{ENV_REGEX_NS}/office_table"
    scale = (1.0, 1.0, 0.7)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueSortingBin(LibraryObject):
    """
    A blue plastic sorting bin.
    """

    name = "blue_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_sorting_bin"
    scale = (4.0, 2.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlueExhaustPipe(LibraryObject):
    """
    A blue exhaust pipe.
    """

    name = "blue_exhaust_pipe"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_exhaust_pipe.usd"
    default_prim_path = "{ENV_REGEX_NS}/blue_exhaust_pipe"
    scale = (0.55, 0.55, 1.4)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BrownBox(LibraryObject):
    """
    A brown box.
    """

    name = "brown_box"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Arena/assets/object_library/brown_box/brown_box.usd"
    default_prim_path = "{ENV_REGEX_NS}/brown_box"
    scale = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


def _get_assets_root() -> str:
    """
    Resolve the assets directory path.
    """
    arena_root = Path(__file__).resolve().parents[2]
    assets_path = arena_root / "assets"
    return str(assets_path)


@register_asset
class Ball(LibraryObject):
    """
    A ball from hunyuan assets.
    """

    name = "ball"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/ball/ball_physics.usd"
    scale = (0.0005, 0.0005, 0.0005)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class BlackBowl(LibraryObject):
    """
    A black bowl from hunyuan assets.
    """

    name = "black_bowl"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/black_bowl/black_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Bottom(LibraryObject):
    """
    A bottom from hunyuan assets.
    """

    name = "bottom"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/bottom/bottom_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Apple(LibraryObject):
    """
    An apple from hunyuan assets.
    """

    name = "apple"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/apple/apple_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Banana(LibraryObject):
    """
    A banana from hunyuan assets.
    """

    name = "banana"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/banana/banana_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Bread(LibraryObject):
    """
    A bread from hunyuan assets.
    """

    name = "bread"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/bread/bread_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ButtonBlue(LibraryObject):
    """
    A blue button from hunyuan assets.
    """

    name = "button_blue"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/button_blue/button_blue_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ButtonGreen(LibraryObject):
    """
    A green button from hunyuan assets.
    """

    name = "button_green"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/button_green/button_green_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ButtonOrange(LibraryObject):
    """
    An orange button from hunyuan assets.
    """

    name = "button_orange"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/button_orange/button_orange_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ButtonPink(LibraryObject):
    """
    A pink button from hunyuan assets.
    """

    name = "button_pink"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/button_pink/button_pink_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class ButtonPurple(LibraryObject):
    """
    A purple button from hunyuan assets.
    """

    name = "button_purple"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/button_purple/button_purple_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Cherry(LibraryObject):
    """
    A cherry from hunyuan assets.
    """

    name = "cherry"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/cherry/cherry_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Fruit(LibraryObject):
    """
    A fruit from hunyuan assets.
    """

    name = "fruit"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/fruit/fruit_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Grape(LibraryObject):
    """
    A grape from hunyuan assets.
    """

    name = "grape"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/grape/grape_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Orange(LibraryObject):
    """
    An orange from hunyuan assets.
    """

    name = "orange"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/orange/orange_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PlatformPink(LibraryObject):
    """
    A pink platform from hunyuan assets.
    """

    name = "platform_pink"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/platform_pink/platform_pink_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PlatformWhite(LibraryObject):
    """
    A white platform from hunyuan assets.
    """

    name = "platform_white"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/platform_white/platform_white_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PlatformYellow(LibraryObject):
    """
    A yellow platform from hunyuan assets.
    """

    name = "platform_yellow"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/platform_yellow/platform_yellow_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class GreenBrick(LibraryObject):
    """
    A green brick from hunyuan assets.
    """

    name = "green_brick"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/green_brick/green_brick_physics.usd"
    scale = (0.0004, 0.0004, 0.0004)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

@register_asset
class RedBrick(LibraryObject):
    """
    A red brick from hunyuan assets.
    """

    name = "red_brick"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/red_brick/red_brick_physics.usd"
    scale = (0.0004, 0.0004, 0.0004)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

@register_asset
class YellowBrick(LibraryObject):
    """
    A yellow brick from hunyuan assets.
    """

    name = "yellow_brick"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/yellow_brick/yellow_brick_physics.usd"
    scale = (0.0004, 0.0004, 0.0004)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

@register_asset
class GreenPaper(LibraryObject):
    """
    A green paper from hunyuan assets.
    """

    name = "green_paper"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/green_paper/green_paper_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class PinkPaper(LibraryObject):
    """
    A pink paper from hunyuan assets.
    """

    name = "pink_paper"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/pink_paper/pink_paper_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class YellowPaper(LibraryObject):
    """
    A yellow paper from hunyuan assets.
    """

    name = "yellow_paper"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/yellow_paper/yellow_paper_physics.usd"
    scale = (0.003, 0.003, 0.003)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Stick(LibraryObject):
    """
    A stick from hunyuan assets.
    """

    name = "stick"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/stick/stick_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class WhiteBowl(LibraryObject):
    """
    A white bowl from hunyuan assets.
    """

    name = "white_bowl"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/white_bowl/white_bowl_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class Wood(LibraryObject):
    """
    A wood piece from hunyuan assets.
    """

    name = "wood"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/wood/wood_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)


@register_asset
class WoodBottom(LibraryObject):
    """
    A wood bottom from hunyuan assets.
    """

    name = "wood_bottom"
    tags = ["object"]
    usd_path = f"{_get_assets_root()}/hunyuan_assets/wood_bottom/wood_bottom_physics.usd"
    scale = (0.001, 0.001, 0.001)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None):
        super().__init__(prim_path=prim_path, initial_pose=initial_pose)

