"""ManipBench object asset registrations (Hunyuan assets)."""

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from manip_bench.utils.paths import get_object_usd_path


class ManipBenchObject(Object):
    """Base class for ManipBench object assets with unified path resolution."""

    name: str
    tags: list[str]
    usd_path: str
    object_type: ObjectType = ObjectType.RIGID
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, prim_path: str | None = None, initial_pose: Pose | None = None, **kwargs):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            object_type=self.object_type,
            scale=self.scale,
            prim_path=prim_path,
            initial_pose=initial_pose,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Brick objects (for color-sorting tasks)
# ---------------------------------------------------------------------------

@register_asset
class GreenBrick(ManipBenchObject):
    name = "green_brick"
    tags = ["object", "brick"]
    usd_path = get_object_usd_path("hunyuan_assets/green_brick/green_brick_physics.usd")
    scale = (0.0004, 0.0004, 0.0004)


@register_asset
class RedBrick(ManipBenchObject):
    name = "red_brick"
    tags = ["object", "brick"]
    usd_path = get_object_usd_path("hunyuan_assets/red_brick/red_brick_physics.usd")
    scale = (0.0004, 0.0004, 0.0004)


@register_asset
class YellowBrick(ManipBenchObject):
    name = "yellow_brick"
    tags = ["object", "brick"]
    usd_path = get_object_usd_path("hunyuan_assets/yellow_brick/yellow_brick_physics.usd")
    scale = (0.0004, 0.0004, 0.0004)


# ---------------------------------------------------------------------------
# Paper objects (destination markers for color-sorting)
# ---------------------------------------------------------------------------

@register_asset
class GreenPaper(ManipBenchObject):
    name = "green_paper"
    tags = ["object", "paper"]
    usd_path = get_object_usd_path("hunyuan_assets/green_paper/green_paper_physics.usd")
    scale = (0.003, 0.003, 0.003)


@register_asset
class PinkPaper(ManipBenchObject):
    name = "pink_paper"
    tags = ["object", "paper"]
    usd_path = get_object_usd_path("hunyuan_assets/pink_paper/pink_paper_physics.usd")
    scale = (0.003, 0.003, 0.003)


@register_asset
class YellowPaper(ManipBenchObject):
    name = "yellow_paper"
    tags = ["object", "paper"]
    usd_path = get_object_usd_path("hunyuan_assets/yellow_paper/yellow_paper_physics.usd")
    scale = (0.003, 0.003, 0.003)


# ---------------------------------------------------------------------------
# Tableware and miscellaneous objects
# ---------------------------------------------------------------------------

@register_asset
class Ball(ManipBenchObject):
    name = "ball"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/ball/ball_physics.usd")
    scale = (0.0005, 0.0005, 0.0005)


@register_asset
class BlackBowl(ManipBenchObject):
    name = "black_bowl"
    tags = ["object", "bowl"]
    usd_path = get_object_usd_path("hunyuan_assets/black_bowl/black_physics.usd")
    scale = (0.001, 0.001, 0.001)


@register_asset
class WhiteBowl(ManipBenchObject):
    name = "white_bowl"
    tags = ["object", "bowl"]
    usd_path = get_object_usd_path("hunyuan_assets/white_bowl/white_bowl_physics.usd")
    scale = (0.001, 0.001, 0.001)


@register_asset
class Plate(ManipBenchObject):
    name = "plate"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/plate/plate_physics.usd")
    scale = (0.004, 0.004, 0.004)


@register_asset
class Bottom(ManipBenchObject):
    name = "bottom"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/bottom/bottom_physics.usd")
    scale = (0.001, 0.001, 0.001)


@register_asset
class Stick(ManipBenchObject):
    name = "stick"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/stick/stick_physics.usd")
    scale = (0.001, 0.001, 0.001)


@register_asset
class Wood(ManipBenchObject):
    name = "wood"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/wood/wood_physics.usd")
    scale = (0.001, 0.001, 0.001)


@register_asset
class WoodBottom(ManipBenchObject):
    name = "wood_bottom"
    tags = ["object"]
    usd_path = get_object_usd_path("hunyuan_assets/wood_bottom/wood_bottom_physics.usd")
    scale = (0.001, 0.001, 0.001)
