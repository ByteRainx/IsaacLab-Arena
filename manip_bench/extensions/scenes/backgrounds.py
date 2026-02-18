"""ManipBench background scene registrations."""

from isaaclab_arena.assets.background import Background
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose

from manip_bench.utils.paths import get_scene_usd_path


class ManipBenchBackground(Background):
    """Base class for ManipBench background assets with unified path resolution."""

    name: str
    tags: list[str]
    usd_path: str
    initial_pose: Pose
    object_min_z: float

    def __init__(self, **kwargs):
        super().__init__(
            name=self.name,
            tags=self.tags,
            usd_path=self.usd_path,
            initial_pose=self.initial_pose,
            object_min_z=self.object_min_z,
            **kwargs,
        )


@register_asset
class CvprBackground(ManipBenchBackground):
    """3DGS-reconstructed tabletop background used for CVPR demos."""

    name = "cvpr_background"
    tags = ["background"]
    usd_path = get_scene_usd_path("scene-3dgs/scene_01.usd")
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
    object_min_z = -0.4

    def __init__(self):
        super().__init__()
