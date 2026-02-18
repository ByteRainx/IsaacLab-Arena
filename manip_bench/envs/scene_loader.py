# SPDX-License-Identifier: Apache-2.0

"""Generic environment loader driven by YAML scene configs.

Instead of writing one Python class per environment, define the physical
layout (background, robot, objects, lighting) in a YAML file and let this
loader build the Arena environment automatically.

Supports two modes:
  * **Nominal** — use the exact poses from the YAML (for eval / Real2Sim).
  * **Randomized** — sample object poses within ``workspace_bounds`` (training).

Usage::

    python -m manip_bench.scripts.record \\
        --env_config configs/envs/cvpr_pick_and_place.yaml \\
        --embodiment ex001arm --teleop_device keyboard
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import yaml

from isaaclab_arena.examples.example_environments.example_environment_base import (
    ExampleEnvironmentBase,
)


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _pose_from_dict(d: dict | None) -> tuple[tuple, tuple]:
    """Extract (position, rotation) from a pose dict.  Defaults to identity."""
    if d is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    pos = tuple(d.get("position", [0.0, 0.0, 0.0]))
    rot = tuple(d.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    return pos, rot


class SceneConfigEnvironment(ExampleEnvironmentBase):
    """Build an Arena environment from a YAML config file.

    The config describes:
      * ``scene`` — background asset, robot, lighting, render settings
      * ``task``  — task type, objects with nominal poses / workspace bounds

    CLI flags (``--embodiment``, ``--object``, ``--teleop_device``) can
    override the defaults baked into the YAML.
    """

    def __init__(self, config: dict | str):
        if isinstance(config, str):
            config = _load_yaml(config)
        self._cfg = config
        self.name: str = config["env_id"]

    # ------------------------------------------------------------------
    # Arena interface
    # ------------------------------------------------------------------

    def get_env(self, args_cli: argparse.Namespace):  # noqa: C901
        import manip_bench.extensions  # noqa: F401

        from isaaclab.assets import AssetBaseCfg
        from isaaclab.sim.spawners.lights import DistantLightCfg

        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        cfg = self._cfg
        scene_cfg = cfg["scene"]
        task_cfg = cfg["task"]

        # --- Background ---
        bg_cfg = scene_cfg["background"]
        background = self.asset_registry.get_asset_by_name(bg_cfg["asset"])()
        bg_pos, bg_rot = _pose_from_dict(bg_cfg.get("pose"))
        background.set_initial_pose(Pose(position_xyz=bg_pos, rotation_wxyz=bg_rot))

        # --- Robot ---
        robot_cfg = scene_cfg["robot"]
        robot_asset = getattr(args_cli, "embodiment", None) or robot_cfg["asset"]
        robot_pos, robot_rot = _pose_from_dict(robot_cfg.get("pose"))
        embodiment = self.asset_registry.get_asset_by_name(robot_asset)(
            enable_cameras=getattr(args_cli, "enable_cameras", False),
            xr_anchor_pos=robot_pos,
            xr_anchor_rot=robot_rot,
        )
        embodiment.set_initial_pose(Pose(position_xyz=robot_pos, rotation_wxyz=robot_rot))

        # --- Teleop device ---
        device_name = getattr(args_cli, "teleop_device", None)
        teleop_device = (
            self.device_registry.get_device_by_name(device_name)() if device_name else None
        )

        # --- Task-specific assembly ---
        task_type = task_cfg["type"]
        task_obj, scene_assets = self._build_task(
            task_type, task_cfg, background, args_cli,
        )

        # --- Lighting (optional) ---
        light_cfg = scene_cfg.get("lighting")

        # Override modify_env_cfg on the task to inject render / light settings
        render_cfg = scene_cfg.get("render", {})
        original_modify = task_obj.modify_env_cfg

        def _patched_modify_env_cfg(env_cfg):
            env_cfg = original_modify(env_cfg)
            if render_cfg.get("wait_for_textures"):
                if hasattr(env_cfg, "wait_for_textures"):
                    env_cfg.wait_for_textures = True
            if render_cfg.get("render_interval"):
                if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "render_interval"):
                    env_cfg.sim.render_interval = render_cfg["render_interval"]
            if light_cfg:
                env_cfg.scene.distant_light = AssetBaseCfg(
                    prim_path="/World/DistantLight",
                    spawn=DistantLightCfg(
                        color=tuple(light_cfg.get("color", [1.0, 1.0, 1.0])),
                        intensity=light_cfg.get("intensity", 1000.0),
                        angle=light_cfg.get("angle", 0.53),
                    ),
                )
            return env_cfg

        task_obj.modify_env_cfg = _patched_modify_env_cfg

        scene = Scene(assets=[background] + scene_assets)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task_obj,
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--embodiment", type=str, default=None)
        parser.add_argument("--object", type=str, default=None)
        parser.add_argument("--background", type=str, default=None)
        parser.add_argument("--teleop_device", type=str, default=None)

    # ------------------------------------------------------------------
    # Task builders
    # ------------------------------------------------------------------

    def _build_task(
        self,
        task_type: str,
        task_cfg: dict,
        background: Any,
        args_cli: argparse.Namespace,
    ) -> tuple[Any, list]:
        """Dispatch to the appropriate task builder.

        Returns ``(task_object, [scene_assets_excluding_background])``.
        """
        builders = {
            "pick_and_place": self._build_pick_and_place,
            "color_sorting": self._build_color_sorting,
        }
        builder = builders.get(task_type)
        if builder is None:
            raise ValueError(
                f"Unknown task type '{task_type}'. Available: {list(builders)}"
            )
        return builder(task_cfg, background, args_cli)

    def _build_pick_and_place(
        self, task_cfg: dict, background: Any, args_cli: argparse.Namespace,
    ) -> tuple[Any, list]:
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        # Target object
        target_cfg = task_cfg["target"]
        target_asset = getattr(args_cli, "object", None) or target_cfg["asset"]
        target = self.asset_registry.get_asset_by_name(target_asset)()
        t_pos, t_rot = _pose_from_dict(target_cfg.get("pose"))
        target.set_initial_pose(Pose(position_xyz=t_pos, rotation_wxyz=t_rot))

        # Destination
        dest_cfg = task_cfg["destination"]
        dest = self._make_destination(dest_cfg, background)

        task = PickAndPlaceTask(target, dest, background)
        return task, [target, dest]

    def _build_color_sorting(
        self, task_cfg: dict, background: Any, args_cli: argparse.Namespace,
    ) -> tuple[Any, list]:
        from isaaclab.managers import EventTermCfg, SceneEntityCfg
        from isaaclab_tasks.manager_based.manipulation.stack.mdp import (
            franka_stack_events,
        )

        from isaaclab_arena.utils.configclass import make_configclass
        from isaaclab_arena.utils.pose import Pose
        from manip_bench.extensions.tasks.put_blocks_to_color import (
            ThreeBlocksToColorTask,
        )

        blocks_cfg = task_cfg["blocks"]
        dests_cfg = task_cfg["destinations"]
        rand_cfg = task_cfg.get("randomize_blocks", {})

        blocks = []
        for bcfg in blocks_cfg:
            b = self.asset_registry.get_asset_by_name(bcfg["asset"])()
            bp, br = _pose_from_dict(bcfg.get("pose"))
            b.set_initial_pose(Pose(position_xyz=bp, rotation_wxyz=br))
            blocks.append(b)

        dests = []
        for dcfg in dests_cfg:
            d = self.asset_registry.get_asset_by_name(dcfg["asset"])()
            dp, dr = _pose_from_dict(dcfg.get("pose"))
            d.set_initial_pose(Pose(position_xyz=dp, rotation_wxyz=dr))
            dests.append(d)

        x_range = tuple(rand_cfg.get("x", (0.0, 0.0)))
        y_range = tuple(rand_cfg.get("y", (0.0, 0.0)))
        z_val = rand_cfg.get("z", blocks_cfg[0]["pose"]["position"][2])

        class _ColorSortingWithRandomization(ThreeBlocksToColorTask):
            def __init__(self, xr, yr, zv, **kwargs):
                self._xr, self._yr, self._zv = xr, yr, zv
                super().__init__(**kwargs)

            def _create_events_cfg(self):
                if self._xr == (0.0, 0.0) and self._yr == (0.0, 0.0):
                    return super()._create_events_cfg()
                asset_cfgs = [SceneEntityCfg(b.name) for b in self.blocks]
                event = EventTermCfg(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": {
                            "x": self._xr,
                            "y": self._yr,
                            "z": (self._zv, self._zv),
                            "yaw": tuple(rand_cfg.get("yaw", (-0.5, 0.5))),
                        },
                        "asset_cfgs": asset_cfgs,
                    },
                )
                Cls = make_configclass(
                    "RandomBlockEventsCfg",
                    [("reset_blocks_pose", EventTermCfg, event)],
                )
                return Cls()

        if len(blocks) != 3 or len(dests) != 3:
            raise ValueError("color_sorting requires exactly 3 blocks and 3 destinations")

        task = _ColorSortingWithRandomization(
            xr=x_range, yr=y_range, zv=z_val,
            block_1=blocks[0], block_2=blocks[1], block_3=blocks[2],
            destination_1=dests[0], destination_2=dests[1], destination_3=dests[2],
            background_scene=background,
        )
        return task, blocks + dests

    def _make_destination(self, dest_cfg: dict, background: Any) -> Any:
        """Build a destination — either a regular asset or an ObjectReference."""
        from isaaclab_arena.utils.pose import Pose

        if dest_cfg.get("type") == "object_reference":
            from isaaclab_arena.assets.object_base import ObjectType
            from isaaclab_arena.assets.object_reference import ObjectReference

            parent = background if dest_cfg.get("parent_asset") == "background" else None
            return ObjectReference(
                name=dest_cfg["name"],
                prim_path=dest_cfg["prim_path"],
                parent_asset=parent,
                object_type=ObjectType.RIGID,
            )

        asset = self.asset_registry.get_asset_by_name(dest_cfg["asset"])()
        dp, dr = _pose_from_dict(dest_cfg.get("pose"))
        asset.set_initial_pose(Pose(position_xyz=dp, rotation_wxyz=dr))
        return asset


# -----------------------------------------------------------------------
# Utilities for discovering and loading YAML env configs
# -----------------------------------------------------------------------

def discover_env_configs(
    config_dir: str | None = None,
) -> dict[str, SceneConfigEnvironment]:
    """Scan a directory for ``*.yaml`` env configs and return name→env map."""
    if config_dir is None:
        config_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "configs", "envs",
        )
    envs: dict[str, SceneConfigEnvironment] = {}
    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        try:
            cfg = _load_yaml(path)
            env = SceneConfigEnvironment(cfg)
            envs[env.name] = env
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
    return envs
