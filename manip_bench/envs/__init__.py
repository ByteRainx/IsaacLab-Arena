"""ManipBench environment compositions.

Environments can be defined in two ways:

1. **Python classes** — hand-written ``ExampleEnvironmentBase`` subclasses
   in this package (legacy, still supported).
2. **YAML configs** — files in ``configs/envs/*.yaml`` loaded automatically
   by :class:`~manip_bench.envs.scene_loader.SceneConfigEnvironment`.

Both are merged into the ``ENVIRONMENTS`` dict and registered with the CLI.
"""

import gymnasium as gym

# --- Legacy Python-class environments ---
from .cvpr_pick_and_place import Ex001ArmCvprScenePickAndPlaceEnvironment
from .cvpr_put_blocks_to_color import Ex001ArmCvprScenePutBlocksToColorEnvironment
from .kitchen_pick_and_place import Ex001ArmKitchenPickAndPlaceEnvironment

ENVIRONMENTS: dict = {
    Ex001ArmCvprScenePickAndPlaceEnvironment.name: Ex001ArmCvprScenePickAndPlaceEnvironment,
    Ex001ArmCvprScenePutBlocksToColorEnvironment.name: Ex001ArmCvprScenePutBlocksToColorEnvironment,
    Ex001ArmKitchenPickAndPlaceEnvironment.name: Ex001ArmKitchenPickAndPlaceEnvironment,
}

# --- YAML-driven environments (auto-discovered) ---
from .scene_loader import discover_env_configs

_yaml_envs = discover_env_configs()
for _name, _env in _yaml_envs.items():
    if _name not in ENVIRONMENTS:
        ENVIRONMENTS[_name] = _env


def get_environment(name: str):
    """Look up a ManipBench environment by name (class or YAML-based)."""
    if name not in ENVIRONMENTS:
        available = ", ".join(sorted(ENVIRONMENTS.keys()))
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    env = ENVIRONMENTS[name]
    if isinstance(env, type):
        return env
    return env  # SceneConfigEnvironment instance


def load_env_from_yaml(config_path: str):
    """Load a single environment from an arbitrary YAML config path."""
    from .scene_loader import SceneConfigEnvironment
    return SceneConfigEnvironment(config_path)


gym.register(
    id="ManipBench-CvprPickAndPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
)

gym.register(
    id="ManipBench-CvprPutBlocksToColor-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
)

gym.register(
    id="ManipBench-KitchenPickAndPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
)
