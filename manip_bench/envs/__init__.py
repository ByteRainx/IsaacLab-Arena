"""ManipBench environment compositions.

Each module defines a complete environment by composing an embodiment, a scene,
and a task using Isaac Lab Arena's ``ExampleEnvironmentBase`` pattern.

Environments are also registered with ``gymnasium`` so they can be instantiated
via the standard ``gym.make()`` API.
"""

import gymnasium as gym

from .cvpr_pick_and_place import Ex001ArmCvprScenePickAndPlaceEnvironment
from .cvpr_put_blocks_to_color import Ex001ArmCvprScenePutBlocksToColorEnvironment
from .kitchen_pick_and_place import Ex001ArmKitchenPickAndPlaceEnvironment

ENVIRONMENTS = {
    Ex001ArmCvprScenePickAndPlaceEnvironment.name: Ex001ArmCvprScenePickAndPlaceEnvironment,
    Ex001ArmCvprScenePutBlocksToColorEnvironment.name: Ex001ArmCvprScenePutBlocksToColorEnvironment,
    Ex001ArmKitchenPickAndPlaceEnvironment.name: Ex001ArmKitchenPickAndPlaceEnvironment,
}


def get_environment(name: str):
    """Look up a ManipBench environment class by name."""
    if name not in ENVIRONMENTS:
        available = ", ".join(ENVIRONMENTS.keys())
        raise KeyError(f"Unknown environment '{name}'. Available: {available}")
    return ENVIRONMENTS[name]


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
