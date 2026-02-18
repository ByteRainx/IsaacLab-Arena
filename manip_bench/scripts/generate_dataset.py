# SPDX-License-Identifier: Apache-2.0
#
# Adapted from isaaclab_arena.scripts.generate_dataset
# Uses ManipBench CLI and auto-registers custom extensions.

"""Mimic data generation for ManipBench environments.

Workflow:
    1. Record source demos  →  ``python -m manip_bench.scripts.record ...``
    2. Annotate subtasks    →  ``python -m manip_bench.scripts.annotate_demos ...``
    3. Generate dataset      →  ``python -m manip_bench.scripts.generate_dataset ...``

Example::

    python -m manip_bench.scripts.generate_dataset \\
        --generation_num_trials 100 \\
        --input_file ./datasets/annotated.hdf5 \\
        --output_file ./datasets/generated.hdf5 \\
        --enable_pinocchio \\
        ex001arm_cvpr_scene_pick_and_place \\
        --embodiment ex001arm --background cvpr_background --object cracker_box
"""

from typing import Any

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from manip_bench.scripts.cli import (
    add_manip_bench_cli_args,
    get_env_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--generation_num_trials",
    type=int,
    required=True,
    help="Number of demos to be generated.",
)
parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="File path to the source (annotated) dataset file.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask during generation (debug only).",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio (required for Pink IK / GR1T2).",
)

add_manip_bench_cli_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import manip_bench.extensions  # noqa: F401

import asyncio
import inspect
import random

import gymnasium as gym
import numpy as np
import torch

import isaaclab_mimic.envs  # noqa: F401
import omni
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation
from isaaclab_mimic.datagen.utils import setup_output_paths


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()


def setup_env_config(
    args_cli: Any,
    output_dir: str,
    output_file_name: str,
    num_envs: int,
    device: str,
    generation_num_trials: int | None = None,
) -> tuple[Any, str, Any]:
    """Configure the environment for mimic data generation."""
    arena_builder = get_env_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    if generation_num_trials is not None:
        env_cfg.datagen_config.generation_num_trials = generation_num_trials

    env_cfg.env_name = env_name

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None
    env_cfg.observations.policy.concatenate_terms = False

    if args_cli.enable_cameras:
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
    else:
        env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    if env_cfg.datagen_config.generation_keep_failed:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, env_name, success_term


def main():
    num_envs = args_cli.num_envs
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)

    env_cfg, env_name, success_term = setup_env_config(
        args_cli=args_cli,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    env.reset()

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    simulation_app.close()
