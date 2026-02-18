# SPDX-License-Identifier: Apache-2.0
#
# Adapted from isaaclab_arena.scripts.annotate_demos
# Uses ManipBench CLI and auto-registers custom extensions.

"""Add mimic subtask annotations to recorded demos.

This script replays recorded HDF5 episodes, then either
- **auto-mode** (``--auto``): uses ``env.get_subtask_term_signals()`` to detect
  subtask boundaries automatically, or
- **manual mode**: prompts the operator to press 'S' at each subtask boundary
  while watching the replay.

Example::

    python -m manip_bench.scripts.annotate_demos \\
        --input_file ./datasets/source_demos.hdf5 \\
        --output_file ./datasets/annotated_demos.hdf5 \\
        --auto \\
        ex001arm_cvpr_scene_pick_and_place \\
        --embodiment ex001arm --background cvpr_background --object cracker_box
"""

import gymnasium as gym

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from manip_bench.scripts.cli import (
    add_manip_bench_cli_args,
    get_env_builder_from_cli,
)

parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--input_file",
    type=str,
    default="./datasets/dataset.hdf5",
    help="File name of the dataset to be annotated.",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")
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

import contextlib
import os

import torch

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

if not args_cli.headless and not os.environ.get("HEADLESS", 0):
    from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

is_paused = False
current_action_index = 0
marked_subtask_action_indices = []
skip_episode = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def skip_episode_cb():
    global skip_episode
    skip_episode = True


def mark_subtask_cb():
    global current_action_index, marked_subtask_action_indices
    marked_subtask_action_indices.append(current_action_index)
    print(f"Marked a subtask signal at action index: {current_action_index}")


class PreStepDatagenInfoRecorder(RecorderTerm):
    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)
        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    global current_action_index, skip_episode, is_paused
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)
    first_action = True
    for action_index, action in enumerate(actions):
        current_action_index = action_index
        if first_action:
            first_action = False
        else:
            while is_paused or skip_episode:
                env.sim.render()
                if skip_episode:
                    return False
                continue
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(torch.Tensor(action_tensor))
    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    global skip_episode
    skip_episode = False
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)
    if skip_episode:
        print("\tSkipping the episode.")
        return False
    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')
    return is_episode_annotated_successfully


def annotate_episode_in_manual_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
    subtask_term_signal_names: dict[str, list[str]] | None = None,
) -> bool:
    global is_paused, marked_subtask_action_indices, skip_episode
    if subtask_term_signal_names is None:
        subtask_term_signal_names = {}

    subtask_term_signal_action_indices = {}
    for eef_name, eef_subtask_term_signal_names in subtask_term_signal_names.items():
        if len(eef_subtask_term_signal_names) == 0:
            continue

        while True:
            is_paused = True
            skip_episode = False
            print(f'\tPlaying the episode for subtask annotations for eef "{eef_name}".')
            print("\tSubtask signals to annotate:")
            print(f"\t\t- Termination:\t{eef_subtask_term_signal_names}")
            print('\n\tPress "N" to begin.')
            print('\tPress "B" to pause.')
            print('\tPress "S" to annotate subtask signals.')
            print('\tPress "Q" to skip the episode.\n')
            marked_subtask_action_indices = []
            task_success_result = replay_episode(env, episode, success_term)
            if skip_episode:
                print("\tSkipping the episode.")
                return False

            print(f"\tSubtasks marked at action indices: {marked_subtask_action_indices}")
            expected_subtask_signal_count = len(eef_subtask_term_signal_names)
            if task_success_result and expected_subtask_signal_count == len(marked_subtask_action_indices):
                print(f'\tAll {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.')
                for marked_signal_index in range(expected_subtask_signal_count):
                    subtask_term_signal_action_indices[eef_subtask_term_signal_names[marked_signal_index]] = (
                        marked_subtask_action_indices[marked_signal_index]
                    )
                break

            if not task_success_result:
                print("\tThe final task was not completed.")
                return False

            if expected_subtask_signal_count != len(marked_subtask_action_indices):
                print(
                    f"\tOnly {len(marked_subtask_action_indices)} out of"
                    f' {expected_subtask_signal_count} subtask signals for eef "{eef_name}" were annotated.'
                )
            print(f'\tThe episode will be replayed again for re-marking subtask signals for the eef "{eef_name}".\n')

    annotated_episode = env.recorder_manager.get_episode(0)
    for subtask_term_signal_name, subtask_term_signal_action_index in subtask_term_signal_action_indices.items():
        subtask_signals = torch.ones(len(episode.data["actions"]), dtype=torch.bool)
        subtask_signals[:subtask_term_signal_action_index] = False
        annotated_episode.add(f"obs/datagen_info/subtask_term_signals/{subtask_term_signal_name}", subtask_signals)
    return True


def main():
    global is_paused, current_action_index, marked_subtask_action_indices

    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    arena_builder = get_env_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()
    env_cfg.env_name = env_name

    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        raise NotImplementedError("No success termination term was found in the environment.")

    env_cfg.terminations = None

    env_cfg.recorders: MimicRecorderManagerCfg = MimicRecorderManagerCfg()
    if not args_cli.auto:
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name

    env: ManagerBasedRLMimicEnv = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )
    else:
        subtask_term_signal_names = {}
        for eef_name, eef_subtask_configs in env.cfg.subtask_configs.items():
            subtask_term_signal_names[eef_name] = [
                subtask_config.subtask_term_signal for subtask_config in eef_subtask_configs
            ]
            subtask_term_signal_names[eef_name].pop()

    env.reset()

    if not args_cli.headless and not os.environ.get("HEADLESS", 0):
        keyboard_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
        keyboard_interface.add_callback("N", play_cb)
        keyboard_interface.add_callback("B", pause_cb)
        keyboard_interface.add_callback("Q", skip_episode_cb)
        if not args_cli.auto:
            keyboard_interface.add_callback("S", mark_subtask_cb)
        keyboard_interface.reset()

    exported_episode_count = 0
    processed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                if args_cli.auto:
                    is_ok = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    is_ok = annotate_episode_in_manual_mode(env, episode, success_term, subtask_term_signal_names)

                if is_ok and not skip_episode:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")
            break

    print(
        f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated"
        f" episode{'s' if exported_episode_count > 1 else ''}."
    )
    print("Exiting the app.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
