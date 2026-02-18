# SPDX-License-Identifier: Apache-2.0

"""Real2Sim evaluation runner.

Loads an eval-suite YAML that defines a set of episodes (each with specific
object poses), runs a policy across all episodes × N trials, and reports
per-episode and aggregate success rates.

Usage::

    python -m manip_bench.scripts.eval_suite \\
        --suite configs/eval/cvpr_pick_and_place_suite.yaml \\
        --policy_type zero_action

    python -m manip_bench.scripts.eval_suite \\
        --suite configs/eval/cvpr_pick_and_place_suite.yaml \\
        --policy_type x2robot_closedloop \\
        --x2robot_server_address localhost --x2robot_server_port 8000
"""

import copy
import os

import yaml

from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--suite", type=str, required=True, help="Path to eval suite YAML.")
parser.add_argument("--output", type=str, default=None, help="Path to write results YAML.")

args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import manip_bench.extensions  # noqa: F401

import datetime
import time

import gymnasium as gym
import torch
import tqdm

from manip_bench.envs.scene_loader import SceneConfigEnvironment


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _apply_episode_overrides(env_cfg: dict, episode: dict) -> dict:
    """Return a new env config with episode-specific pose overrides applied."""
    overrides = episode.get("overrides")
    if not overrides:
        return env_cfg

    cfg = copy.deepcopy(env_cfg)
    task = cfg["task"]
    task_type = task["type"]

    if task_type == "pick_and_place":
        if "target" in overrides:
            ov = overrides["target"]
            if "position" in ov:
                task["target"]["pose"]["position"] = ov["position"]
            if "rotation" in ov:
                task["target"]["pose"]["rotation"] = ov["rotation"]
            if "asset" in ov:
                task["target"]["asset"] = ov["asset"]
        if "destination" in overrides:
            ov = overrides["destination"]
            if "position" in ov:
                task["destination"]["pose"]["position"] = ov["position"]
            if "rotation" in ov:
                task["destination"]["pose"]["rotation"] = ov["rotation"]

    elif task_type == "color_sorting":
        if "blocks" in overrides:
            for i, ov in enumerate(overrides["blocks"]):
                if i < len(task["blocks"]):
                    if "position" in ov:
                        task["blocks"][i]["pose"]["position"] = ov["position"]
                    if "rotation" in ov:
                        task["blocks"][i]["pose"]["rotation"] = ov["rotation"]
        if "destinations" in overrides:
            for i, ov in enumerate(overrides["destinations"]):
                if i < len(task["destinations"]):
                    if "position" in ov:
                        task["destinations"][i]["pose"]["position"] = ov["position"]
                    if "rotation" in ov:
                        task["destinations"][i]["pose"]["rotation"] = ov["rotation"]

    return cfg


def _check_success(success_term, env) -> bool:
    if success_term is None:
        return False
    return bool(success_term.func(env, **success_term.params)[0])


def main():
    from manip_bench.scripts.policy_runner_cli import create_policy, setup_policy_argument_parser

    policy_parser = setup_policy_argument_parser(parser)
    args_cli_full = policy_parser.parse_args()

    suite = _load_yaml(args_cli_full.suite)
    suite_name = suite.get("name", os.path.basename(args_cli_full.suite))
    num_trials = suite.get("num_trials", 3)
    max_steps = suite.get("max_steps", 500)
    episodes = suite.get("episodes", [])

    env_config_path = suite["env_config"]
    if not os.path.isabs(env_config_path):
        suite_dir = os.path.dirname(os.path.abspath(args_cli_full.suite))
        candidate = os.path.join(suite_dir, env_config_path)
        if os.path.exists(candidate):
            env_config_path = candidate
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env_config_path = os.path.join(project_root, env_config_path)

    base_env_cfg = _load_yaml(env_config_path)

    print("=" * 60)
    print(f"  Real2Sim Eval Suite: {suite_name}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Trials per episode: {num_trials}")
    print(f"  Max steps: {max_steps}")
    print("=" * 60)

    results = {
        "name": suite_name,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {
            "suite": args_cli_full.suite,
            "env_config": env_config_path,
            "num_trials": num_trials,
            "max_steps": max_steps,
            "policy_type": getattr(args_cli_full, "policy_type", "unknown"),
        },
        "per_episode": [],
    }

    total_successes = 0
    total_trials = 0

    for ep_idx, episode in enumerate(episodes):
        ep_id = episode.get("id", f"ep_{ep_idx:03d}")
        ep_desc = episode.get("description", "")

        ep_env_cfg = _apply_episode_overrides(base_env_cfg, episode)
        # Disable randomization for eval: zero out workspace_bounds
        task = ep_env_cfg.get("task", {})
        if "randomize_blocks" in task:
            task["randomize_blocks"] = {}
        for key in ("target", "destination"):
            if key in task and isinstance(task[key], dict):
                task[key].pop("workspace_bounds", None)

        env_obj = SceneConfigEnvironment(ep_env_cfg)
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        arena_env = env_obj.get_env(args_cli_full)
        builder = ArenaEnvBuilder(arena_env, args_cli_full)
        env_name, env_cfg = builder.build_registered()

        success_term = None
        if hasattr(env_cfg.terminations, "success"):
            success_term = env_cfg.terminations.success
            env_cfg.terminations.success = None
        env_cfg.terminations.time_out = None
        env_cfg.observations.policy.concatenate_terms = False

        env = gym.make(env_name, cfg=env_cfg).unwrapped

        policy, _ = create_policy(args_cli_full)

        ep_successes = 0

        print(f"\n--- Episode {ep_idx+1}/{len(episodes)}: {ep_id} ---")
        if ep_desc:
            print(f"    {ep_desc}")

        for trial in range(num_trials):
            env.reset()
            policy.reset()
            trial_success = False
            t_start = time.time()

            with torch.inference_mode():
                for step in range(max_steps):
                    obs = env.obs_buf
                    actions = policy.get_action(env, obs)
                    env.step(actions)

                    if _check_success(success_term, env):
                        trial_success = True
                        break

            elapsed = time.time() - t_start
            status = "SUCCESS" if trial_success else "FAIL"
            print(f"    Trial {trial+1}/{num_trials}: {status} ({step+1} steps, {elapsed:.1f}s)")

            if trial_success:
                ep_successes += 1

        ep_rate = ep_successes / num_trials
        total_successes += ep_successes
        total_trials += num_trials

        results["per_episode"].append({
            "id": ep_id,
            "description": ep_desc,
            "trials": num_trials,
            "successes": ep_successes,
            "success_rate": round(ep_rate, 3),
        })

        print(f"    Episode result: {ep_successes}/{num_trials} = {ep_rate:.1%}")

        env.close()

    overall_rate = total_successes / total_trials if total_trials > 0 else 0.0
    results["summary"] = {
        "total_episodes": len(episodes),
        "total_trials": total_trials,
        "total_successes": total_successes,
        "success_rate": round(overall_rate, 3),
    }

    print("\n" + "=" * 60)
    print(f"  OVERALL: {total_successes}/{total_trials} = {overall_rate:.1%}")
    print("=" * 60)

    output_path = args_cli_full.output
    if output_path is None:
        os.makedirs("results", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/eval_{ts}.yaml"

    with open(output_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
    print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
    simulation_app.close()
