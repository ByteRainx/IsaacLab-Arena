"""ManipBench: Run an environment with a specified policy.

Usage examples:

    # Zero-action (robot stays still, for visualization)
    python -m manip_bench.scripts.run_env --policy_type zero_action \
        ex001arm_cvpr_scene_pick_and_place --embodiment ex001arm

    # Replay a recorded demonstration
    python -m manip_bench.scripts.run_env --policy_type replay \
        --replay_file_path ./recordings/demo.hdf5 \
        ex001arm_cvpr_scene_pick_and_place

    # X2Robot closed-loop inference
    python -m manip_bench.scripts.run_env --policy_type x2robot_closedloop \
        --x2robot_server_address localhost --x2robot_server_port 8000 \
        ex001arm_cvpr_scene_pick_and_place --enable_cameras
"""

import numpy as np
import random
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from manip_bench.scripts.policy_runner_cli import create_policy, setup_policy_argument_parser
from manip_bench.scripts.cli import get_env_builder_from_cli


def main():
    args_parser = get_isaaclab_arena_cli_parser()
    args_cli, unknown = args_parser.parse_known_args()

    # Must import after AppLauncher — simulation engine startup
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext

    with SimulationAppContext(args_cli):
        # Register all ManipBench extensions before building the environment
        import manip_bench.extensions  # noqa: F401

        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()

        arena_builder = get_env_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()

        policy, num_steps = create_policy(args_cli)

        from isaaclab_arena.metrics.metrics import compute_metrics

        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    env_ids = (terminated | truncated).nonzero().flatten()
                    print(f"Resetting: terminated={terminated.nonzero().flatten()}, truncated={truncated.nonzero().flatten()}")
                    policy.reset(env_ids=env_ids)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")
        env.close()


if __name__ == "__main__":
    main()
