"""ManipBench policy runner CLI — argument parsing and policy creation.

Supports: zero_action, replay, x2robot_closedloop policy types.
"""

import argparse

from manip_bench.scripts.cli import get_manip_bench_cli_parser
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy


def add_zero_action_arguments(parser: argparse.ArgumentParser) -> None:
    zero_action_group = parser.add_argument_group("Zero Action Policy")
    zero_action_group.add_argument(
        "--num_steps", type=int, default=100,
        help="Number of steps to run the policy for",
    )


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    replay_group = parser.add_argument_group("Replay Action Policy")
    replay_group.add_argument(
        "--replay_file_path", type=str,
        help="Path to the HDF5 file containing the episode",
    )
    replay_group.add_argument(
        "--episode_name", type=str, default=None,
        help="Name of the episode to replay (default: first episode)",
    )


def add_x2robot_closedloop_arguments(parser: argparse.ArgumentParser) -> None:
    x2robot_group = parser.add_argument_group("X2Robot Closedloop Policy")
    x2robot_group.add_argument(
        "--x2robot_server_address", type=str, default="localhost",
        help="Address of the X2Robot inference server",
    )
    x2robot_group.add_argument(
        "--x2robot_server_port", type=int, default=8000,
        help="Port of the X2Robot inference server",
    )
    x2robot_group.add_argument(
        "--x2robot_instruction", type=str, default="pick up the object",
        help="Task instruction for the X2Robot model",
    )
    x2robot_group.add_argument(
        "--x2robot_control_mode", type=str, choices=["end_pose", "joints"],
        default="end_pose", help="Control mode: 'end_pose' or 'joints'",
    )


def setup_policy_argument_parser(
    args_parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Build a complete CLI parser with environment + policy arguments."""
    args_parser = get_manip_bench_cli_parser(args_parser)

    args_parser.add_argument(
        "--policy_type", type=str, required=True,
        choices=["zero_action", "replay", "x2robot_closedloop"],
        help="Type of policy to use",
    )

    add_zero_action_arguments(args_parser)
    add_replay_arguments(args_parser)
    add_x2robot_closedloop_arguments(args_parser)

    parsed_args = args_parser.parse_args()

    if parsed_args.policy_type == "replay" and parsed_args.replay_file_path is None:
        raise ValueError("--replay_file_path is required when using --policy_type replay")
    return args_parser


def create_policy(args: argparse.Namespace) -> tuple[PolicyBase, int]:
    """Create the appropriate policy based on the arguments."""
    if args.policy_type == "replay":
        policy = ReplayActionPolicy(args.replay_file_path, args.episode_name)
        num_steps = len(policy)
    elif args.policy_type == "zero_action":
        policy = ZeroActionPolicy()
        num_steps = args.num_steps
    elif args.policy_type == "x2robot_closedloop":
        from manip_bench.extensions.policies.closedloop_policy import (
            X2RobotClosedloopPolicy,
            X2RobotPolicyConfig,
        )
        config = X2RobotPolicyConfig(
            model_address=args.x2robot_server_address,
            model_port=args.x2robot_server_port,
            instruction=args.x2robot_instruction,
            control_mode=args.x2robot_control_mode,
        )
        policy = X2RobotClosedloopPolicy(config, num_envs=args.num_envs, device=args.device)
        num_steps = args.num_steps
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")
    return policy, num_steps
