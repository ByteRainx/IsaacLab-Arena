"""ManipBench CLI utilities.

Provides argument parsing and environment lookup for ManipBench scripts.
This replaces direct modification of Isaac Lab Arena's internal CLI by
injecting ManipBench environments through the ``--environment`` mechanism.
"""

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

from manip_bench.envs import ENVIRONMENTS


def add_manip_bench_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ManipBench environment subcommands to the CLI parser."""
    subparsers = parser.add_subparsers(
        dest="example_environment", required=True, help="ManipBench environment to run"
    )
    for env_class in ENVIRONMENTS.values():
        subparser = subparsers.add_parser(env_class.name)
        env_class.add_cli_args(subparser)
    return parser


def get_manip_bench_cli_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """Build a CLI parser with ManipBench environments."""
    if parser is None:
        parser = get_isaaclab_arena_cli_parser()
    parser = add_manip_bench_cli_args(parser)
    return parser


def get_env_builder_from_cli(args_cli: argparse.Namespace):
    """Build an ArenaEnvBuilder from parsed CLI arguments."""
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    assert hasattr(args_cli, "example_environment"), "Environment must be specified"
    assert args_cli.example_environment in ENVIRONMENTS, (
        f"Unknown environment: {args_cli.example_environment}"
    )
    env_class = ENVIRONMENTS[args_cli.example_environment]()
    return ArenaEnvBuilder(env_class.get_env(args_cli), args_cli)


# Aliases for backward compatibility with scripts
add_example_environments_cli_args = add_manip_bench_cli_args
get_arena_builder_from_cli = get_env_builder_from_cli
