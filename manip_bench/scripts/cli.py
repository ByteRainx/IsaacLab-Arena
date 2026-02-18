"""ManipBench CLI utilities.

Provides argument parsing and environment lookup for ManipBench scripts.

Environments can be selected in two ways:
  * **By name** (subcommand):  ``... cvpr_pick_and_place --embodiment ex001arm``
  * **By YAML** (``--env_config``):  ``... --env_config configs/envs/custom.yaml``
"""

import argparse

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

from manip_bench.envs import ENVIRONMENTS


def add_manip_bench_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ManipBench environment subcommands and --env_config flag."""
    parser.add_argument(
        "--env_config", type=str, default=None,
        help="Path to a YAML env config (overrides environment subcommand).",
    )
    subparsers = parser.add_subparsers(
        dest="example_environment", required=False,
        help="ManipBench environment to run (or use --env_config instead)",
    )
    for env_entry in ENVIRONMENTS.values():
        name = env_entry.name if hasattr(env_entry, "name") else getattr(env_entry, "__name__", str(env_entry))
        subparser = subparsers.add_parser(name)
        if hasattr(env_entry, "add_cli_args"):
            env_entry.add_cli_args(subparser)
        elif isinstance(env_entry, type) and hasattr(env_entry, "add_cli_args"):
            env_entry.add_cli_args(subparser)
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
    """Build an ArenaEnvBuilder from parsed CLI arguments.

    If ``--env_config`` is provided, loads the environment from that YAML.
    Otherwise falls back to the subcommand-selected environment.
    """
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    env_config = getattr(args_cli, "env_config", None)
    if env_config:
        from manip_bench.envs import load_env_from_yaml
        env_instance = load_env_from_yaml(env_config)
        return ArenaEnvBuilder(env_instance.get_env(args_cli), args_cli)

    assert hasattr(args_cli, "example_environment") and args_cli.example_environment, (
        "Either --env_config or an environment subcommand must be specified."
    )
    assert args_cli.example_environment in ENVIRONMENTS, (
        f"Unknown environment: {args_cli.example_environment}"
    )
    env_entry = ENVIRONMENTS[args_cli.example_environment]
    if isinstance(env_entry, type):
        env_instance = env_entry()
    else:
        env_instance = env_entry
    return ArenaEnvBuilder(env_instance.get_env(args_cli), args_cli)


# Aliases for backward compatibility
add_example_environments_cli_args = add_manip_bench_cli_args
get_arena_builder_from_cli = get_env_builder_from_cli
