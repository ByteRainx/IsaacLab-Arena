# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone EX001Arm VR teleop example using cvpr_assets scene."""

import argparse

from isaaclab.app import AppLauncher
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser


def _teleop_device_requires_xr(device_name: str | None) -> bool:
    if not device_name:
        return False
    name = device_name.lower()
    return any(token in name for token in ("handtracking", "openxr")) or name in {
        "avp_handtracking",
        "ex001arm_openxr_bimanual",
    }


def main() -> None:
    parser = get_isaaclab_arena_cli_parser()
    parser.add_argument("--teleop_device", type=str, default="ex001arm_openxr_bimanual")
    parser.add_argument("--background", type=str, default="cvpr_assets")
    parser.add_argument("--object", type=str, default="cracker_box")
    parser.add_argument(
        "--object_xyz",
        type=float,
        nargs=3,
        default=(-1.41099, -0.17982, -0.10477),
        help="Initial XYZ of the pick-up object in the background frame.",
    )
    parser.add_argument(
        "--destination_xyz",
        type=float,
        nargs=3,
        default=(0.9, 0.0, 0.05),
        help="Initial XYZ of the destination marker in the background frame.",
    )
    parser.add_argument("--embodiment", type=str, default="ex001arm")
    parser.add_argument(
        "--enable_pinocchio",
        action="store_true",
        default=False,
        help="Enable Pinocchio.",
    )
    
    args_cli = parser.parse_args()

    app_launcher_args = vars(args_cli)

    if _teleop_device_requires_xr(args_cli.teleop_device):
        app_launcher_args["xr"] = True
        setattr(args_cli, "xr", True)

    if args_cli.enable_pinocchio:
        import pinocchio  # noqa: F401

    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

    import torch

    import omni.log
    from isaaclab.devices.teleop_device_factory import create_teleop_device
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.examples.example_environments.kitchen_pick_and_place_ex001arm_cvpr_assets_environment import (
        KitchenPickAndPlaceEx001ArmCvprAssetsEnvironment,
    )

    example_env = KitchenPickAndPlaceEx001ArmCvprAssetsEnvironment()
    arena_env = example_env.get_env(args_cli)
    env_builder = ArenaEnvBuilder(arena_env, args_cli)
    env_name, env_cfg = env_builder.build_registered()

    if args_cli.xr:
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env = None
    try:
        env = __import__("gymnasium").make(env_name, cfg=env_cfg).unwrapped
    except Exception as exc:
        omni.log.error(f"Failed to create environment: {exc}")
        simulation_app.close()
        return

    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, {})
        else:
            omni.log.error(f"No teleop device '{args_cli.teleop_device}' in env config.")
    except Exception as exc:
        omni.log.error(f"Failed to create teleop device: {exc}")

    if teleop_interface is None:
        env.close()
        simulation_app.close()
        return

    env.reset()
    teleop_interface.reset()

    print("Teleoperation started.")

    while simulation_app.is_running():
        with torch.inference_mode():
            action = teleop_interface.advance()
            actions = action.repeat(env.num_envs, 1)
            env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
