#!/bin/bash
# ManipBench — Launch keyboard teleoperation
#
# Usage:
#   bash teleop.sh                                    # Default: CVPR pick-and-place
#   bash teleop.sh ex001arm_cvpr_scene_put_blocks_to_color  # Color sorting
#
# Add --enable_cameras to record camera observations.

ENV="${1:-ex001arm_cvpr_scene_pick_and_place}"
shift 2>/dev/null || true

python -m manip_bench.scripts.teleop_keyboard \
    "$ENV" \
    --embodiment ex001arm \
    "$@"
