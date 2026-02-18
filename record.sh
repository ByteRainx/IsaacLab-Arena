#!/bin/bash
# ManipBench — Record demonstration data
#
# Usage:
#   bash record.sh                                    # Default: CVPR pick-and-place
#   bash record.sh ex001arm_cvpr_scene_put_blocks_to_color  # Color sorting
#
# Data is saved to ./recordings/ in HDF5 format.

ENV="${1:-ex001arm_cvpr_scene_pick_and_place}"
shift 2>/dev/null || true

python -m manip_bench.scripts.record \
    "$ENV" \
    --enable_cameras \
    "$@"
