#!/bin/bash
# ManipBench — Replay and evaluate a recorded demonstration
#
# Usage:
#   bash eval.sh <env_name> <hdf5_path>
#
# Example:
#   bash eval.sh ex001arm_cvpr_scene_pick_and_place ./recordings/demo.hdf5

if [ $# -lt 2 ]; then
    echo "Usage: bash eval.sh <env_name> <hdf5_path>"
    echo "Example: bash eval.sh ex001arm_cvpr_scene_pick_and_place ./recordings/demo.hdf5"
    exit 1
fi

ENV="$1"
HDF5_PATH="$2"
shift 2

python -m manip_bench.scripts.replay \
    "$ENV" \
    --replay_file_path "$HDF5_PATH" \
    --enable_cameras \
    "$@"
