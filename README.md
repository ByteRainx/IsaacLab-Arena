# ManipBench

<div align="center">

**A bimanual manipulation simulation benchmark built on Isaac Lab Arena.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.0-orange.svg)](https://developer.nvidia.com/isaac-sim)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

[Architecture](docs/architecture.md) · [Getting Started](docs/getting_started.md) · [Tasks](docs/tasks.md) · [Teleop Guide](docs/teleop_guide.md)

</div>

---

## Overview

**ManipBench** is a simulation benchmark for **bimanual robot manipulation**, built as an extension layer on top of [Isaac Lab Arena](https://github.com/isaac-sim/IsaacLab-Arena). It provides ready-to-use embodiments, tasks, scenes, and data tools for research in dual-arm coordination, teleoperation, and imitation learning.

### Key Features

- **Bimanual Manipulation Focus** — Purpose-built for dual-arm research with EX001Arm (6-DOF × 2) and OpenArm (7-DOF × 2) robots.
- **Extension-over-Framework Architecture** — Zero modification to Isaac Lab Arena source code. All customizations live in a clean extension layer. See [Architecture](docs/architecture.md).
- **Contact-Aware Grasping** — Custom `ContactLimitedGripperAction` monitors finger contact forces to prevent penetration and enable realistic grasps.
- **Multiple Teleoperation Modes** — Keyboard, VR (OpenXR hand tracking), and real robot arm (WebSocket remote) teleoperation.
- **Complete Data Pipeline** — Record → Replay → Validate → Convert (HDF5 / LeRobot format) workflow for imitation learning.
- **3DGS-Reconstructed Scenes** — Photorealistic tabletop backgrounds from 3D Gaussian Splatting reconstruction.
- **Color-Sorting Benchmark** — Structured task with randomized initial conditions and automatic success detection.

## Quick Start

### Prerequisites

- Linux with NVIDIA GPU (RTX recommended)
- Python 3.10+
- CUDA 12.8

### Installation

```bash
# Clone with LFS
git clone --recurse-submodules https://github.com/<your-org>/manip-bench.git
cd manip-bench
git lfs pull

# One-line install (Isaac Sim + Isaac Lab + Arena + ManipBench)
bash install.sh
```

### Run an Environment

```bash
# CVPR tabletop pick-and-place with keyboard teleop
python -m manip_bench.scripts.teleop_keyboard \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm \
    --teleop_device keyboard \
    --enable_cameras

# Color-sorting benchmark
python -m manip_bench.scripts.record \
    ex001arm_cvpr_scene_put_blocks_to_color \
    --enable_cameras
```

### Record & Replay Demonstrations

```bash
# Record
python -m manip_bench.scripts.record \
    ex001arm_cvpr_scene_pick_and_place \
    --enable_cameras

# Replay
python -m manip_bench.scripts.replay \
    ex001arm_cvpr_scene_pick_and_place \
    --replay_file_path ./recordings/demo.hdf5

# Convert to LeRobot format
python -m manip_bench.scripts.convert2lerobot \
    --input ./recordings/demo.hdf5 \
    --output ./datasets/lerobot_demo
```

## Project Structure

```
manip-bench/
├── manip_bench/
│   ├── extensions/           # Extension components (see Architecture doc)
│   │   ├── embodiments/      #   Robot definitions (EX001Arm, OpenArm)
│   │   ├── tasks/            #   Task logic (color sorting, pick-and-place)
│   │   ├── scenes/           #   Background & object asset registrations
│   │   ├── policies/         #   Policy implementations (closed-loop inference)
│   │   └── devices/          #   Teleop devices (VR, WebSocket remote)
│   ├── envs/                 # Environment compositions (Embodiment + Scene + Task)
│   ├── scripts/              # User-facing entry points
│   └── utils/                # Shared utilities (path resolution, etc.)
│
├── assets/                   # USD models, textures, 3DGS data (git-lfs)
├── configs/                  # YAML configuration files
├── tools/                    # Standalone utilities (ROS bridge, debug client)
├── docs/                     # Documentation
└── third_party/
    └── IsaacLab-Arena/       # Official Isaac Lab Arena (git submodule, unmodified)
```

## Supported Robots

| Robot | DOF | Type | Description |
|-------|-----|------|-------------|
| EX001Arm | 6+1 × 2 | Bimanual | 6-DOF arms with parallel grippers, contact sensors, wrist & head cameras |
| OpenArm | 7+1 × 2 | Bimanual | 7-DOF arms with finger grippers |

## Available Tasks

| Task | Description | Success Criterion |
|------|-------------|-------------------|
| Pick and Place | Grasp an object and place it on a target | Object within threshold of destination |
| Put Blocks to Color | Sort 3 colored bricks onto matching papers | All bricks on correct papers, stationary |

## Available Environments

| Environment ID | Scene | Task | Robot |
|----------------|-------|------|-------|
| `ex001arm_cvpr_scene_pick_and_place` | 3DGS tabletop | Pick and Place | EX001Arm |
| `ex001arm_cvpr_scene_put_blocks_to_color` | 3DGS tabletop | Color Sorting | EX001Arm |
| `ex001arm_kitchen_pick_and_place` | Kitchen | Pick and Place | EX001Arm |

## Teleoperation

ManipBench supports three teleoperation modes:

| Mode | Device | Script |
|------|--------|--------|
| Keyboard | Standard keyboard | `manip_bench.scripts.teleop_keyboard` |
| VR | OpenXR (Vision Pro, Quest, PICO) | Built into `record` script with `--teleop_device` |
| Real Arm | ARX robotic arm via WebSocket | `manip_bench.scripts.teleop_real_arm` |

See [Teleop Guide](docs/teleop_guide.md) for detailed setup instructions.

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Code organization, extension patterns, packaging conventions |
| [Getting Started](docs/getting_started.md) | Detailed installation and first-run guide |
| [Tasks](docs/tasks.md) | Task specifications, success criteria, configuration |
| [Teleop Guide](docs/teleop_guide.md) | Teleoperation setup for keyboard, VR, and real arm |
| [Data Pipeline](docs/data_pipeline.md) | Record → replay → convert workflow |

## Citation

```bibtex
@software{manip_bench,
  title = {ManipBench: A Bimanual Manipulation Simulation Benchmark},
  url = {https://github.com/<your-org>/manip-bench},
  year = {2025},
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).
