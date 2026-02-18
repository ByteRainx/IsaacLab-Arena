# ManipBench Assets

This directory contains USD models, textures, and 3DGS scene data used by ManipBench.
All binary files are tracked by **git-lfs**.

## Directory Structure

```
assets/
├── robots/
│   ├── ex001arm_bimanual/     # EX001Arm 6-DOF bimanual robot USD
│   └── openarm_bimanual/      # OpenArm 7-DOF bimanual robot USD
├── scenes/
│   └── scene-3dgs/            # 3DGS-reconstructed tabletop scenes
└── objects/
    └── hunyuan_assets/        # Hunyuan object assets (bricks, bowls, plates, etc.)
```

## Setup

After cloning the repository, pull LFS files:

```bash
git lfs pull
```
