# Getting Started

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| GPU | NVIDIA RTX 3060 | NVIDIA RTX 4090 |
| VRAM | 8 GB | 16+ GB |
| CUDA | 12.6 | 12.8 |
| Driver | 560+ | 570+ |
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32+ GB |

## Installation

### Step 1: Clone Repository

```bash
git clone --recurse-submodules https://github.com/<your-org>/manip-bench.git
cd manip-bench
git lfs pull
```

If you forgot `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### Step 2: Create Conda Environment (Recommended)

```bash
conda create -n manip_bench python=3.11 -y
conda activate manip_bench
```

### Step 3: Run Installer

```bash
bash install.sh
```

This script performs the following in order:
1. Installs PyTorch 2.7 with CUDA 12.8
2. Installs Isaac Sim 5.0.0
3. Initializes git submodules
4. Installs Isaac Lab from submodule
5. Installs Isaac Lab Arena from submodule
6. Installs ManipBench in editable mode

### Step 4: Verify Installation

```bash
python -c "import manip_bench; print(f'ManipBench {manip_bench.__version__}')"
python -c "import isaaclab_arena; print('Isaac Lab Arena OK')"
```

## First Run

Launch a simple environment with zero-action policy (robot stays still):

```bash
python -m manip_bench.scripts.teleop_keyboard \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm \
    --num_envs 1
```

You should see the Isaac Sim viewport with the EX001Arm robot in a 3DGS tabletop scene.

## Manual Installation (Advanced)

If you prefer to install components separately:

```bash
# PyTorch
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# Isaac Lab (from submodule)
cd third_party/IsaacLab-Arena/submodules/IsaacLab
bash isaaclab.sh --install
cd ../..

# Isaac Lab Arena
pip install -e .
cd ../..

# ManipBench
pip install -e .

# Optional: LeRobot data format support
pip install -e ".[lerobot]"

# Optional: WebSocket remote teleoperation
pip install -e ".[remote]"
```

## Troubleshooting

### Isaac Sim fails to start

Ensure your NVIDIA driver version matches the CUDA requirement. Check with:

```bash
nvidia-smi
```

### Import errors for `isaaclab` or `isaaclab_arena`

Make sure submodules are initialized and installed:

```bash
git submodule update --init --recursive
cd third_party/IsaacLab-Arena/submodules/IsaacLab && bash isaaclab.sh --install
cd ../.. && pip install -e .
```

### Git LFS files not downloaded

```bash
git lfs install
git lfs pull
```
