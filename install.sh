#!/bin/bash
# ManipBench installation script
set -e

echo "=========================================="
echo "  ManipBench Installer"
echo "=========================================="

pip install --upgrade pip
pip install uv

# Step 1: PyTorch (CUDA 12.8)
echo "[1/5] Installing PyTorch..."
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Step 2: Isaac Sim
echo "[2/5] Installing Isaac Sim 5.0.0..."
uv pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com

# Step 3: Submodules
echo "[3/5] Initializing submodules..."
git submodule update --init --recursive

# Step 4: Isaac Lab + Isaac Lab Arena
echo "[4/5] Installing Isaac Lab & Isaac Lab Arena..."
cd third_party/IsaacLab-Arena/submodules/IsaacLab
bash isaaclab.sh --install
cd ../..
uv pip install -e .
cd ../..

# Step 5: ManipBench
echo "[5/5] Installing ManipBench..."
uv pip install -e .

echo ""
echo "=========================================="
echo "  ManipBench installed successfully!"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  python -m manip_bench.scripts.run_env --help"
echo ""
