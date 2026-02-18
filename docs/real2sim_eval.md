# Real2Sim Evaluation Guide

ManipBench provides a structured workflow for evaluating manipulation policies
in simulation under conditions that mirror the real world.

## Architecture Overview

```
┌─────────────────────┐     ┌──────────────────────────┐
│  Real World          │     │  Simulation               │
│                      │     │                           │
│  3DGS scan ──────────┼────→│  Background (visual)      │
│  Robot calibration ──┼────→│  Robot position (fixed)   │
│  Object poses ───────┼────→│  Object positions (exact) │
│                      │     │                           │
│  Policy (trained) ───┼────→│  Run policy → metrics     │
└─────────────────────┘     └──────────────────────────┘
```

## Key Concepts

### Scene Config (YAML)

A scene config describes the **physical setup** of an environment — the
calibrated positions of background, robot, lighting, and default object poses.

```yaml
# configs/envs/cvpr_pick_and_place.yaml
env_id: cvpr_pick_and_place
scene:
  background:
    asset: cvpr_background
    pose: {position: [-1.0, -0.562, 0.0], rotation: [0.04457, 0.0, 0.0, -0.999]}
  robot:
    asset: ex001arm
    pose: {position: [-0.51676, -0.25918, -0.58061], rotation: [1.0, 0.0, 0.0, 0.0]}
task:
  type: pick_and_place
  target:
    asset: cracker_box
    pose: {position: [0.3, 0.0, -0.2]}
    workspace_bounds:       # used only in training mode
      x: [0.15, 0.45]
      y: [-0.25, 0.25]
```

### Eval Suite (YAML)

An eval suite defines a set of **episodes** — each with specific object poses
that represent conditions captured from the real world (or designed manually).

```yaml
# configs/eval/cvpr_pick_and_place_suite.yaml
name: "CVPR Pick-and-Place Evaluation v1"
env_config: configs/envs/cvpr_pick_and_place.yaml
num_trials: 5
max_steps: 500
episodes:
  - id: nominal
  - id: left_offset
    overrides:
      target: {position: [0.2, -0.1, -0.2]}
```

## Usage

### Running an Eval Suite

```bash
# Zero-action baseline (robot stays still)
python -m manip_bench.scripts.eval_suite \
    --suite configs/eval/cvpr_pick_and_place_suite.yaml \
    --policy_type zero_action

# X2Robot closed-loop inference
python -m manip_bench.scripts.eval_suite \
    --suite configs/eval/cvpr_pick_and_place_suite.yaml \
    --policy_type x2robot_closedloop \
    --x2robot_server_address localhost --x2robot_server_port 8000

# Custom output path
python -m manip_bench.scripts.eval_suite \
    --suite configs/eval/cvpr_pick_and_place_suite.yaml \
    --policy_type replay --replay_file_path ./demos/demo.hdf5 \
    --output results/my_eval.yaml
```

### Using YAML Configs Directly (not just eval)

You can use `--env_config` with any ManipBench script:

```bash
# Record with a YAML-defined environment
python -m manip_bench.scripts.record \
    --env_config configs/envs/cvpr_pick_and_place.yaml \
    --teleop_device keyboard

# Run policy with YAML config
python -m manip_bench.scripts.run_env \
    --env_config configs/envs/cvpr_color_sorting.yaml \
    --policy_type zero_action
```

### Results Format

The eval runner writes a YAML results file:

```yaml
name: "CVPR Pick-and-Place Evaluation v1"
timestamp: "2026-02-19T12:00:00"
summary:
  total_episodes: 8
  total_trials: 40
  total_successes: 28
  success_rate: 0.7
per_episode:
  - id: nominal
    trials: 5
    successes: 4
    success_rate: 0.8
  - id: left_offset
    trials: 5
    successes: 3
    success_rate: 0.6
```

## Real2Sim Workflow

### Step 1: Capture Real-World Scene State

Before a real-world evaluation episode, capture object poses:

- **AprilTag method**: Place markers on objects, capture one image, run
  pose estimation. Accuracy: ~1mm. Remove markers before execution.
- **Model-based method**: Use FoundationPose / MegaPose with your USD
  models to estimate 6DoF poses from RGB. No markers needed.
- **Manual**: Measure with a ruler for small eval sets.

### Step 2: Create Eval Episodes

Convert captured poses into eval suite entries:

```yaml
episodes:
  - id: real_session_001_ep01
    source: apriltag
    timestamp: "2026-02-19T10:30:00"
    overrides:
      target:
        position: [0.28, 0.03, -0.19]
        rotation: [0.98, 0.0, 0.0, 0.19]
```

### Step 3: Run in Sim

```bash
python -m manip_bench.scripts.eval_suite \
    --suite configs/eval/real_session_001.yaml \
    --policy_type x2robot_closedloop \
    --x2robot_server_address localhost
```

### Step 4: Compare

Compare sim success rates with real-world results. If they correlate
(sim-good ≈ real-good), your Real2Sim pipeline is working.

## Creating New Scene Configs

To add a new real-world scene:

1. **Reconstruct** the scene with 3DGS → register as a background asset
2. **Calibrate** the robot position relative to the reconstruction
3. **Write a YAML** with the calibrated poses:

```yaml
env_id: my_new_scene
scene:
  background:
    asset: my_3dgs_background
    pose: {position: [...], rotation: [...]}
  robot:
    asset: ex001arm
    pose: {position: [...], rotation: [...]}
task:
  type: pick_and_place
  target:
    asset: cracker_box
    pose: {position: [...]}
```

4. Drop the YAML into `configs/envs/` — it will be auto-discovered.
5. Write eval episodes for this scene.

No Python code needed.
