# Data Pipeline

ManipBench provides a complete workflow for collecting, validating, and converting manipulation demonstration data.

```
┌──────────┐     ┌──────────┐     ┌───────────┐     ┌────────────┐
│  Record   │ ──→ │  Replay  │ ──→ │  Validate │ ──→ │  Convert   │
│ (teleop)  │     │  (HDF5)  │     │  (visual) │     │ (LeRobot)  │
└──────────┘     └──────────┘     └───────────┘     └────────────┘
                       │
                       ▼
              ┌──────────────┐     ┌────────────────┐
              │  Annotate    │ ──→ │  Generate Data  │
              │  (subtasks)  │     │  (Mimic)        │
              └──────────────┘     └────────────────┘
```

---

## Step 1: Record Demonstrations

### Keyboard Teleop

```bash
python -m manip_bench.scripts.record \
    ex001arm_cvpr_scene_put_blocks_to_color \
    --enable_cameras
```

### VR Teleop

```bash
python -m manip_bench.scripts.record \
    ex001arm_cvpr_scene_pick_and_place \
    --teleop_device ex001arm_openxr_bimanual \
    --enable_cameras
```

### Real Arm (WebSocket Remote)

```bash
python -m manip_bench.scripts.teleop_real_arm \
    ex001arm_cvpr_scene_pick_and_place \
    --teleop_device ex001arm_ws_remote \
    --enable_cameras
```

**Output**: HDF5 file in `./recordings/` containing:
- Joint positions, velocities, and targets
- End-effector poses (position + quaternion) for both arms
- Gripper states
- Camera images (if `--enable_cameras`)
- Episode metadata (success, task config, timestamps)

---

## Step 2: Replay & Validate

Replay recorded demonstrations to visually verify quality:

```bash
python -m manip_bench.scripts.replay \
    ex001arm_cvpr_scene_pick_and_place \
    --replay_file_path ./recordings/demo_001.hdf5 \
    --enable_cameras
```

---

## Step 3: Convert to Training Format

### HDF5 → LeRobot

```bash
python -m manip_bench.scripts.convert2lerobot \
    --input ./recordings/demo_001.hdf5 \
    --output ./datasets/lerobot_demo
```

### Batch HDF5 → LeRobot

```bash
python -m manip_bench.scripts.hdf5_to_lerobot \
    --input_dir ./recordings/ \
    --output_dir ./datasets/
```

---

## Step 4: Mimic Data Generation (Automated)

Arena provides an **Isaac Lab Mimic** integration that can automatically generate
large amounts of demonstration data from a small number of human-annotated source
demos. ManipBench wraps this pipeline with its own CLI so custom environments and
assets are registered automatically.

### 4a. Annotate Subtask Boundaries

Before Mimic can synthesize new trajectories, each source demo must be annotated
with **subtask boundary markers** (e.g., "grasp completed", "place completed").

```bash
# Auto mode — uses env.get_subtask_term_signals() if the task implements it
python -m manip_bench.scripts.annotate_demos \
    --input_file ./datasets/source_demos.hdf5 \
    --output_file ./datasets/annotated_demos.hdf5 \
    --auto \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm --background cvpr_background --object cracker_box

# Manual mode — press 'S' to mark subtask boundaries during replay
python -m manip_bench.scripts.annotate_demos \
    --input_file ./datasets/source_demos.hdf5 \
    --output_file ./datasets/annotated_demos.hdf5 \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm --background cvpr_background --object cracker_box
```

Keyboard controls in manual mode:

| Key | Function |
|-----|----------|
| N   | Play / resume |
| B   | Pause |
| S   | Mark subtask boundary |
| Q   | Skip episode |

### 4b. Generate Dataset with Mimic

Once annotations exist, Mimic synthesizes new episodes by perturbing object poses
and replanning trajectories:

```bash
python -m manip_bench.scripts.generate_dataset \
    --generation_num_trials 200 \
    --input_file ./datasets/annotated_demos.hdf5 \
    --output_file ./datasets/generated_200.hdf5 \
    --num_envs 4 \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm --background cvpr_background --object cracker_box
```

Key flags:

| Flag | Description |
|------|-------------|
| `--generation_num_trials` | Total number of episodes to generate |
| `--input_file` | Path to the annotated source dataset |
| `--output_file` | Where to write generated episodes |
| `--num_envs` | Parallel environments (faster generation) |
| `--pause_subtask` | Debug: pause after each subtask |
| `--enable_pinocchio` | Enable Pink IK solver (needed by some embodiments) |
| `--enable_cameras` | Record camera observations during generation |

### 4c. Using Arena's Scripts Directly

You can also invoke Arena's original scripts with ManipBench environments via the
`--environment` flag:

```bash
python -m isaaclab_arena.scripts.generate_dataset \
    --environment "manip_bench.envs.cvpr_pick_and_place:Ex001ArmCvprScenePickAndPlaceEnvironment" \
    --generation_num_trials 100 \
    --input_file ./datasets/annotated.hdf5 \
    --output_file ./datasets/output.hdf5 \
    ex001arm_cvpr_scene_pick_and_place \
    --embodiment ex001arm --background cvpr_background --object cracker_box
```

> The `--environment module_path:ClassName` mechanism dynamically imports and
> registers the environment. ManipBench's `get_env()` method automatically calls
> `import manip_bench.extensions` to ensure all custom assets are available.

---

## Data Format

### HDF5 Structure

```
demo.hdf5
├── data/
│   ├── ep_0000/
│   │   ├── actions          # (T, 14) — [left_ee(6) + left_grip(1) + right_ee(6) + right_grip(1)]
│   │   ├── observations/
│   │   │   ├── joint_pos    # (T, 14) — all joint positions
│   │   │   ├── joint_vel    # (T, 14) — all joint velocities
│   │   │   ├── eef_pos      # (T, 3)  — left end-effector position
│   │   │   ├── eef_quat     # (T, 4)  — left end-effector quaternion
│   │   │   ├── right_eef_pos
│   │   │   ├── right_eef_quat
│   │   │   ├── gripper_pos  # (T, 1)  — left gripper opening
│   │   │   ├── right_gripper_pos
│   │   │   ├── left_wrist_cam   # (T, H, W, 3) — RGB
│   │   │   ├── right_wrist_cam
│   │   │   └── head_cam
│   │   └── initial_state/
│   └── ep_0001/
│       └── ...
└── metadata/
    ├── task_name
    ├── embodiment
    └── success
```

### Action Space Convention

The 14D unified action vector:

| Indices | Content | Range |
|---------|---------|-------|
| 0–2 | Left arm position delta (x, y, z) | continuous |
| 3–5 | Left arm rotation delta (roll, pitch, yaw) | continuous |
| 6 | Left gripper command | -1 (close) / +1 (open) |
| 7–9 | Right arm position delta (x, y, z) | continuous |
| 10–12 | Right arm rotation delta (roll, pitch, yaw) | continuous |
| 13 | Right gripper command | -1 (close) / +1 (open) |
