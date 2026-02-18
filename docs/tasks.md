# Tasks

ManipBench tasks define the **objective, success criteria, and MDP configuration** for manipulation experiments. All tasks inherit from `isaaclab_arena.tasks.task_base.TaskBase`.

---

## Pick and Place

**Source**: `isaaclab_arena.tasks.pick_and_place_task.PickAndPlaceTask` (provided by Arena)

| Property | Value |
|----------|-------|
| Objects | 1 target object + 1 destination |
| Success | Object within distance threshold of destination AND stationary |
| Action Space | Bimanual: 7D × 2 (6D EE delta + 1D gripper) |

### Environments using this task

- `ex001arm_cvpr_scene_pick_and_place` — 3DGS tabletop scene
- `ex001arm_kitchen_pick_and_place` — Kitchen scene

---

## Put Blocks to Color

**Source**: `manip_bench.extensions.tasks.put_blocks_to_color`

A structured bimanual sorting task: pick up colored bricks and place them onto matching color paper destinations.

| Property | Value |
|----------|-------|
| Objects | 3 bricks (yellow, green, red) + 3 papers (yellow, green, pink) |
| Success | All 3 bricks on their matching papers, within threshold, stationary |
| Randomization | Brick positions randomized on reset within configurable range |
| Action Space | Bimanual: 7D × 2 (6D EE delta + 1D gripper) |

### Success Criteria (per brick)

- X distance < 0.12 m
- Y distance < 0.16 m
- Z distance < 0.06 m
- Velocity < 0.5 m/s

### Variants

| Class | Bricks | Description |
|-------|--------|-------------|
| `ThreeBlocksToColorTask` | 3 | Standard three-brick sorting |

### Environments using this task

- `ex001arm_cvpr_scene_put_blocks_to_color` — 3DGS tabletop with randomized brick positions

---

## Adding New Tasks

See [Architecture — Extension封装规范](architecture.md#33-task任务定义) for the detailed pattern.

Quick summary:

1. Create `manip_bench/extensions/tasks/my_new_task.py`
2. Inherit from `TaskBase`
3. Define `get_object_cfgs()`, `get_termination_cfg()`, and success function
4. Register any new objects in `extensions/scenes/objects.py`
5. Create an environment composition in `manip_bench/envs/`
