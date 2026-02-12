# EX001Arm Mimic 数据增强 — 完整实施文档

## 1. 概述

本文档记录了为 **EX001Arm 双臂机器人**在 `put_blocks_to_color` 任务上实现 Isaac Lab Mimic 数据增强功能的全部代码变更和使用方法。

### 目标

从 **40 条 VR 遥操作演示**出发，通过 Mimic 数据增强流水线，生成 **数百至数千条**新的高质量演示数据，最终转换为 LeRobot v2.1 格式用于策略训练。

### 整体流程

```
40条VR演示 (.hdf5)
    │
    ▼
[merge_hdf5_demos.py] 合并为单文件
    │
    ▼
merged_demos.hdf5
    │
    ▼
[annotate_demos.py] 标注子任务边界
    │
    ▼
annotated_demos.hdf5
    │
    ▼
[generate_dataset.py] Mimic数据增强
    │
    ▼
mimic_generated.hdf5 (200~1000条新演示)
    │
    ▼
[hdf5_to_lerobot.py] 格式转换
    │
    ▼
LeRobot v2.1 数据集 (可直接训练)
```

---

## 2. 代码变更详情

### 2.1 新增 `EX001ArmMimicEnv` 类

**文件**: `isaaclab_arena/embodiments/ex001arm/ex001arm.py`

**变更内容**:

#### (a) 新增导入

```python
from collections.abc import Sequence
import torch
import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
```

#### (b) 在 `EX001ArmEmbodiment.__init__` 中注册 Mimic 环境

```python
self.mimic_env = EX001ArmMimicEnv
```

这一行让 `arena_env_builder.py` 在检测到 `--mimic` 参数时，能够找到并使用 `EX001ArmMimicEnv` 作为环境入口。

#### (c) 新增 `EX001ArmMimicEnv` 类

继承自 `ManagerBasedRLMimicEnv`，实现了以下方法：

| 方法 | 功能 | 说明 |
|------|------|------|
| `get_robot_eef_pose(eef_name, env_ids)` | 获取末端执行器位姿 | 支持 `"left"` 和 `"right"` 双臂，从 obs_buf 中读取 eef_pos/eef_quat |
| `target_eef_pose_to_action(...)` | 目标位姿 → 14D action | 对每个臂计算 delta_pos(3) + delta_rot(3) + gripper(1)，拼接为 14D |
| `action_to_target_eef_pose(action)` | 14D action → 目标位姿 | 逆变换，将 action 中的 delta 加到当前位姿上 |
| `actions_to_gripper_actions(actions)` | 提取夹爪动作 | `left` = `actions[:, 6:7]`, `right` = `actions[:, 13:14]` |
| `get_object_poses(env_ids)` | 获取场景物体位姿 | 调用 `get_rigid_and_articulated_object_poses` |
| `get_subtask_term_signals(env_ids)` | 子任务终止信号 | 返回空字典，建议使用手动标注 |

**Action 维度布局** (14D):

```
索引:  [0  1  2  3  4  5  6  7  8  9  10 11 12 13]
含义:  [----左臂IK----][左夹][----右臂IK----][右夹]
       pos_x/y/z  rot_x/y/z  grip  pos_x/y/z  rot_x/y/z  grip
```

辅助方法 `_compute_single_eef_action` 和 `_delta_to_target_pose` 封装了单臂的 delta 计算逻辑，避免代码重复。

---

### 2.2 扩展 `PutBlocksToColorMimicEnvCfg` 支持 ex001arm

**文件**: `isaaclab_arena/tasks/put_blocks_to_color_task.py`

**变更内容**: 在 `get_mimic_env_cfg()` 方法末尾的 embodiment 分支中，新增 `ex001arm` 分支：

```python
elif self.embodiment_name == "ex001arm":
    # 右臂为主操作臂
    self.subtask_configs["right"] = subtask_configs
    # 左臂保持静态
    left_subtask_configs = [SubTaskConfig(...)]
    self.subtask_configs["left"] = left_subtask_configs
```

**设计决策**：

- **右臂**配置为主操作臂，负责抓取和放置 block 的全部子任务序列（grasp_1 → place_1 → grasp_2 → ...）
- **左臂**配置为静态保持模式（无操作子任务，仅维持当前位置）
- `subtask_configs` 的 key (`"left"`, `"right"`) 与 `EX001ArmMimicEnv.get_robot_eef_pose` 的 `eef_name` 参数保持一致

---

### 2.3 新增 HDF5 合并脚本

**文件**: `isaaclab_arena/scripts/merge_hdf5_demos.py`

**用途**: 将多个单 episode 的 HDF5 文件合并为 Mimic 流水线期望的统一格式。

**输入**: `demos/vr_episode0.hdf5` ~ `demos/vr_episode39.hdf5`（每个文件内含 `data/demo_0`）

**输出**: `demos/merged_demos.hdf5`（内含 `data/demo_0` ~ `data/demo_39`）

**用法**:

```bash
python isaaclab_arena/scripts/merge_hdf5_demos.py \
    --input_dir demos \
    --output_file demos/merged_demos.hdf5

# 可选：限制合并数量
python isaaclab_arena/scripts/merge_hdf5_demos.py \
    --input_dir demos \
    --output_file demos/merged_demos.hdf5 \
    --max_episodes 10
```

**当前状态**: 已执行完毕，40 个 episode 成功合并为 `demos/merged_demos.hdf5`。

---

## 3. 运行 Mimic 流水线

> 以下步骤需要在有 GPU 和 Isaac Sim 环境的终端中执行。

### 3.1 Step 1 — 标注子任务边界 (Annotate)

使用**手动标注模式**（推荐，因为 `get_subtask_term_signals` 当前返回空信号）：

```bash
cd /home/xr/yan/new/IsaacLab-Arena

python isaaclab_arena/scripts/annotate_demos.py \
    --mimic \
    --input_file demos/merged_demos.hdf5 \
    --output_file demos/annotated_demos.hdf5 \
    --num_envs 1 \
    ex001arm_cvpr_scene_put_blocks_to_color
```

**手动标注操作键**:

| 按键 | 功能 |
|------|------|
| `N` | 播放演示 |
| `B` | 暂停 |
| `S` | 标记当前帧为子任务边界 |
| `Q` | 跳过当前 episode |

**标注要点**:

- 每个 episode 需要标记所有子任务的完成时刻
- 对于 `put_blocks_to_color` 任务（3 个 block），子任务序列为：
  - `grasp_1` → `place_1` → `grasp_2` → `place_2` → `grasp_3` → `place_3`
- 在每个子任务完成的那一帧按 `S`

标注完成后，HDF5 中会新增：

- `datagen_info/`: 每帧的物体位姿和 EEF 位姿
- `subtask_term_signals/`: 子任务完成信号

---

### 3.2 Step 2 — 生成新数据 (Generate)

```bash
python isaaclab_arena/scripts/generate_dataset.py \
    --mimic \
    --input_file demos/annotated_demos.hdf5 \
    --output_file demos/mimic_generated.hdf5 \
    --generation_num_trials 200 \
    --num_envs 1 \
    --headless \
    ex001arm_cvpr_scene_put_blocks_to_color
```

**参数说明**:

| 参数 | 说明 |
|------|------|
| `--generation_num_trials 200` | 尝试生成 200 条新演示（成功率通常 50-80%） |
| `--num_envs 1` | 并行环境数（GPU 内存允许时可增大） |
| `--headless` | 无头模式，不渲染画面（加速生成） |

生成原理：Mimic 从标注好的演示中提取子任务片段，在新的随机初始条件下，通过轨迹变换和插值重新拼接为完整的演示轨迹。

---

### 3.3 Step 3 — 转换为 LeRobot 格式

```bash
python isaaclab_arena/scripts/hdf5_to_lerobot.py \
    --input_dir demos \
    --output_dir demos/lerobot_mimic_dataset
```

转换后的数据集符合 LeRobot v2.1 规范，包含：

```
demos/lerobot_mimic_dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   └── episodes_stats.jsonl
├── data/
│   └── chunk-000/
│       └── episode_XXXXXX.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.leftImg/
        ├── observation.images.rightImg/
        └── observation.images.faceImg/
```

---

## 4. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `isaaclab_arena/embodiments/ex001arm/ex001arm.py` | **修改** | 新增 imports、`self.mimic_env` 赋值、`EX001ArmMimicEnv` 类 |
| `isaaclab_arena/tasks/put_blocks_to_color_task.py` | **修改** | 在 `PutBlocksToColorMimicEnvCfg` 中新增 `ex001arm` 分支 |
| `isaaclab_arena/scripts/merge_hdf5_demos.py` | **新增** | HDF5 演示文件合并脚本 |

## 5. 生成的数据文件

| 文件 | 说明 |
|------|------|
| `demos/merged_demos.hdf5` | 40 条 VR 演示合并文件（已生成） |
| `demos/annotated_demos.hdf5` | 标注后的演示文件（待 Step 1 生成） |
| `demos/mimic_generated.hdf5` | Mimic 增强后的新演示（待 Step 2 生成） |
| `demos/lerobot_mimic_dataset/` | LeRobot 格式训练数据集（待 Step 3 生成） |

---

## 6. 关键设计决策

### 双臂分工策略

当前采用 **右臂主操作 + 左臂静态保持** 的策略。原因：

1. Mimic 的子任务配置要求明确的操作臂和子任务序列
2. 从 VR 演示的实际操作模式来看，右臂通常承担主要的抓放操作
3. 如果后续发现双臂都需要参与操作，可以修改 `put_blocks_to_color_task.py` 中 `ex001arm` 分支，为左臂也配置完整的子任务链

### 标注方式选择

推荐**手动标注**（不加 `--auto` 参数）：

- 自动标注需要可靠的 `get_subtask_term_signals` 实现
- 当前返回空信号，自动标注不可用
- 手动标注虽然耗时，但标注质量更高，直接影响数据生成效果
- 40 条演示手动标注预计需要 1-2 小时

### EEF 命名约定

`subtask_configs` 的 key 和 `get_robot_eef_pose` 的 `eef_name` 统一使用 `"left"` 和 `"right"`，与观测空间中的命名一致（`eef_pos` / `right_eef_pos`）。

---

## 7. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `Embodiment name xxx not supported` | 环境名称不匹配 | 确认使用 `ex001arm_cvpr_scene_put_blocks_to_color` |
| 标注时没有画面 | 使用了 `--headless` | 标注必须有画面，不要加 `--headless` |
| 生成成功率很低 (<30%) | 子任务边界标注不准 | 重新标注，确保每个边界帧准确 |
| Action 维度不匹配 | Action 索引错误 | 检查 `EX001ArmActionsCfg` 顺序是否与 14D 布局一致 |
| `get_robot_eef_pose` 报错 | obs_buf 中缺少对应 key | 检查 `EX001ArmObservationsCfg` 是否包含所需观测 |
