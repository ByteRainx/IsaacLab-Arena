# EX001 Arm 仿真模块 — 迁移指南

本文档说明从 `rain/migration-clean` 分支迁移 EX001 双臂机器人仿真模块到 IsaacLab Arena 主仓库所涉及的全部代码变更、目录结构和启动方式。

---

## 1. 变更总览

本次迁移在 `release/0.1.1` 基础上新增 **EX001 Arm 双臂仿真**全套功能，包括：

- 1 个 Embodiment（EX001Arm，含 ee / joint 两种控制模式）
- 3 个任务（彩块归位、按钮接触、水果入篮）
- 3 个示例环境（对应上述 3 个任务的 CVPR 场景）
- 3 种遥操作方式（键盘、VR OpenXR、远程 WebSocket 主从臂）
- 统一录制脚本（支持全部 3 种遥操作设备）
- 回放再录制脚本（补齐 VR 模式下的渲染）
- 混元 3D 场景资产（~30 种物体 USD）

---

## 2. 新增与修改的文件清单

### 2.1 新增文件

#### `assets/` — 二进制资产（USD / 纹理 / 场景）

```
assets/
├── ex001arm_bimanual/          # EX001 双臂机器人 USD（含鱼眼相机）
│   └── ex001_arm.usd
├── hunyuan_assets/             # 混元 3D 物体资产（~30种）
│   ├── apple/                  # 每个物体含 .usd, _physics.usd, textures/
│   ├── banana/
│   ├── ball/
│   ├── black_bowl/
│   ├── bottom/
│   ├── bread/
│   ├── button_blue/
│   ├── button_green/
│   ├── button_orange/
│   ├── button_pink/
│   ├── button_purple/
│   ├── cherry/
│   ├── fruit/
│   ├── grape/
│   ├── green_brick/
│   ├── green_paper/
│   ├── orange/
│   ├── pink_paper/
│   ├── plate/
│   ├── platform_pink/
│   ├── platform_white/
│   ├── platform_yellow/
│   ├── red_brick/
│   ├── stick/
│   ├── white_bowl/
│   ├── wood/
│   ├── wood_bottom/
│   ├── yellow_brick/
│   └── yellow_paper/
└── scene-3dgs/                 # 3DGS 背景场景
    ├── 12_25_01/
    ├── 12_25_02/
    └── scene_01.usd
```

#### `isaaclab_arena/embodiments/ex001arm/` — Embodiment 定义

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块导出 |
| `ex001arm.py` | `EX001ArmEmbodiment` 主类 + 场景/动作/观测配置 |
| `actions.py` | `ThreeStateGripperAction` (自适应开合度) + `ContactLimitedGripperAction` (legacy) |
| `observations.py` | EE 位姿 (root frame)、夹爪归一化等观测函数 |

#### `isaaclab_arena/tasks/` — 任务判定逻辑

| 文件 | 说明 |
|------|------|
| `put_blocks_to_color_task.py` | 彩块归位：逐对检查 block-目标点距离 |
| `buttons_contact_task.py` | 按钮接触：ContactSensor 累计触碰历史 |
| `put_fruits_to_basket_task.py` | 水果入篮：检查水果与篮子距离 + 速度约束 |

#### `isaaclab_arena/examples/example_environments/` — 场景组装

| 文件 | 环境名 |
|------|--------|
| `ex001arm_cvpr_scene_put_blocks_to_color_environment.py` | `ex001arm_cvpr_scene_put_blocks_to_color` |
| `ex001arm_cvpr_scene_buttons_contact_environment.py` | `ex001arm_cvpr_scene_buttons_contact` |
| `ex001arm_cvpr_scene_put_fruits_to_basket_environment.py` | `ex001arm_cvpr_scene_put_fruits_to_basket` |

#### `isaaclab_arena/teleop_devices/` — 遥操作设备

| 文件 | 说明 |
|------|------|
| `ex001arm_openxr_bimanual.py` | VR OpenXR 双臂遥操 (14D: left_se3 + grip + right_se3 + grip) |
| `ex001arm_ws_remote.py` | WebSocket 远程遥操（支持 `ee` / `joint` 两种模式） |

#### `isaaclab_arena/scripts/` — 运行脚本

| 文件 | 说明 |
|------|------|
| `teleop_bimanual_keyboard.py` | 双臂键盘遥操控制 |
| `record_ex001_demos.py` | **统一录制脚本**（keyboard / VR / remote 三合一） |
| `replay_and_record_demos.py` | 回放 HDF5 并补录渲染（解决 VR 下 3DGS 黑屏） |
| `ros1_ws_bridge.py` | 机器人端 ROS1→WebSocket 桥接脚本 |

#### `docs/` — 文档

| 文件 | 说明 |
|------|------|
| `migration_readme.md` | 本文档 |

### 2.2 修改的现有文件

| 文件 | 变更内容 |
|------|----------|
| `.gitignore` | 新增 `*demos/`、`*data/`、`*lerobot_data/`、`*videos/`、`*openxr/` |
| `isaaclab_arena/assets/object_library.py` | 注册混元 3D 物体（Apple, Banana, Grape 等 ~30 个 `LibraryObject` 子类） |
| `isaaclab_arena/assets/background_library.py` | 注册 3DGS 背景场景 |
| `isaaclab_arena/embodiments/__init__.py` | 新增 `from .ex001arm.ex001arm import *` |
| `isaaclab_arena/examples/example_environments/cli.py` | 注册 3 个新环境到 `ExampleEnvironments` 字典 |
| `isaaclab_arena/teleop_devices/__init__.py` | 新增 `ex001arm_openxr_bimanual` 和 `ex001arm_ws_remote` 导出 |

---

## 3. 架构说明

### 3.1 Embodiment 结构

`EX001ArmEmbodiment` 初始化构建三部分配置：

```
EX001ArmEmbodiment
├── scene_config (EX001ArmSceneCfg)
│   ├── robot: ArticulationCfg (双臂 + 夹爪)
│   ├── ee_frame / right_ee_frame: FrameTransformerCfg
│   ├── left/right_wrist_camera: CameraCfg (鱼眼, USD 内置)
│   ├── head_camera: CameraCfg (针孔, 代码添加)
│   ├── left/right_gripper_contact: ContactSensorCfg
│   └── dome_light: AssetBaseCfg
├── action_config
│   ├── EX001ArmActionsCfg (ee 模式, IK scale=0.5, 键盘/VR 使用)
│   └── EX001ArmJointActionsCfg (joint 模式, 关节直通, 远程主从臂使用)
└── observation_config (EX001ArmObservationsCfg)
    ├── 关节位置/速度
    ├── 末端位姿 (world frame + root frame)
    ├── 夹爪状态 (原始 + 归一化 [0,1])
    └── 相机图像 (左腕/右腕/头部)
```

### 3.2 三态夹爪控制 (`ThreeStateGripperAction`)

```
OPEN ──(cmd=close)──→ CLOSE ──(contact force)──→ GRASP
  ↑                     │                          │
  └──(cmd=open)─────────┘                          │
  └──(cmd=open)────────────────────────────────────┘
```

- **CLOSE → GRASP**：双指接触力均超过阈值时触发
- **GRASP 开合度**：接触触发瞬间的关节位置快照再乘以系数（`grasp_hold_ratio`），自适应物体大小（非固定值）
- **GRASP → OPEN**：仅响应明确的 open 指令

### 3.3 环境组装模式

```
Scene(assets=[hunyuan物体...]) + Task + Embodiment + TeleopDevice → IsaacLabArenaEnvironment
```

每个环境文件负责：
- 配置场景内物体的初始位姿与重置随机化
- `modify_env_cfg`：3DGS 渲染参数（必须配置，否则黑屏）
- 注册 `--teleop_device` 等 CLI 参数

### 3.4 远程遥操链路（Joint 关节直通模式）

```
物理双臂(ROS1) → ros1_ws_bridge.py(WebSocket JSON, 200Hz) → ex001arm_ws_remote.py → Isaac Sim
```

数据流：物理臂 `joint_pos[0:6]` 经过 sign/offset 映射后，通过 `JointPositionActionCfg` 直接设置仿真关节目标位置。

**关节符号映射**（ARX X5 默认值）：

| 关节 | sign | 原因 |
|------|------|------|
| joint1 (底座) | +1 | 轴向一致 |
| joint2 (肩部) | +1 | 轴向一致 |
| joint3 (肘部) | -1 | URDF origin 有 π 旋转，USD 轴向相反 |
| joint4 (腕部1) | -1 | Y 轴方向反转 |
| joint5 (腕部2) | -1 | 轴向相反 |
| joint6 (末端) | +1 | 轴向一致 |

---

## 4. 启动命令

### 4.1 键盘遥操控制

```bash
# 彩块归位
python isaaclab_arena/scripts/teleop_bimanual_keyboard.py \
  --enable_cameras \
  --task put_blocks_to_color_task \
  ex001arm_cvpr_scene_put_blocks_to_color \
  --teleop_device keyboard

# 按钮接触
python isaaclab_arena/scripts/teleop_bimanual_keyboard.py \
  --enable_cameras \
  --task buttons_contact_task \
  ex001arm_cvpr_scene_buttons_contact \
  --teleop_device keyboard

# 水果入篮
python isaaclab_arena/scripts/teleop_bimanual_keyboard.py \
  --enable_cameras \
  --task put_fruits_to_basket_task \
  ex001arm_cvpr_scene_put_fruits_to_basket \
  --teleop_device keyboard
```

### 4.2 数据录制（统一脚本，支持 keyboard / VR / remote）

#### 键盘录制

```bash
python isaaclab_arena/scripts/record_ex001_demos.py \
  --enable_cameras \
  --teleop_device keyboard \
  ex001arm_cvpr_scene_put_blocks_to_color
```

#### VR 录制

```bash
# 步骤 1: 录制动作（VR 下 3DGS 可能黑屏）
python isaaclab_arena/scripts/record_ex001_demos.py \
  --teleop_device ex001arm_openxr_bimanual \
  ex001arm_cvpr_scene_put_blocks_to_color

# 步骤 2: 回放补齐渲染
python isaaclab_arena/scripts/replay_and_record_demos.py \
  --enable_cameras \
  --replay_mode state \
  --dataset_name=vr_episode0.hdf5 \
  ex001arm_cvpr_scene_put_blocks_to_color
```

#### 远程主从臂录制（Joint 关节直通模式）

```bash
# 机器人端（Docker 容器内）
python isaaclab_arena/scripts/ros1_ws_bridge.py --mode joint --hz 200

# 仿真端
python isaaclab_arena/scripts/record_ex001_demos.py \
  --teleop_device remote \
  --control_mode joint \
  --disable_pinocchio \
  --enable_cameras \
  --remote_ip 10.100.21.249 \
  ex001arm_cvpr_scene_put_blocks_to_color
```

### 4.3 录制脚本完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--teleop_device` | `keyboard` | `keyboard` / `ex001arm_openxr_bimanual` / `remote` |
| `--dataset_file` | `./data/<env>` | 输出 HDF5 目录 |
| `--reset_duration` | `10.0` | 成功后自动重置轨迹时长 (秒) |
| `--enable_cameras` | - | 启用相机渲染 |
| `--disable_pinocchio` | - | 禁用 Pinocchio（避免库冲突） |
| `--debug` | - | 打印关节诊断信息 |
| `--remote_ip` | `10.100.21.249` | 机器人端 IP (remote 模式) |
| `--remote_port` | `5555` | WebSocket 端口 (remote 模式) |
| `--control_mode` | `ee` | 远程控制模式 (`ee` / `joint`, remote 模式) |
| `--joint_signs` | `1,1,-1,-1,-1,1` | 关节符号翻转 (6 个逗号分隔) |
| `--joint_offsets` | `0,0,0,0,0,0` | 关节偏移量 (弧度, 6 个逗号分隔) |

### 4.4 操作方式

- 键盘/远程模式：启动后**自动开始录制**
- VR 模式：需在 VR 中点击 **START** 开始录制
- 按 `R`：重置环境（丢弃当前数据，不导出）
- 任务成功后：自动等待 2s → 执行重置轨迹 → 导出 HDF5 → 准备下一条
- `Ctrl+C`：结束录制

### 4.5 纯仿真回放（HDF5）

```bash
python isaaclab_arena/scripts/replay_demos.py \
  --dataset_file data/ex001arm_cvpr_scene_put_blocks_to_color/arx_remote_joint_episode0.hdf5 \
  --replay_mode joint \
  ex001arm_cvpr_scene_put_blocks_to_color
```

`--replay_mode` 可选：

- `action`：按数据集 `actions` 回放
- `joint`：按 `obs/joint_pos` 回放（关节控制）
- `ee`：按 `obs/eef_state`（或 `eef_pos/eef_quat/...`）回放
- `state`：按记录状态精确回放（最接近原轨迹）

---

## 5. 迁移步骤

### 5.1 复制文件

```bash
# 1. 从 rain/migration-clean 分支检出到目标仓库
git checkout rain/migration-clean

# 2. 需要复制的目录/文件：
#    新增目录:
#      assets/ex001arm_bimanual/
#      assets/hunyuan_assets/
#      assets/scene-3dgs/
#      isaaclab_arena/embodiments/ex001arm/
#      docs/migration_readme.md
#
#    新增文件 (放入已有目录):
#      isaaclab_arena/tasks/put_blocks_to_color_task.py
#      isaaclab_arena/tasks/buttons_contact_task.py
#      isaaclab_arena/tasks/put_fruits_to_basket_task.py
#      isaaclab_arena/examples/example_environments/ex001arm_cvpr_scene_*.py (3个)
#      isaaclab_arena/teleop_devices/ex001arm_openxr_bimanual.py
#      isaaclab_arena/teleop_devices/ex001arm_ws_remote.py
#      isaaclab_arena/scripts/record_ex001_demos.py
#      isaaclab_arena/scripts/replay_and_record_demos.py
#      isaaclab_arena/scripts/teleop_bimanual_keyboard.py
#      isaaclab_arena/scripts/ros1_ws_bridge.py
```

### 5.2 修改现有文件

```bash
# 以下文件需要 merge 修改（不是整体替换）:
#   .gitignore                                    — 追加忽略规则
#   isaaclab_arena/assets/object_library.py       — 追加 LibraryObject 注册
#   isaaclab_arena/assets/background_library.py   — 追加背景注册
#   isaaclab_arena/embodiments/__init__.py         — 追加 ex001arm 导出
#   isaaclab_arena/examples/example_environments/cli.py — 追加环境注册
#   isaaclab_arena/teleop_devices/__init__.py      — 追加设备导出
```

### 5.3 验证

```bash
# 清理缓存
find isaaclab_arena -name "__pycache__" -type d -exec rm -rf {} +

# 安装（editable mode）
pip install -e .

# 测试键盘控制
python isaaclab_arena/scripts/teleop_bimanual_keyboard.py \
  --enable_cameras \
  --task put_blocks_to_color_task \
  ex001arm_cvpr_scene_put_blocks_to_color \
  --teleop_device keyboard
```

---

## 6. 依赖说明

| 包 | 用途 | 安装位置 |
|----|------|----------|
| `websocket-client` | WebSocket 远程遥操 (仿真端) | `pip install websocket-client` |
| `websockets` | WebSocket 桥接服务 (机器人端) | `pip install websockets` |
| `scipy` | 旋转计算 | 通常已安装 |
| `rospy` + `arm_control` | ROS1 话题订阅 (机器人端) | ROS1 环境 |

核心依赖 (Isaac Lab / IsaacLab Arena) 不变。

---

## 7. 与原始仓库的关键区别

本次迁移在原始开发分支基础上做了以下**简化与重构**：

| 项目 | 原始 | 迁移后 |
|------|------|--------|
| 录制脚本 | `record_ex001_demos.py` + `record_ex001_remote_demos.py` (历史：两个独立脚本) | 当前仅使用 `record_ex001_demos.py` |
| `ee_to_joint_action.py` | 独立文件 (测试用) | 已删除，不再需要 |
| 夹爪 GRASP 开合度 | 固定 `grasp_command_expr` (如 1.7) | **自适应**：接触快照关节位置 × `grasp_hold_ratio` |
| `tools/` 目录 | 独立顶层目录 | 合并到 `isaaclab_arena/scripts/` |
| OpenArm embodiment | 有完整代码 | 已删除，不再迁移 |
| `x2robot_closedloop_policy` | 有完整代码 + CLI 集成 | 已移除 |
| 数据转换脚本 | `convert2lerobot.py` + `hdf5_to_lerobot.py` | 已移除（待独立维护） |

---

## 8. 已知局限

1. **观测对齐**：仿真与真机的观测数据格式尚未完全对齐，数据采集后需验证
2. **VR 渲染**：VR 模式下 3DGS 场景可能黑屏，需要"录制→回放"两步流程
3. **碰撞体精度**：混元 3D 资产的碰撞体由 IsaacSim 自动生成，复杂几何（如 grape）较粗糙
4. **夹爪颜色**：机械臂做了简易上色，与真机颜色有差异
5. **自碰撞**：夹爪碰撞体粗糙，建议关闭自碰撞（当前已关闭）
