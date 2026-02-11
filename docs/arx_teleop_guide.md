# ARX 方舟臂遥操作集成指南

## 概述

本模块通过 [arx5-sdk](https://github.com/yihuai-gao/arx5-sdk) 的 CAN 总线直连方式，将物理 ARX X5 双臂接入 IsaacLab-Arena 仿真平台，实现实时遥操作数据采集。**不依赖 ROS2**。

支持两种控制模式：

| 模式 | 说明 | 仿真端 Action 配置 | 适用场景 |
|------|------|---------------------|----------|
| `ee` (末端执行器) | 读取物理臂 EEF 位姿，计算帧间 delta pose | `EX001ArmActionsCfg` (DifferentialIK) | 精细操作、与 VR/键盘数据格式一致 |
| `joint` (关节位置) | 直接读取物理臂关节角度（弧度） | `EX001ArmJointActionsCfg` (AbsoluteJointPosition) | 关节级精确复现、无 IK 误差 |

## 新增/修改文件清单

```
IsaacLab-Arena/
├── isaaclab_arena/
│   ├── teleop_devices/
│   │   ├── __init__.py                          # [修改] 添加新模块导入
│   │   └── ex001arm_arx_bimanual.py             # [新增] ARX 双臂遥操作设备类
│   ├── embodiments/ex001arm/
│   │   └── ex001arm.py                          # [修改] 新增 EX001ArmJointActionsCfg
│   └── scripts/
│       └── record_ex001_arx_demos.py            # [新增] ARX 遥操作录制脚本
└── docs/
    └── arx_teleop_guide.md                      # [新增] 本文档
```

## 环境准备

### 1. 硬件连接

将两个 USB-CAN 适配器分别连接左臂和右臂，确认系统识别到 CAN 接口：

```bash
ip link show | grep can
```

### 2. 激活 CAN 接口

每次插入 USB-CAN 适配器或系统重启后，需要执行：

```bash
sudo ip link set up can0 type can bitrate 1000000
sudo ip link set up can1 type can bitrate 1000000
```

> 默认约定：`can0` = 左臂，`can1` = 右臂。可通过 CLI 参数 `--left_interface` / `--right_interface` 调整。

### 3. arx5-sdk 安装

确保 arx5-sdk 已编译并将 Python 绑定路径加入环境变量：

```bash
# 根据实际安装路径调整
export PYTHONPATH=$PYTHONPATH:/path/to/arx5-sdk/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/arx5-sdk/lib/x86_64
```

建议将上述两行添加到 `~/.bashrc` 或 Isaac Sim 的启动脚本中。

### 4. 验证 arx5-sdk 可用

```bash
python -c "from arx5_interface import Arx5JointController; print('arx5-sdk OK')"
```

### 5. pynput（可选，用于键盘回调）

```bash
pip install pynput
```

## 使用方式

### 基本命令

```bash
# EE 模式（默认）
python -m isaaclab_arena.scripts.record_ex001_arx_demos \
    --embodiment ex001arm \
    --task <task_name> \
    --control_mode ee

# 关节模式
python -m isaaclab_arena.scripts.record_ex001_arx_demos \
    --embodiment ex001arm \
    --task <task_name> \
    --control_mode joint
```

### 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--arx_model` | `X5` | ARX 机械臂型号（X5 / L5 / X7） |
| `--left_interface` | `can0` | 左臂 CAN 总线接口 |
| `--right_interface` | `can1` | 右臂 CAN 总线接口 |
| `--control_mode` | `ee` | 控制模式：`ee` 或 `joint` |
| `--teleop_hz` | `50` | 遥操作循环频率（Hz） |
| `--dataset_file` | `./demos` | 数据集输出路径 |
| `--reset_duration` | `10.0` | 自动重置轨迹时长（秒） |

### 示例

```bash
# 使用 X5 双臂，EE 模式，30Hz，输出到 ./my_data
python -m isaaclab_arena.scripts.record_ex001_arx_demos \
    --embodiment ex001arm \
    --task stack \
    --control_mode ee \
    --teleop_hz 30 \
    --dataset_file ./my_data

# 使用关节模式
python -m isaaclab_arena.scripts.record_ex001_arx_demos \
    --embodiment ex001arm \
    --task stack \
    --control_mode joint
```

## 操作流程

1. **启动前**：确认 CAN 接口已激活、双臂已上电
2. **启动脚本**：运行上述命令，等待 Isaac Sim 窗口打开
3. **开始录制**：脚本启动后自动进入录制状态，物理臂处于 damping 模式（可自由拖拽）
4. **执行任务**：手动拖拽物理臂完成任务，仿真中的机械臂会实时跟随
5. **重置**：按键盘 `R` 重置环境（不导出数据）
6. **自动导出**：当任务成功检测触发后，系统自动等待 2 秒 → 执行重置轨迹 → 导出 HDF5 文件
7. **结束**：按 `Ctrl+C` 退出

## 数据输出

每条轨迹保存为独立的 HDF5 文件，命名格式：

- EE 模式: `arx_ee_episode0.hdf5`, `arx_ee_episode1.hdf5`, ...
- 关节模式: `arx_joint_episode0.hdf5`, `arx_joint_episode1.hdf5`, ...

## 技术细节

### 14D 动作张量格式

**EE 模式** (与 VR/键盘格式完全一致)：

```
[left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
 right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]
```

- 位置 delta：当前帧与上一帧 EEF 位置之差（米）
- 旋转 delta：旋转向量（axis-angle），由前后帧旋转矩阵差计算
- 夹爪：`-1`（闭合）到 `+1`（张开）

**关节模式**：

```
[left_j1, left_j2, left_j3, left_j4, left_j5, left_j6, left_gripper,
 right_j1, right_j2, right_j3, right_j4, right_j5, right_j6, right_gripper]
```

- 关节角度：弧度值，直接来自 arx5-sdk
- 夹爪：`0.0`（闭合）到 `5.0`（张开），由物理夹爪米值线性映射

### 关节映射关系

| arx5-sdk (X5 URDF) | 仿真 (ex001arm USD) | 说明 |
|---------------------|---------------------|------|
| `joint1` | `left/right_arm_joint1` | 基座旋转 |
| `joint2` | `left/right_arm_joint2` | 肩部 |
| `joint3` | `left/right_arm_joint3` | 肘部 |
| `joint4` | `left/right_arm_joint4` | 腕部1 |
| `joint5` | `left/right_arm_joint5` | 腕部2 |
| `joint6` | `left/right_arm_joint6` | 腕部3 |
| `gripper_pos` (0~0.088m) | `left/right_arm_gripper` (0~5.0) | 夹爪 |

两者均为从基座到末端的顺序，索引一一对应。

### 夹爪映射公式

```
# EE 模式: 物理值 → 归一化
normalized = clip(gripper_pos / gripper_width, 0, 1) * 2 - 1
# 结果范围: -1 (闭合) → +1 (张开)

# 关节模式: 物理值 → 仿真关节值
joint_val = clip(gripper_pos / gripper_width, 0, 1) * 5.0
# 结果范围: 0.0 (闭合) → 5.0 (张开)
```

`gripper_width` 从 arx5-sdk 的 `RobotConfig` 动态读取（X5 = 0.088m）。

## 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `arx5_interface not found` | arx5-sdk 未安装或未加入 PYTHONPATH | 检查 `export PYTHONPATH` 设置 |
| `Network is down` (CAN) | CAN 接口未激活 | 执行 `sudo ip link set up canX ...` |
| 仿真臂不动 | CAN 接口名称错误 | 检查 `--left_interface` / `--right_interface` |
| 物理臂拖不动 | damping 过高 | 减小 `damping_scale`（默认 0.1，可尝试 0.05） |
| EE 模式漂移 | 坐标系不对齐 | 确认物理臂基座朝向与仿真一致 |
| 关节模式抖动 | 关节限位冲突 | 确认物理臂工作范围在仿真关节限位内 |
