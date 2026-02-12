# EX001Arm 远程遥操作指南

> 适用于 ARX X5 双臂机械臂通过网络远程控制 Isaac Sim 仿真。

---

## 1. 总体架构

两台电脑通过网线连接，使用 WebSocket 传输机械臂状态数据。支持 **EE 末端控制**和**关节直通控制**两种模式。

```
┌─────────────────────────────────────────┐            ┌──────────────────────────────────────────┐
│         机器人电脑 (arm-pc)              │            │          仿真电脑 (Isaac Sim)             │
│         Docker 容器 / ROS1              │            │                                          │
│                                         │            │                                          │
│  物理双臂 (CAN 总线)                     │            │                                          │
│       ↓                                 │            │                                          │
│  ROS1 topics (200Hz):                   │            │                                          │
│    EE:    /master1_pos_back (PosCmd)    │            │                                          │
│           /master2_pos_back (PosCmd)    │            │                                          │
│    Joint: /joint_control  (JointCtrl)   │            │                                          │
│           /joint_control2 (JointCtrl)   │            │                                          │
│       ↓                                 │            │                                          │
│  ros1_ws_bridge.py                      │  WebSocket │  ex001arm_ws_remote.py                   │
│  (订阅 ROS topic → JSON → WS推送)       │ ─────────► │  (接收 JSON → 计算动作 → 14D action)     │
│  默认: ws://0.0.0.0:5555  200Hz        │  :5555     │       ↓                                  │
│                                         │            │  EE模式: DifferentialIK (scale=1.0)      │
│  依赖: pip install websockets           │            │  Joint模式: JointPositionAction (直通)    │
│                                         │            │       ↓                                  │
│                                         │            │  Isaac Sim 仿真环境                       │
│                                         │            │  record_ex001_remote_demos.py             │
│                                         │            │  依赖: pip install websocket-client       │
└─────────────────────────────────────────┘            └──────────────────────────────────────────┘
```

---

## 2. 两种控制模式对比

| 特性 | EE 末端控制 (`ee`) | 关节直通控制 (`joint`) |
|------|---------------------|------------------------|
| **输入数据** | `PosCmd` (xyz + rpy + gripper) | `JointControl` (6关节 + gripper) |
| **控制方式** | 逆运动学 (IK) 求解关节速度 | 直接设置关节目标角度 |
| **延迟** | 稍高 (IK 计算) | 最低 (直通) |
| **精度** | 末端空间精确 | 关节空间 1:1 |
| **适用场景** | 需要末端轨迹精确 | 需要关节一致性 / 采集关节数据 |
| **Action Config** | `EX001ArmPhysicalTeleopActionsCfg` | `EX001ArmJointActionsCfg` |
| **Bridge `--mode`** | `ee` | `joint` |

---

## 3. ROS1 消息格式

### 3.1 EE 模式话题

| Topic | 消息类型 | 说明 |
|-------|----------|------|
| `/master1_pos_back` | `arm_control/PosCmd` | 左臂 master arm 末端位姿 |
| `/master2_pos_back` | `arm_control/PosCmd` | 右臂 master arm 末端位姿 |

**PosCmd 字段：**

| 字段 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `x` | float | ~-0.4 ~ 0.4 | 末端 X 位置 (米), 相对于臂基座 |
| `y` | float | ~-0.15 ~ 0.15 | 末端 Y 位置 (米) |
| `z` | float | ~-0.01 ~ 0.53 | 末端 Z 位置 (米) |
| `roll` | float | 弧度 | 末端姿态 Roll (绕X轴) |
| `pitch` | float | 弧度 | 末端姿态 Pitch (绕Y轴) |
| `yaw` | float | 弧度 | 末端姿态 Yaw (绕Z轴) |
| `gripper` | float | 0.0 ~ 4.5 | 夹爪: 0=闭合, 4.5=完全张开 |
| `mode1` | int | 0/1 | 模式标志 (当前未使用) |
| `mode2` | int | 0/1 | 模式标志 (当前未使用) |

### 3.2 Joint 模式话题

| Topic | 消息类型 | 说明 |
|-------|----------|------|
| `/joint_control` | `arm_control/JointControl` | 左臂关节状态 |
| `/joint_control2` | `arm_control/JointControl` | 右臂关节状态 |

**JointControl 字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `joint_pos` | float32[7] | 关节位置: `[j1, j2, j3, j4, j5, j6, gripper]` (弧度) |
| `joint_vel` | float32[7] | 关节速度 (弧度/秒) |
| `joint_cur` | float32[7] | 关节电流 (A) |
| `mode` | int32 | 控制模式标志 |

**`joint_pos` 各索引含义 (ARX X5)：**

| 索引 | 物理关节 | URDF 轴向 | 说明 |
|------|----------|-----------|------|
| 0 | joint1 (底座) | Z (0,0,1) | 底座水平旋转 |
| 1 | joint2 (肩部) | Y (0,1,0) | 肩部俯仰 |
| 2 | joint3 (肘部) | Y (0,1,0) | 肘部俯仰 |
| 3 | joint4 (腕部1) | Y (0,1,0) | 腕部俯仰 |
| 4 | joint5 (腕部2) | Z (0,0,1) | 腕部旋转 |
| 5 | joint6 (腕部3) | X (1,0,0) | 末端滚转 |
| 6 | gripper | - | 夹爪开合 |

发布频率: **200Hz**

---

## 4. WebSocket 传输协议

桥接脚本以 `--hz` 指定的频率推送 JSON。

### EE 模式 Payload

```json
{
  "timestamp": 1770898856.993228,
  "left": {
    "x": -0.0080, "y": 0.0180, "z": 0.0021,
    "roll": -0.0421, "pitch": 0.1579, "yaw": 0.3261,
    "gripper": 0.0, "mode1": 0, "mode2": 0
  },
  "right": { ... }
}
```

### Joint 模式 Payload

```json
{
  "timestamp": 1770898856.993228,
  "left": {
    "joint_pos": [0.0002, 0.0010, 0.0013, 0.0021, -0.0063, 0.0006, 0.0],
    "joint_vel": [-0.0132, -0.0044, -0.0220, -0.0110, 0.0110, 0.0110, 0.0]
  },
  "right": { ... }
}
```

### Both 模式 Payload

两者合并，`left`/`right` 同时包含 EE 和 Joint 字段。

---

## 5. EE 模式 — Delta 计算详解

位于 `ex001arm_ws_remote.py` 的 `_advance_ee()` 方法。

### 5.1 位置 delta

直接做差，单位是米：

```python
delta_pos = curr_pose[:3] - prev_pose[:3]
# 例: curr=[0.01, 0.02, 0.53], prev=[0.00, 0.02, 0.50]
#   → delta_pos = [0.01, 0.00, 0.03]  (X向前移了1cm, Z向上移了3cm)
```

### 5.2 旋转 delta

从连续帧的 Euler 角计算旋转向量 (rotation vector)，这是 DifferentialIK 期望的格式：

```python
from scipy.spatial.transform import Rotation

R_prev = Rotation.from_euler("xyz", [roll_prev, pitch_prev, yaw_prev])
R_curr = Rotation.from_euler("xyz", [roll_curr, pitch_curr, yaw_curr])

# 旋转差: R_delta = R_curr * R_prev^(-1)
delta_rot = (R_curr * R_prev.inv()).as_rotvec()
# → 输出3D向量 [rx, ry, rz], 方向=旋转轴, 大小=旋转角度(弧度)
```

### 5.3 缩放

```python
delta_pos *= cfg.pos_scale   # 默认 1.0 (1:1映射)
delta_rot *= cfg.rot_scale   # 默认 1.0
```

`pos_scale` 和 `rot_scale` 保留作为微调手段，同型号臂理论上不需要调。

### 5.4 夹爪归一化

物理臂夹爪 `[0, 4.5]` 映射到仿真 `[-1, +1]`：

```python
ratio = clip((gripper_val - 0.0) / (4.5 - 0.0), 0, 1)  # → [0, 1]
normalized = ratio * 2.0 - 1.0                            # → [-1, +1]

# 0.0 (闭合) → -1.0
# 2.25 (半开) → 0.0
# 4.5 (全开) → +1.0
```

### 5.5 输出: 14D Action Tensor

```
索引:  [0    1    2    3     4     5     6      7    8    9    10    11    12    13  ]
含义:  [L_dx L_dy L_dz L_drx L_dry L_drz L_grip R_dx R_dy R_dz R_drx R_dry R_drz R_grip]
       |←── 左臂 6D delta + 夹爪 ──→|           |←── 右臂 6D delta + 夹爪 ──→|
```

- `[0:3]` = 左臂位置 delta (米)
- `[3:6]` = 左臂旋转 delta (rotation vector, 弧度)
- `[6]`   = 左臂夹爪 (-1=闭, +1=开)
- `[7:10]` = 右臂位置 delta
- `[10:13]` = 右臂旋转 delta
- `[13]`  = 右臂夹爪

### 5.6 DifferentialIK 控制器

```python
# 定义在 ex001arm.py → EX001ArmPhysicalTeleopActionsCfg
DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["left_arm_joint[1-6]"],
    body_name="left_arm_gripper_base_link",
    controller=DifferentialIKControllerCfg(
        command_type="pose",        # 接收 6D delta (3D pos + 3D rot)
        use_relative_mode=True,     # 输入是相对增量，不是绝对位姿
        ik_method="dls",            # Damped Least Squares
    ),
    scale=1.0,   # ← 关键: 物理臂遥操作用 1.0 (原来键盘模式是 0.5)
)
```

**`scale=1.0` 的意义**: delta 不会被缩放，物理臂移动 1cm，仿真臂也移动 1cm。

夹爪使用 `ContactLimitedGripperActionCfg`：
- `-1` → 闭合 (joint position 0.0)
- `+1` → 张开 (joint position 5.0)
- 带接触力限制 (15N)，防止夹碎物体

---

## 6. Joint 模式 — 关节直通详解

位于 `ex001arm_ws_remote.py` 的 `_advance_joint()` 方法。

### 6.1 数据流

```
物理臂编码器 → ROS1 JointControl (200Hz)
    ↓
ros1_ws_bridge.py --mode joint
    ↓  (提取 joint_pos, joint_vel → JSON)
WebSocket → ex001arm_ws_remote.py
    ↓
_advance_joint():
    1. 提取 joint_pos[0:6] 作为 6 个关节角度
    2. 提取 joint_pos[6] 作为夹爪值
    3. 应用符号翻转: sim_joint = sign * physical_joint + offset
    4. 拼接为 14D action tensor
    ↓
JointPositionActionCfg (Isaac Lab)
    ↓  直接设置关节目标位置
Isaac Sim 物理仿真 (PD控制器跟踪目标)
```

### 6.2 关节符号映射 (joint_signs)

由于物理臂 URDF 和仿真 USD 模型在转换过程中部分关节轴向相反，需要对关节 3-6 做符号翻转。

**映射公式：**

```
sim_joint[i] = joint_signs[i] × physical_joint[i] + joint_offsets[i]
```

**ARX X5 默认符号 (已内置)：**

| 关节 | joint_signs | 原因 |
|------|-------------|------|
| joint1 (底座) | **+1** | 轴向一致 |
| joint2 (肩部) | **+1** | 轴向一致 |
| joint3 (肘部) | **-1** | URDF origin 有 π 旋转，USD 轴向相反 |
| joint4 (腕部1) | **-1** | 同上，Y轴方向反转 |
| joint5 (腕部2) | **-1** | 轴向相反 |
| joint6 (末端) | **-1** | URDF origin 有 π 旋转，X轴方向反转 |

默认值: `joint_signs = (1, 1, -1, -1, -1, -1)`

> 如果发现某个关节还是反的，可以通过 `--joint_signs` 参数微调。

### 6.3 输出: 14D Action Tensor

```
索引:  [0    1    2    3    4    5    6      7    8    9    10   11   12   13  ]
含义:  [L_j1 L_j2 L_j3 L_j4 L_j5 L_j6 L_grip R_j1 R_j2 R_j3 R_j4 R_j5 R_j6 R_grip]
       |←── 左臂 6 关节角(弧度) + 夹爪 ──→|     |←── 右臂 6 关节角(弧度) + 夹爪 ──→|
```

- `[0:6]` = 左臂 joint1-joint6 角度 (弧度, 经过 sign/offset 变换)
- `[6]`   = 左臂夹爪 (绝对关节位置)
- `[7:13]` = 右臂 joint1-joint6 角度
- `[13]`  = 右臂夹爪

### 6.4 JointPositionAction 控制器

```python
# 定义在 ex001arm.py → EX001ArmJointActionsCfg
JointPositionActionCfg(
    asset_name="robot",
    joint_names=["left_arm_joint[1-6]"],
    scale=1.0,
    use_default_offset=False,   # 使用绝对角度，不加默认偏移
)
```

关节角度直接作为 PD 控制器的目标位置，由 `ImplicitActuatorCfg` 的刚度/阻尼驱动：

```python
# 臂关节: stiffness=80, damping=8  (响应快)
# 夹爪:   stiffness=200, damping=8  (非常响应)
```

---

## 7. 执行器参数

### 7.1 臂关节执行器

| 参数 | 值 | 说明 |
|------|-----|------|
| `effort_limit_sim` | 87.0 N·m | 最大关节力矩 |
| `stiffness` | 80.0 | PD 比例增益 (位置跟踪刚度) |
| `damping` | 8.0 | PD 微分增益 (运动阻尼) |

### 7.2 夹爪执行器

| 参数 | 值 | 说明 |
|------|-----|------|
| `effort_limit_sim` | 200.0 N·m | 最大夹爪力矩 |
| `stiffness` | 200.0 | 高刚度确保夹爪快速响应 |
| `damping` | 8.0 | 低阻尼避免拖沓感 |

> 夹爪刚度设为 200 (臂关节的 2.5 倍)，确保按下夹爪后仿真即时跟随。

---

## 8. 首帧处理与 Reset

- **首帧 (EE模式)**: `_prev_left = None` 时，记录当前位姿作为基准，输出零向量（不产生跳变）
- **首帧 (Joint模式)**: 无需基准，直接传入绝对角度
- **Reset**: 调用 `teleop.reset()` 清空 `_prev_left/right`，EE 模式下一帧重新初始化基准
- **断线重连**: WebSocket 断开后后台线程自动每 2 秒尝试重连

---

## 9. 文件清单

| 文件 | 运行位置 | 功能 |
|------|----------|------|
| `tools/ros1_ws_bridge.py` | 机器人电脑 (Docker) | ROS1 → WebSocket 桥接 |
| `isaaclab_arena/teleop_devices/ex001arm_ws_remote.py` | 仿真电脑 | WebSocket 接收 + 动作计算 |
| `isaaclab_arena/scripts/record_ex001_remote_demos.py` | 仿真电脑 | 录制脚本 (启动仿真 + 遥操作) |
| `isaaclab_arena/embodiments/ex001arm/ex001arm.py` | 仿真电脑 | Action/Actuator 配置定义 |
| `tools/ws_debug_client.py` | 仿真电脑 | 独立诊断工具 (查看原始数据) |

---

## 10. 使用步骤

### 10.1 机器人端 (Docker 容器内)

```bash
# 1. 确保 ROS master 和机械臂控制器已启动
# 2. 进入容器
dexec

# 3. 安装依赖 (首次)
pip install websockets

# 4. 启动桥接
#    EE 模式 (默认):
python ros1_ws_bridge.py --hz 200

#    Joint 模式:
python ros1_ws_bridge.py --mode joint --hz 200

#    同时发送 EE + Joint:
python ros1_ws_bridge.py --mode both --hz 200
```

成功输出:
```
[ws_bridge] Joint topics: left=/joint_control  right=/joint_control2
[ws_bridge] WebSocket server listening on 0.0.0.0:5555  (mode=joint, 200 Hz)
```

> **注意**: Docker 使用 `--network host` 模式时不需要端口映射。
> 否则需要 `docker run -p 5555:5555 ...`

### 10.2 仿真端

#### EE 模式 (默认)

```bash
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --disable_pinocchio \
    --enable_cameras \
    --remote_ip 10.100.21.249 \
    --dataset_file ./demos \
    ex001arm_cvpr_scene_put_blocks_to_color
```

#### Joint 模式 (推荐: 关节一致性最好)

```bash
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --control_mode joint \
    --disable_pinocchio \
    --enable_cameras \
    --remote_ip 10.100.21.249 \
    --dataset_file ./demos \
    ex001arm_cvpr_scene_put_blocks_to_color
```

#### 带调试输出 (排查问题时使用)

```bash
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --control_mode joint \
    --disable_pinocchio \
    --enable_cameras \
    --remote_ip 10.100.21.249 \
    --dataset_file ./demos \
    --debug \
    ex001arm_cvpr_scene_put_blocks_to_color
```

启动时会打印 **关节诊断表** (joint 模式或 --debug 时自动打印):

```
================================================================================
  SIMULATION JOINT DIAGNOSTIC (from USD articulation)
================================================================================
  Total joints: 14
  Idx  Joint Name                          Curr Pos    Default      Lower      Upper
  ---------------------------------------------------------------------------------
  0    left_arm_joint1                      +0.0000    +0.0000   -10.0000   +10.0000 <<
  1    left_arm_joint2                      +0.0000    +0.0000    +0.0000    +3.1400 <<
  ...

  ACTION TERM -> JOINT MAPPING:
    arm_action: joint_ids=[0, 1, 2, 3, 4, 5] -> [left_arm_joint1, ..., left_arm_joint6]
    ...
================================================================================
```

运行时 Debug 输出 (每 50 帧):

```
[Joint Debug #50] signs=[1.0, 1.0, -1.0, -1.0, -1.0, -1.0]  offsets=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  L_raw   = [-0.0269, +0.0010, -0.0044, -0.0277, -0.3180, +0.0387]  grip=0.000
  L_mapped= [-0.0269, +0.0010, +0.0044, +0.0277, +0.3180, -0.0387]
  R_raw   = [-0.0292, +0.0040, +0.0185, +0.2096, -0.0216, -0.2169]  grip=0.000
  R_mapped= [-0.0292, +0.0040, -0.0185, -0.2096, +0.0216, +0.2169]
```

### 10.3 操作

- 脚本启动后 **自动开始录制**
- 手动操作物理臂，仿真臂实时跟随
- 按 `R` 重置环境（不导出数据）
- 任务成功后自动导出 HDF5 + 重置
- `Ctrl+C` 结束

---

## 11. 完整参数列表

### 11.1 桥接脚本 (`ros1_ws_bridge.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `5555` | WebSocket 监听端口 |
| `--hz` | `200` | 推送频率 (Hz) |
| `--mode` | `ee` | 数据模式: `ee` / `joint` / `both` |
| `--left_topic` | `/master1_pos_back` | 左臂 PosCmd topic |
| `--right_topic` | `/master2_pos_back` | 右臂 PosCmd topic |
| `--joint_left_topic` | `/joint_control` | 左臂 JointControl topic |
| `--joint_right_topic` | `/joint_control2` | 右臂 JointControl topic |

### 11.2 录制脚本 (`record_ex001_remote_demos.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote_ip` | `192.168.1.100` | 机器人电脑 IP |
| `--remote_port` | `5555` | WebSocket 端口 |
| `--control_mode` | `ee` | 控制模式: `ee` (末端delta) / `joint` (关节直通) |
| `--pos_scale` | `1.0` | 位置 delta 缩放 (仅 EE 模式) |
| `--rot_scale` | `1.0` | 旋转 delta 缩放 (仅 EE 模式) |
| `--joint_signs` | `1,1,-1,-1,-1,-1` | 关节符号翻转 (仅 Joint 模式, 6个逗号分隔) |
| `--joint_offsets` | `0,0,0,0,0,0` | 关节偏移量 (仅 Joint 模式, 弧度, 6个逗号分隔) |
| `--debug` | `false` | 打印调试信息 + 关节诊断表 |
| `--dataset_file` | `./demos` | 数据输出目录 |
| `--reset_duration` | `10.0` | 自动重置时长 (秒) |
| `--disable_pinocchio` | - | 禁用 Pinocchio (避免库冲突) |
| `--enable_cameras` | - | 启用仿真相机 |

---

## 12. 数据输出

每条轨迹保存为独立 HDF5 文件:

```
demos/
├── arx_remote_ee_episode0.hdf5      # EE 模式采集
├── arx_remote_ee_episode1.hdf5
├── arx_remote_joint_episode0.hdf5   # Joint 模式采集
└── ...
```

文件名自动包含控制模式 (`ee` / `joint`) 和自增序号。

---

## 13. 诊断工具

### WebSocket 调试客户端

不启动仿真，直接查看桥接推送的原始数据:

```bash
python tools/ws_debug_client.py --ip 10.100.21.249 --port 5555
```

### Python 快速连通性测试

```python
import websocket
ws = websocket.create_connection('ws://10.100.21.249:5555', timeout=3)
print('OK:', ws.recv()[:80])
ws.close()
```

### 检查 ROS topic 数据

```bash
# 查看 EE 数据
rostopic echo /master1_pos_back -n 1

# 查看关节数据
rostopic echo /joint_control -n 1

# 确认频率
rostopic hz /joint_control
```

---

## 14. 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Connection refused` | 桥接未启动 / 端口未映射 | 检查桥接是否运行; Docker 用 `--network host` 或 `-p 5555:5555` |
| 仿真启动时 `Pinocchio` 报错 | 库冲突 | 加 `--disable_pinocchio` |
| 仿真启动时 `Camera` 报错 | 缺少启用标志 | 加 `--enable_cameras` |
| 仿真臂完全不动 | WebSocket 未连通 | 用 debug client 或 Python 测试连通性 |
| 仿真臂动作幅度不对 (EE) | 缩放参数不对 | 检查是否用了 `EX001ArmPhysicalTeleopActionsCfg` (scale=1.0) |
| 夹爪响应慢 | 执行器刚度不够 | 确认 gripper stiffness=200 (已更新) |
| 夹爪不动 | 数据没有 gripper 值 | 用 `--debug` 查看 grip 值 |
| 某个关节方向反 (Joint) | 符号映射不对 | 调整 `--joint_signs` 对应位翻转 |
| 关节 3-6 不动/反向 (Joint) | 默认 signs 不对 | 已默认 `1,1,-1,-1,-1,-1`，如仍不对请微调 |
| 只有左臂动 / 右臂不动 | 右臂 topic 没数据 | `rostopic echo /master2_pos_back` (EE) 或 `/joint_control2` (Joint) 确认 |
| 延迟高 / 不流畅 | 桥接 Hz 太低 | 确保 `--hz 200`，使用千兆网线 |

---

## 15. 关键设计决策

1. **为什么用 WebSocket 而不是 ROS2 bridge?**
   仿真电脑不需要装 ROS，WebSocket 是纯 Python 实现，零依赖，跨平台。

2. **为什么 EE delta 模式用 scale=1.0?**
   物理臂和仿真臂是同型号 (ARX X5)，末端坐标系一致，delta 应该 1:1 映射。
   原来 scale=0.5 是给键盘/VR 设计的虚拟输入衰减。

3. **为什么移除了 RateLimiter?**
   物理臂遥操作时，仿真应尽可能快地处理每一帧。人为限速 (50Hz) 会造成
   动作延迟和不流畅。移除后仿真以自身最大帧率运行。

4. **为什么桥接默认 200Hz?**
   匹配物理臂 ROS topic 的发布频率 (200Hz)，避免数据丢失。

5. **为什么关节 3-6 需要符号翻转?**
   ARX X5 的 URDF 中 joint3 和 joint6 的 `<origin>` 包含 π 旋转
   (`rpy="-3.1416 0 0"`)，转换为 USD 后关节轴向与物理臂相反。
   joint4 和 joint5 在双臂组装过程中也产生了轴向差异。
   通过 `joint_signs=(1,1,-1,-1,-1,-1)` 在软件层面修正。

6. **为什么夹爪刚度设为 200?**
   原来的 stiffness=40 / damping=15 导致夹爪响应迟缓。提高到
   stiffness=200 / damping=8 后，PD 控制器能快速跟踪目标位置，
   按下夹爪后仿真立即响应。
