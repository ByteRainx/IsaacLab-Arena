# EX001Arm 远程遥操作文档 — EE 末端控制模式

## 1. 总体架构

两台电脑通过网线连接，使用 WebSocket 传输机械臂状态数据。

```
┌─────────────────────────────────────────┐            ┌──────────────────────────────────────────┐
│         机器人电脑 (arm-pc)              │            │          仿真电脑 (Isaac Sim)             │
│         Docker 容器 / ROS1              │            │                                          │
│                                         │            │                                          │
│  物理双臂 (CAN 总线)                     │            │                                          │
│       ↓                                 │            │                                          │
│  ROS1 topics (200Hz):                   │            │                                          │
│    /master1_pos_back (左臂 EE)          │            │                                          │
│    /master2_pos_back (右臂 EE)          │            │                                          │
│       ↓                                 │            │                                          │
│  ros1_ws_bridge.py                      │  WebSocket │  ex001arm_ws_remote.py                   │
│  (订阅 ROS topic → JSON → WS推送)       │ ─────────► │  (接收 JSON → 计算 delta → 14D action)   │
│  默认: ws://0.0.0.0:5555  200Hz        │  :5555     │       ↓                                  │
│                                         │            │  DifferentialIK (scale=1.0)              │
│  依赖: pip install websockets           │            │       ↓                                  │
│                                         │            │  Isaac Sim 仿真环境                       │
│                                         │            │  record_ex001_remote_demos.py             │
│                                         │            │  依赖: pip install websocket-client       │
└─────────────────────────────────────────┘            └──────────────────────────────────────────┘
```

## 2. 核心数据流 (EE 模式)

```
物理臂编码器
   ↓ (CAN总线)
ROS1 节点 → 发布 arm_control/PosCmd (200Hz)
   ↓
ros1_ws_bridge.py (机器人端)
   │  订阅 /master1_pos_back (左臂) 和 /master2_pos_back (右臂)
   │  每次收到消息更新内部缓存 (线程安全)
   │  以 --hz 频率 (默认200) 把缓存快照推送给 WebSocket 客户端
   ↓
WebSocket JSON 传输 (局域网)
   ↓
Ex001ArmWsRemoteTeleop (仿真端)
   │  后台线程持续接收 JSON，存入 self._latest
   │  每次 advance() 被调用时：
   │    1. 读取最新的 left/right EE pose
   │    2. 与上一帧比较，计算 delta
   │    3. 映射夹爪值
   │    4. 输出 14D action tensor
   ↓
DifferentialIK 控制器 (Isaac Lab)
   │  接收 14D action = [6D_left_delta + grip, 6D_right_delta + grip]
   │  通过 DLS 逆运动学求解关节速度
   │  驱动仿真关节到目标位姿
   ↓
Isaac Sim 渲染 + 物理仿真
```

## 3. ROS1 消息格式

### 话题

| Topic | 消息类型 | 说明 |
|-------|----------|------|
| `/master1_pos_back` | `arm_control/PosCmd` | 左臂 master arm |
| `/master2_pos_back` | `arm_control/PosCmd` | 右臂 master arm |

### PosCmd 字段

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

发布频率: **200Hz**

## 4. WebSocket 传输协议

### JSON Payload (EE 模式)

桥接脚本以 `--hz` 指定的频率推送：

```json
{
  "timestamp": 1770898856.993228,
  "left": {
    "x": -0.0080, "y": 0.0180, "z": 0.0021,
    "roll": -0.0421, "pitch": 0.1579, "yaw": 0.3261,
    "gripper": 0.0, "mode1": 0, "mode2": 0
  },
  "right": {
    "x": -0.0014, "y": 0.0074, "z": 0.0116,
    "roll": 0.0537, "pitch": -0.0776, "yaw": 0.0847,
    "gripper": 4.5, "mode1": 0, "mode2": 0
  }
}
```

## 5. Delta 计算详解 (仿真端核心逻辑)

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

`pos_scale` 和 `rot_scale` 保留作为微调手段，理论上同型号臂不需要调。

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

## 6. DifferentialIK 控制器

14D action 送入 Isaac Lab 的 `DifferentialInverseKinematicsActionCfg`：

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

**`scale=1.0` 的意义**: 你发送的 delta 不会被缩放，物理臂移动 1cm，仿真臂也移动 1cm。

夹爪使用 `ContactLimitedGripperActionCfg`：
- `-1` → 闭合 (joint position 0.0)
- `+1` → 张开 (joint position 5.0)
- 带接触力限制 (15N)，防止夹碎物体

## 7. 首帧处理与 Reset

- **首帧**: `_prev_left = None` 时，记录当前位姿作为基准，输出零向量（不产生跳变）
- **Reset**: 调用 `teleop.reset()` 会清空 `_prev_left` 和 `_prev_right`，下一帧重新初始化基准
- **断线重连**: WebSocket 断开后后台线程自动每 2 秒尝试重连

## 8. 文件清单

| 文件 | 运行位置 | 功能 |
|------|----------|------|
| `tools/ros1_ws_bridge.py` | 机器人电脑 (Docker) | ROS1 → WebSocket 桥接 |
| `isaaclab_arena/teleop_devices/ex001arm_ws_remote.py` | 仿真电脑 | WebSocket 接收 + delta 计算 |
| `isaaclab_arena/scripts/record_ex001_remote_demos.py` | 仿真电脑 | 录制脚本 (启动仿真+遥操作) |
| `isaaclab_arena/embodiments/ex001arm/ex001arm.py` | 仿真电脑 | Action config 定义 |
| `tools/ws_debug_client.py` | 仿真电脑 | 独立诊断工具 (查看原始数据) |

## 9. 使用步骤

### 9.1 机器人端 (Docker 容器内)

```bash
# 1. 确保 ROS master 和机械臂控制器已启动 (按 ctrl1 等)
# 2. 进入容器
dexec

# 3. 安装依赖 (首次)
pip install websockets

# 4. 启动桥接
python ros1_ws_bridge.py --hz 200
```

成功输出:
```
[ws_bridge] EE topics: left=/master1_pos_back  right=/master2_pos_back
[ws_bridge] WebSocket server listening on 0.0.0.0:5555  (mode=ee, 200 Hz)
```

> **注意**: Docker 使用 `--network host` 模式时不需要端口映射。
> 否则需要 `docker run -p 5555:5555 ...`

### 9.2 仿真端

```bash
# 标准命令
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --disable_pinocchio \
    --enable_cameras \
    --remote_ip 10.100.21.249 \
    --dataset_file ./demos \
    ex001arm_cvpr_scene_put_blocks_to_color

# 带调试输出
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --disable_pinocchio \
    --enable_cameras \
    --remote_ip 10.100.21.249 \
    --dataset_file ./demos \
    --debug \
    ex001arm_cvpr_scene_put_blocks_to_color
```

### 9.3 操作

- 脚本启动后自动开始录制
- 手动操作物理臂，仿真臂实时跟随
- 按 `R` 重置环境（不导出数据）
- 任务成功后自动导出 HDF5 + 重置
- `Ctrl+C` 结束

## 10. 完整参数列表

### 桥接脚本 (`ros1_ws_bridge.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `5555` | WebSocket 监听端口 |
| `--hz` | `200` | 推送频率 (Hz) |
| `--mode` | `ee` | 数据模式: `ee` / `joint` / `both` |
| `--left_topic` | `/master1_pos_back` | 左臂 PosCmd topic |
| `--right_topic` | `/master2_pos_back` | 右臂 PosCmd topic |
| `--joint_left_topic` | `/joint_control` | 左臂 JointControl topic |
| `--joint_right_topic` | `/joint_control2` | 右臂 JointControl topic |

### 录制脚本 (`record_ex001_remote_demos.py`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote_ip` | `192.168.1.100` | 机器人电脑 IP |
| `--remote_port` | `5555` | WebSocket 端口 |
| `--control_mode` | `ee` | 控制模式: `ee` (末端delta) / `joint` (关节直通) |
| `--pos_scale` | `1.0` | 位置 delta 缩放 (EE模式) |
| `--rot_scale` | `1.0` | 旋转 delta 缩放 (EE模式) |
| `--debug` | `false` | 打印调试信息 |
| `--dataset_file` | `./demos` | 数据输出目录 |
| `--reset_duration` | `10.0` | 自动重置时长 (秒) |
| `--disable_pinocchio` | - | 禁用 Pinocchio (避免库冲突) |
| `--enable_cameras` | - | 启用仿真相机 |

## 11. 数据输出

每条轨迹保存为独立 HDF5 文件:

```
demos/
├── arx_remote_ee_episode0.hdf5
├── arx_remote_ee_episode1.hdf5
└── ...
```

## 12. 诊断工具

### WebSocket 调试客户端

不启动仿真，直接查看桥接推送的原始数据:

```bash
python tools/ws_debug_client.py --ip 10.100.21.249 --port 5555
```

输出示例:
```
Frame #50
  LEFT  (master1):  x=-0.0080  y=0.0180  z=0.0021
          roll=-0.0421  pitch=0.1579  yaw=0.3261
          gripper=0.0000
          delta_pos: dx=0.000011  dy=-0.000044  dz=-0.000000
  RIGHT (master2):  x=-0.0014  y=0.0074  z=0.0116
          ...
```

### Python 快速连通性测试

```python
import websocket
ws = websocket.create_connection('ws://10.100.21.249:5555', timeout=3)
print('OK:', ws.recv()[:80])
ws.close()
```

## 13. 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Connection refused` | 桥接未启动 / 端口未映射 | 检查桥接是否运行; Docker 用 `--network host` 或 `-p 5555:5555` |
| 仿真启动时 `Pinocchio` 报错 | 库冲突 | 加 `--disable_pinocchio` |
| 仿真启动时 `Camera` 报错 | 缺少 `--enable_cameras` | 加 `--enable_cameras` |
| 仿真臂完全不动 | WebSocket 未连通 | 用 debug client 或 Python 测试连通性 |
| 仿真臂动作幅度不对 | `pos_scale` / `rot_scale` | 检查是否用了 `EX001ArmPhysicalTeleopActionsCfg` (scale=1.0) |
| 夹爪不动 | 数据没有 gripper 值 | 用 `--debug` 查看 `grip_raw` 值 |
| 只有左臂动 / 右臂不动 | 右臂 topic 没数据 | `rostopic echo /master2_pos_back` 确认 |
| 延迟高 / 不流畅 | 桥接 Hz 太低 | 确保 `--hz 200`，使用千兆网线 |

## 14. 关键设计决策记录

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
