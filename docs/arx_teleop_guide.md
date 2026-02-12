# EX001Arm 远程遥操作指南 (WebSocket)

## 概述

通过 WebSocket 实现双机远程遥操作数据采集。机器人电脑读取物理臂状态并通过网络发送，仿真电脑接收后驱动 Isaac Sim 中的 ex001arm 模型。

```
┌────────────────────────────────────┐           ┌──────────────────────────────┐
│       机器人电脑 (Docker/ROS1)      │   网线     │      仿真电脑 (Isaac Sim)     │
│                                    │           │                              │
│  物理双臂 → ROS1 PosCmd topics     │           │                              │
│       ↓                            │           │                              │
│  ros1_ws_bridge.py                 │  ──────►  │  Ex001ArmWsRemoteTeleop      │
│  (订阅topic → WebSocket推送)        │  ws://    │  (接收 → 计算delta → 14D)    │
│                                    │  :5555    │         ↓                    │
│  pip install websockets            │           │     Isaac Sim 仿真环境        │
│                                    │           │  pip install websocket-client │
└────────────────────────────────────┘           └──────────────────────────────┘
```

## 数据流

1. 物理臂通过 CAN 总线发布 ROS1 topic (`/master1_pos_back`, `/master2_pos_back`)
2. 桥接脚本订阅 topic，以 50Hz 推送 JSON 到 WebSocket
3. 仿真端接收 JSON，计算帧间 delta 位姿
4. DifferentialIK 控制器驱动仿真臂跟随

## ROS1 消息格式

| Topic | 消息类型 | 说明 |
|-------|----------|------|
| `/master1_pos_back` | `arm_control/PosCmd` | 左臂 (master1) |
| `/master2_pos_back` | `arm_control/PosCmd` | 右臂 (master2) |

`PosCmd` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `x`, `y`, `z` | float | 末端位置 (米) |
| `roll`, `pitch`, `yaw` | float | 末端姿态 (Euler 角, 弧度) |
| `gripper` | float | 夹爪开合 (0=闭合, 4.5=张开) |
| `mode1`, `mode2` | int | 模式标志 |

发布频率：200Hz

## 文件清单

```
IsaacLab-Arena/
├── tools/
│   └── ros1_ws_bridge.py              # 机器人电脑端 (ROS1 → WebSocket)
├── isaaclab_arena/
│   ├── teleop_devices/
│   │   ├── __init__.py                # 已更新导入
│   │   └── ex001arm_ws_remote.py      # 仿真端 WebSocket 遥操作设备
│   ├── embodiments/ex001arm/
│   │   └── ex001arm.py                # 含 EX001ArmJointActionsCfg (备用)
│   └── scripts/
│       └── record_ex001_remote_demos.py  # 远程录制脚本
└── docs/
    └── arx_teleop_guide.md            # 本文档
```

## 环境准备

### 机器人电脑 (Docker 容器内)

```bash
# 安装 WebSocket 库
pip install websockets

# 将 ros1_ws_bridge.py 复制到容器内 (或挂载)
# 确保 arm_control 消息包已编译并在 ROS 环境中
```

Docker 运行时需要映射端口：

```bash
docker run -p 5555:5555 ...
# 或在 docker-compose.yml 中添加:
# ports:
#   - "5555:5555"
```

### 仿真电脑

```bash
# 在 Isaac Sim Python 环境中安装
pip install websocket-client
```

### 网络配置

两台电脑需要在同一局域网中能互相访问：

```bash
# 在仿真电脑上测试连通性
ping <机器人电脑IP>

# 测试端口是否可达 (桥接启动后)
python -c "import websocket; ws = websocket.create_connection('ws://<IP>:5555'); print('OK'); ws.close()"
```

## 使用步骤

### 1. 启动机器人端

```bash
# 进入 Docker 容器
dexec

# 先确保 ROS master 和机械臂节点已运行
# 然后启动桥接脚本
python ros1_ws_bridge.py --port 5555 --hz 50
```

看到以下输出表示成功：
```
[ws_bridge] Subscribing: left=/master1_pos_back  right=/master2_pos_back
[ws_bridge] WebSocket server listening on 0.0.0.0:5555  (50 Hz)
```

### 2. 启动仿真端

```bash
python -m isaaclab_arena.scripts.record_ex001_remote_demos \
    --embodiment ex001arm \
    --task stack \
    --remote_ip <机器人电脑IP> \
    --remote_port 5555
```

### 3. 操作

- 脚本启动后自动开始录制
- 手动操作物理臂，仿真臂实时跟随
- 按 `R` 重置环境（不导出数据）
- 任务成功后自动导出 HDF5 + 重置
- `Ctrl+C` 结束

## 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--remote_ip` | `192.168.1.100` | 机器人电脑 IP |
| `--remote_port` | `5555` | WebSocket 端口 |
| `--teleop_hz` | `50` | 遥操作频率 (Hz) |
| `--dataset_file` | `./demos` | 数据输出目录 |
| `--reset_duration` | `10.0` | 自动重置时长 (秒) |

桥接脚本参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `5555` | WebSocket 监听端口 |
| `--hz` | `50` | 推送频率 (Hz) |
| `--left_topic` | `/master1_pos_back` | 左臂 PosCmd topic |
| `--right_topic` | `/master2_pos_back` | 右臂 PosCmd topic |

## 数据输出

每条轨迹保存为独立 HDF5 文件：

```
demos/
├── arx_remote_ee_episode0.hdf5
├── arx_remote_ee_episode1.hdf5
└── ...
```

## 技术细节

### 14D 动作张量格式

```
[left_dx, left_dy, left_dz, left_drx, left_dry, left_drz, left_grip,
 right_dx, right_dy, right_dz, right_drx, right_dry, right_drz, right_grip]
```

- 位置 delta：当前帧与上一帧 EE 位置之差（米）
- 旋转 delta：从连续 Euler 角帧计算旋转向量（`R_curr * R_prev^-1 → rotvec`）
- 夹爪：`-1`（闭合）到 `+1`（张开），由 `gripper / 4.5 * 2 - 1` 映射

### WebSocket 协议

JSON 格式，每帧推送：

```json
{
  "timestamp": 1234567890.123,
  "left": {
    "x": 0.3, "y": 0.0, "z": 0.4,
    "roll": 0.0, "pitch": 1.57, "yaw": 0.0,
    "gripper": 2.0, "mode1": 0, "mode2": 0
  },
  "right": {
    "x": 0.3, "y": 0.0, "z": 0.4,
    "roll": 0.0, "pitch": 1.57, "yaw": 0.0,
    "gripper": 2.0, "mode1": 0, "mode2": 0
  }
}
```

## 故障排查

| 问题 | 可能原因 | 解决方法 |
|------|----------|----------|
| `Connection refused` | 桥接脚本未启动或端口未映射 | 检查 Docker 端口映射 `-p 5555:5555` |
| `[WS Teleop] Disconnected` | 网络断开 | 设备会自动重连，检查网线/IP |
| 仿真臂不动 | 桥接收不到 ROS topic | 在容器内 `rostopic echo /master1_pos_back` 确认有数据 |
| 仿真臂漂移 | 坐标系不对齐 | 确认物理臂基座朝向与仿真一致 |
| 数据是 None | 机器人节点未启动 | 先启动 ctrl1 再启动桥接脚本 |
| 延迟过高 | 网络带宽不足或 Hz 设过高 | 降低 `--hz`，使用千兆网线 |
