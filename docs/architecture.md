# ManipBench 架构设计与代码封装规范

本文档定义了 ManipBench 的**代码组织原则**和**扩展封装规范**。所有贡献者在添加新功能时必须遵循本文档描述的模式。

---

## 1. 设计哲学

ManipBench 采用 **"Extension-over-Framework"（扩展覆盖框架）** 的设计理念：

- **Framework（框架）** = Isaac Lab Arena（原版，作为 git submodule 引入，永不修改）
- **Extensions（扩展）** = ManipBench 中定义的所有自定义组件

核心原则：

| 原则 | 说明 |
|------|------|
| **零侵入** | 不修改 Isaac Lab Arena 任何一行源码。所有定制都在 ManipBench 封装层完成。 |
| **单一注册** | 每个扩展通过装饰器或 import 自动注册到框架的全局 registry，无需手动编辑注册表。 |
| **可组合** | 遵循 Arena 的 Embodiment + Scene + Task → Environment 组合模式，扩展之间松耦合。 |
| **自包含** | 每个扩展目录包含自身所需的全部定义（类、配置、辅助函数），外部只需 import 其包。 |

---

## 2. 项目结构总览

```
manip-bench/
├── manip_bench/                  # Python 包（唯一的顶层包）
│   ├── extensions/               # 扩展组件（按类型分组）
│   │   ├── embodiments/          #   机器人定义
│   │   ├── tasks/                #   任务逻辑
│   │   ├── scenes/               #   场景、背景、物体注册
│   │   ├── policies/             #   策略实现
│   │   └── devices/              #   遥操作设备
│   ├── envs/                     # 环境组合（将扩展组装为完整环境）
│   ├── scripts/                  # 入口脚本（面向用户的命令行工具）
│   └── utils/                    # 公共工具函数
│
├── assets/                       # 资产文件（USD, 纹理, 3DGS, git-lfs）
├── configs/                      # YAML 配置文件
├── tools/                        # 独立辅助工具（不依赖仿真环境）
├── docs/                         # 文档
└── third_party/
    └── IsaacLab-Arena/           # 原版框架（git submodule）
```

### 为什么只有一个顶层包？

不同于某些项目将 tasks、rl、policy 拆分为多个独立的顶层包，ManipBench 将所有代码统一在 `manip_bench` 一个命名空间下。原因：

1. **Import 路径清晰**：所有 ManipBench 代码以 `manip_bench.` 开头，一眼可辨
2. **避免命名冲突**：不会和其他项目的 `xxx_tasks`、`xxx_rl` 冲突
3. **安装简单**：一次 `pip install -e .` 安装所有内容

---

## 3. Extension 封装规范

### 3.1 通用模式

所有扩展遵循同一套模式：

```
extensions/<type>/<name>/
├── __init__.py          # 导出公共接口，触发注册
├── <name>.py            # 主定义文件
└── (可选辅助文件)       # actions.py, observations.py 等
```

**规则：**

1. `__init__.py` 必须用 `from .xxx import *` 导出主定义，使 import 包时自动触发装饰器注册
2. 主定义文件中的核心类继承自 Isaac Lab Arena 的对应基类
3. 资产路径通过 `manip_bench.utils.paths` 模块解析，不使用硬编码路径
4. 每个扩展的 `__init__.py` 头部必须包含模块 docstring，说明该扩展的用途

### 3.2 Embodiment（机器人定义）

**基类**：`isaaclab_arena.embodiments.embodiment_base.EmbodimentBase`

**文件结构**：
```
extensions/embodiments/ex001arm/
├── __init__.py
├── ex001arm.py           # EmbodimentBase 子类 + configclass 配置
├── actions.py            # 自定义 ActionTerm（如 ContactLimitedGripperAction）
└── observations.py       # 自定义观测函数
```

**封装要点**：

```python
# ex001arm.py

from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.assets.register import register_asset
from manip_bench.utils.paths import get_robot_usd_path

@register_asset
@configclass
class Ex001ArmBimanual(EmbodimentBase):
    name = "ex001arm_bimanual"
    
    def __init__(self, enable_cameras=False, **kwargs):
        # USD 路径通过 paths 模块解析
        usd_path = get_robot_usd_path("ex001arm_bimanual")
        super().__init__(usd_path=usd_path, ...)
```

- 使用 `@register_asset` 装饰器注册到 Arena 全局 registry
- USD 路径通过 `manip_bench.utils.paths` 解析到 `assets/robots/` 目录
- 所有 configclass 的物理参数（刚度、阻尼、力矩限制等）在类内明确定义

### 3.3 Task（任务定义）

**基类**：`isaaclab_arena.tasks.task_base.TaskBase`

**文件结构**：
```
extensions/tasks/
├── __init__.py
└── put_blocks_to_color.py    # TaskBase 子类
```

**封装要点**：

```python
# put_blocks_to_color.py

from isaaclab_arena.tasks.task_base import TaskBase

class PutBlocksToColorTask(TaskBase):
    """按颜色分拣积木到对应目标区域。"""
    
    # 任务标识
    task_name: str = "put_blocks_to_color"
    
    # 物体配置、成功条件、终止条件
    def get_object_cfgs(self) -> list: ...
    def get_termination_cfg(self): ...
    def get_success_fn(self): ...
```

- 任务类只定义**任务逻辑**（物体配置、成功判定、奖励设计）
- 不包含场景布局或机器人选择——这些在 `envs/` 层组合
- 成功判定函数作为 `TerminationTermCfg` 注册到 MDP

### 3.4 Scene（场景资产注册）

**文件结构**：
```
extensions/scenes/
├── __init__.py
├── backgrounds.py     # 背景场景注册
└── objects.py         # 操作物体注册
```

**封装要点**：

```python
# objects.py

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.register import register_asset
from manip_bench.utils.paths import get_object_usd_path

class ManipBenchObject(Object):
    """ManipBench 物体基类，统一路径解析。"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

@register_asset
class GreenBrick(ManipBenchObject):
    name = "green_brick"
    tags = ["object", "brick"]
    usd_path = get_object_usd_path("hunyuan_assets/green_brick/green_brick_physics.usd")
    scale = (0.0004, 0.0004, 0.0004)
```

- 每个物体/背景类通过 `@register_asset` 自动注册
- import `manip_bench.extensions.scenes` 时即完成所有注册
- 物体按用途用 `tags` 标记，便于任务检索

### 3.5 Policy（策略实现）

**基类**：`isaaclab_arena.policy.policy_base.PolicyBase`

```
extensions/policies/
├── __init__.py
└── closedloop_policy.py
```

- 策略类实现 `PolicyBase` 接口的 `get_action()` 方法
- 支持远程推理服务器（WebSocket / gRPC）

### 3.6 Device（遥操作设备）

**基类**：Isaac Lab 的 `DeviceBase` 或 Arena 的设备接口

```
extensions/devices/
├── __init__.py
├── openxr_bimanual.py     # VR 双手控制器
└── ws_remote.py           # WebSocket 远程控制（真实机械臂）
```

- 使用 Arena 的 device registry 注册
- 设备类提供统一的 `advance()` → `(delta_pose, gripper)` 接口

---

## 4. Environment 组合层（envs/）

`envs/` 是**唯一允许将多种扩展组装在一起**的层。每个文件定义一个完整的实验环境配置。

**文件结构**：
```
envs/
├── __init__.py                          # gymnasium.register() 注册
├── cvpr_pick_and_place.py               # CVPR 场景 + 抓取任务
├── cvpr_put_blocks_to_color.py          # CVPR 场景 + 色块分拣任务
└── kitchen_pick_and_place.py            # 厨房场景 + 抓取任务
```

**封装要点**：

```python
# cvpr_pick_and_place.py

from isaaclab_arena.examples.example_environments.example_environment_base import (
    ExampleEnvironmentBase,
)

class CvprPickAndPlace(ExampleEnvironmentBase):
    """CVPR 展示场景下的双臂抓取任务。"""
    name: str = "cvpr_pick_and_place"

    def get_env(self, args_cli):
        # 在此组合: Embodiment + Scene + Task → IsaacLabArenaEnvironment
        from isaaclab_arena.environments.isaaclab_arena_environment import (
            IsaacLabArenaEnvironment,
        )
        from manip_bench.extensions.embodiments.ex001arm import Ex001ArmBimanual
        from manip_bench.extensions.scenes.backgrounds import CvprBackground
        from manip_bench.extensions.tasks.pick_and_place import PickAndPlaceTask
        
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=Ex001ArmBimanual(...),
            scene=Scene(background=CvprBackground(), objects=[...]),
            task=PickAndPlaceTask(...),
        )
```

**组合规则**：
- 只在 `envs/` 层做 import 组合，extensions 之间不互相 import
- 延迟 import（在 `get_env()` 方法内），避免仿真引擎启动前导入 Omniverse 模块
- `__init__.py` 中用 `gymnasium.register()` 注册每个环境，使其可通过 gym API 访问

---

## 5. Scripts 入口层

`scripts/` 包含面向用户的命令行入口点。

**组织原则**：
- 每个脚本对应一个**工作流动作**（run, record, replay, convert, evaluate）
- 脚本只做参数解析和流程编排，业务逻辑在 extensions 中
- 脚本开头统一处理 `AppLauncher`（Isaac Sim 启动器）

```
scripts/
├── run_env.py           # 运行环境（带 GUI 可视化）
├── record.py            # 录制演示数据
├── replay.py            # 回放并验证演示
├── teleop_keyboard.py   # 键盘遥操作
├── teleop_real_arm.py   # 真实机械臂遥操作
└── convert_data.py      # 数据格式转换（HDF5 → LeRobot）
```

**脚本模板**：

```python
"""ManipBench: <功能描述>"""

from isaaclab.app import AppLauncher

# --- 参数解析（必须在 AppLauncher 之前） ---
import argparse
parser = argparse.ArgumentParser(description="...")
parser.add_argument("--env", type=str, required=True, help="环境名称")
# ... 其他参数
args = parser.parse_args()

# --- 启动仿真 ---
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- 业务逻辑（仿真启动后才能 import） ---
import manip_bench.extensions  # 触发所有扩展注册
from manip_bench.envs import get_environment
# ... 后续逻辑
```

---

## 6. 路径解析规范

所有资产路径通过 `manip_bench.utils.paths` 模块统一解析：

```python
# manip_bench/utils/paths.py

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSETS_ROOT = _PROJECT_ROOT / "assets"

def get_project_root() -> Path:
    return _PROJECT_ROOT

def get_assets_root() -> Path:
    return _ASSETS_ROOT

def get_robot_usd_path(robot_name: str) -> str:
    return (_ASSETS_ROOT / "robots" / robot_name).as_posix()

def get_object_usd_path(relative_path: str) -> str:
    return (_ASSETS_ROOT / "objects" / relative_path).as_posix()

def get_scene_usd_path(relative_path: str) -> str:
    return (_ASSETS_ROOT / "scenes" / relative_path).as_posix()
```

**规则**：
- 禁止在扩展代码中使用 `Path(__file__).parent` 拼接资产路径
- 统一使用 `get_*_usd_path()` 系列函数
- 函数返回 `str`（posix 格式），兼容 Isaac Lab 的路径要求

---

## 7. 配置文件规范

配置文件存放在 `configs/` 目录，按工作流阶段组织：

```
configs/
├── teleop/
│   ├── keyboard.yaml       # 键盘遥操作配置
│   └── vr.yaml             # VR 遥操作配置
└── eval/
    └── default.yaml         # 默认评估配置
```

**YAML 配置格式**：

```yaml
# configs/teleop/keyboard.yaml

env:
  name: cvpr_pick_and_place
  num_envs: 1

embodiment:
  name: ex001arm_bimanual
  enable_cameras: true

record:
  enabled: true
  output_dir: ./recordings
  format: hdf5
```

- 配置只做**参数覆盖**，默认值在代码 dataclass 中定义
- 不使用配置继承（`_base_`），保持配置文件自解释性
- 每个配置文件顶部注释说明用途

---

## 8. 数据流与层级关系

```
┌─────────────────────────────────────────────────────────────────┐
│                     scripts/  (用户入口)                         │
│  run_env.py  record.py  replay.py  teleop_*.py  convert_data.py │
└──────────────────────┬──────────────────────────────────────────┘
                       │ 调用
┌──────────────────────▼──────────────────────────────────────────┐
│                      envs/  (环境组合层)                         │
│  将 embodiment + scene + task 组合成 IsaacLabArenaEnvironment    │
└──────┬──────────┬───────────┬──────────┬───────────┬────────────┘
       │          │           │          │           │
┌──────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────┐
│embodiment│ │ tasks  │ │ scenes │ │policies│ │ devices │
│ (机器人)  │ │ (任务)  │ │ (场景)  │ │ (策略)  │ │ (设备)   │
└──────┬───┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬─────┘
       │         │          │          │           │
       └─────────┴──────────┴──────────┴───────────┘
                            │
               ┌────────────▼────────────┐
               │  Isaac Lab Arena (原版)   │
               │  third_party/ submodule  │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │  Isaac Lab / Isaac Sim   │
               └─────────────────────────┘
```

**调用方向**：自上而下，不可逆向。即：
- `scripts` 可以调用 `envs` 和 `extensions`
- `envs` 可以调用 `extensions`
- `extensions` 只调用 Isaac Lab Arena 基类和 `utils`
- **禁止 `extensions` 之间的横向 import**（除 `utils` 外）

---

## 9. 添加新组件的标准流程

### 添加新机器人

1. 在 `manip_bench/extensions/embodiments/` 下创建新目录
2. 实现 `EmbodimentBase` 子类，使用 `@register_asset` 装饰器
3. 将 USD 资产放入 `assets/robots/<name>/`
4. 在 `__init__.py` 中导出
5. 在 `envs/` 中创建使用该机器人的环境组合

### 添加新任务

1. 在 `manip_bench/extensions/tasks/` 下创建新文件
2. 实现 `TaskBase` 子类，定义成功条件和物体配置
3. 如需新物体，在 `extensions/scenes/objects.py` 中注册
4. 在 `envs/` 中创建使用该任务的环境组合

### 添加新遥操作设备

1. 在 `manip_bench/extensions/devices/` 下创建新文件
2. 实现设备接口（`advance()` 方法）
3. 注册到 Arena 的 device registry

---

## 10. 与 Isaac Lab Arena 的接口边界

ManipBench 依赖 Isaac Lab Arena 的以下公共接口（如果 Arena 更新，只需检查这些接口是否兼容）：

| Arena 接口 | ManipBench 使用方式 |
|------------|-------------------|
| `EmbodimentBase` | 继承，定义自定义机器人 |
| `TaskBase` | 继承，定义自定义任务 |
| `Scene` | 实例化，传入背景和物体 |
| `IsaacLabArenaEnvironment` | 实例化，组合 embodiment + scene + task |
| `ArenaEnvBuilder` | 用于构建 ManagerBasedRLEnv |
| `ExampleEnvironmentBase` | 继承，定义环境组合 |
| `@register_asset` | 装饰器，注册资产到全局 registry |
| `PolicyBase` | 继承，实现自定义策略 |
| `AssetRegistry` | 查询已注册资产 |
| `DeviceRegistry` | 查询已注册设备 |

---

## 总结

ManipBench 的封装核心是 **"Extensions + Composition"**：

- **Extensions** 负责定义独立的组件（机器人、任务、场景、策略、设备）
- **Envs** 负责将组件组合为完整的实验环境
- **Scripts** 负责面向用户的工作流入口
- **Isaac Lab Arena** 作为不可变的底层框架提供基础设施

这种分层设计使得添加新机器人、新任务或新场景时，只需在对应的 extensions 目录下添加文件，不需要修改任何已有代码。
