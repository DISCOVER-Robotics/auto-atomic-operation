# IK Control for Robot Arms

本文档说明 auto_atom 框架中基于逆运动学 (IK) 的机械臂关节空间控制的实现方案和参数配置。

## 概述

框架支持两种 operator 控制模式：

| 模式 | 触发条件 | 控制方式 |
|------|----------|----------|
| **Mocap** | `arm_actuators: []` | 浮动基座 + weld 约束，运动学驱动 |
| **Joint** | `arm_actuators: [a1, a2, ...]` | IK 求解 + PD 关节位控，动力学驱动 |

当 YAML 中 `arm_actuators` 非空时，自动进入 Joint 模式。此时需要提供 `IKSolver` 实例。

## 控制链路

```
TaskRunner.update()
  └─ operator.move_to_pose(target_eef_pose_world)
      └─ env.world_to_base(target) → target_eef_pose_base
          └─ env.step_operator_toward_target(target_eef_pose_base)
              ├─ ik_solver.solve(target_pose_base, current_qpos)
              │    ├─ base→world 坐标变换
              │    ├─ mink 微分IK迭代求解
              │    └─ max_joint_delta clamp（防奇异点跳变）
              ├─ ctrl[arm_aidx] = clamped_joint_targets
              └─ env.step(ctrl)  # PD控制器驱动关节
```

### 关键设计

**每步求解 + delta clamp（而非一次求解 + 插值）：**

1. 每个控制周期从**当前 qpos** 出发精确求解 IK（300 迭代）
2. 求解完成后检查最大关节位移：若超过 `max_joint_delta`，等比缩放整个 delta 向量
3. 缩放后的关节目标写入 actuator ctrl，PD 控制器在 substeps 内跟踪
4. 由于单次 solve 的 delta 被限制，下一步会从新的 qpos 继续求解，逐步逼近目标

**为什么不用一次求解 + 关节插值？**

关节空间线性插值的笛卡尔轨迹不可预测——中间点可能偏离直线或经过障碍物。
每步重新求解 IK 保证了笛卡尔空间的连续追踪，且 `max_joint_delta` clamp
在关节空间限制了速度，避免奇异点附近的解分支跳变。

## IK Solver 实现：MinkIKSolver

位于 [`auto_atom/backend/mjc/ik/mink_ik_solver.py`](../auto_atom/backend/mjc/ik/mink_ik_solver.py)，
基于 [mink](https://github.com/kevinzakka/mink) 微分 IK 库。

### 求解过程

```python
def solve(target_pose_in_base, current_qpos) -> Optional[np.ndarray]:
    # 1. 坐标变换：base frame → world frame（mink 在 world frame 工作）
    pos_w = R_base @ pos_b + base_pos
    quat_w = quat_base ⊗ quat_b

    # 2. 设置 mink 目标 SE3
    eef_task.set_target(SE3(R_w, pos_w))

    # 3. 用 current_qpos 初始化 mink Configuration
    configuration.update(q_seed)
    posture_task.set_target_from_configuration(configuration)  # 动态 posture target

    # 4. 迭代求解
    for _ in range(n_iterations):
        vel = mink.solve_ik(configuration, [eef_task, posture_task], dt, ...)
        configuration.integrate_inplace(vel, dt)

    # 5. Clamp：限制最大关节位移
    delta = solved - current_qpos
    if max(|delta|) > max_joint_delta:
        solved = current_qpos + delta * (max_joint_delta / max(|delta|))
    return solved
```

### posture_task 的作用

每次 solve 时，posture target 被更新为当前 seed（即 current_qpos）。这意味着：

- IK 在满足末端目标的前提下，倾向于保持关节接近当前配置
- 防止求解器跳到等价但关节差异很大的另一个 IK 分支
- `posture_cost` 控制这个约束的强度（越大越保守，但可能导致末端精度下降）

## YAML 配置

### 基础配置：base_franka.yaml

```yaml
env:
  env:
    config:
      operators:
        - name: arm
          arm_actuators: [actuator1, actuator2, ..., actuator7]  # 触发 joint 模式
          eef_actuators: [fingers_actuator]
          pose_site: gripper        # EEF 位姿读取 site
      sim_freq: 500
      update_freq: 100              # 每个控制步的物理 substeps = sim_freq / update_freq

backend: auto_atom.backend.mjc.ik.mink_ik_solver.build_franka_backend

operators:
  - name: arm
    ik:                             # IK 超参（可选，均有默认值）
      n_iterations: 300
      dt: 0.1
      position_cost: 1.0
      orientation_cost: 1.0
      posture_cost: 1e-4
      max_joint_delta: 0.8
```

### IK 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_iterations` | 300 | 每次 solve 的 mink 迭代步数。越大求解越精确，但越慢 |
| `dt` | 0.1 | 每个 IK 迭代的虚拟时间步（秒）。`n_iterations × dt` = 总积分时长 |
| `position_cost` | 1.0 | EEF 位置跟踪权重 |
| `orientation_cost` | 1.0 | EEF 姿态跟踪权重。增大可提高姿态精度 |
| `posture_cost` | 1e-4 | 关节姿态正则化权重。增大使关节更保守（不易跳变），但降低末端精度 |
| `max_joint_delta` | 0.8 | 单次 solve 允许的最大关节位移（rad）。防止奇异点附近的解跳变 |

### 参数调优指南

**机械臂运动太慢：**
- 增大 `max_joint_delta`（如 1.2），允许每步走更远
- 降低 `update_freq`（如 50），增加每步的物理仿真时间，使 PD 控制器有更多时间跟踪

**经过奇异点时关节跳变 / 末端翻转：**
- 减小 `max_joint_delta`（如 0.5），限制关节速度
- 增大 `posture_cost`（如 1e-3），使 IK 更倾向于保持当前关节构型
- 调整 keyframe 中的初始关节角，使 home 配置远离奇异区域

**末端姿态不准确：**
- 增大 `orientation_cost`（如 2.0）
- 增大 `n_iterations`（如 500），给更多迭代时间
- 减小 `posture_cost`（如 1e-5），放松关节约束

**IK 求解太慢（影响实时性）：**
- 减小 `n_iterations`（如 100），但需确保精度足够
- 增大 `dt`（如 0.2），每步走更远但可能不稳定

### 任务配置中的注意事项

对于 Franka 等固定基座机械臂：

1. **所有 waypoint 都应显式指定 orientation**——如果省略，IK 可能求出不同的腕关节构型
2. **keyframe 中的 joint7 应接近任务所需的末端朝向**——避免首次移动时大幅旋转
3. **`base_pose` 应匹配 XML 中机械臂底座的实际位置**

```yaml
operators:
  - name: arm
    initial_state:
      base_pose:
        position: [-0.45, -0.06, 0.0]  # 与 XML 中 link0 位置一致
        orientation: [0, 0, 0, 1]
```

## 自定义 IK Solver

实现 `IKSolver` 协议即可替换 mink：

```python
from auto_atom.runtime import IKSolver
from auto_atom.utils.pose import PoseState

class MyIKSolver:
    def solve(
        self,
        target_pose_in_base: PoseState,  # 基座系下的末端目标位姿 (xyzw)
        current_qpos: np.ndarray,         # 当前关节角
    ) -> Optional[np.ndarray]:            # 目标关节角，无解返回 None
        ...
```

然后编写自己的 `build_*_backend` 工厂函数，在 YAML 的 `backend` 字段中引用。
