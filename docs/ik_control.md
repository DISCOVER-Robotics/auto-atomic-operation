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
      ├─ 若配置了 Cartesian step 限幅:
      │    ├─ 当前位置 → 目标位姿做笛卡尔分段
      │    ├─ 位置按直线小步逼近
      │    └─ 姿态按 SLERP 小步逼近
      └─ env.world_to_base(target) → target_eef_pose_base
          └─ env.step_operator_toward_target(target_eef_pose_base)
              ├─ 首次到达该目标时 ik_solver.solve(target_pose_base, current_qpos)
              │    ├─ base→world 坐标变换
              │    ├─ mink 微分IK迭代求解
              │    └─ max_joint_delta clamp（防奇异点跳变）
              ├─ 若 `joint_control_mode=solve_once_interpolate`
              │    └─ 在多个 control step 中对 solved_joint_targets 做线性插值
              ├─ 若 `joint_control_mode=per_step_ik`
              │    └─ 每个 control step 都重新 solve 一次
              ├─ ctrl[arm_aidx] = current_step_joint_targets
              └─ env.step(ctrl)  # PD控制器驱动关节
```

### 关键设计

框架现在支持两种 joint mode 执行策略：

| 策略 | 配置值 | 行为 |
|------|--------|------|
| 每步重求 IK | `per_step_ik` | 每个控制周期都从当前 qpos 出发重新做一次 IK |
| 一次求解 + 关节插值 | `solve_once_interpolate` | 目标改变时只解一次 IK，再按关节位移大小自适应计算插值步数 |

此外，pose 控制现在还支持独立于 joint mode 的笛卡尔分段：

- 位置按 operator 默认的 `control.cartesian_max_linear_step`，或 waypoint 自己的 `max_linear_step` 做直线分段
- 姿态按 operator 默认的 `control.cartesian_max_angular_step`，或 waypoint 自己的 `max_angular_step` 做 SLERP 分段

这层分段发生在 IK 之前，目的是约束末端轨迹形状，而不是只约束关节轨迹形状。

#### 一次求解 + 关节插值

这是当前 Franka 示例任务推荐的模式，`examples/mujoco/pick_and_place_franka.yaml`
默认使用它。

执行过程：

1. 当目标 EEF pose 发生变化时，从**当前 qpos** 出发求一次 IK
2. 求解完成后检查最大关节位移，若超过 `max_joint_delta` 则整体缩放 delta
3. 将“当前关节角 -> IK 解”缓存成一条关节轨迹
4. 根据 `max(abs(q_target - q_current)) / joint_interp_speed` 自适应计算插值步数
5. 在后续这些 control step 中做线性插值
6. 每一步把插值结果写入 actuator ctrl，由 PD 控制器跟踪

这个策略的特点是：

- IK 求解次数更少
- 关节目标变化更平滑，更接近“给一条 joint trajectory”
- 小位移自动用更少插值步，大位移自动分更多步
- 末端轨迹不再是严格意义上的每步笛卡尔重跟踪，而是“先规划一个终点，再在关节空间执行过去”

#### 每步重求 IK

这是之前框架中的默认逻辑：

1. 每个控制周期从**当前 qpos** 出发重新求一次 IK
2. 求解完成后做 `max_joint_delta` clamp
3. 将该步的关节目标直接写入 actuator ctrl
4. 下一步再从新的 qpos 继续 solve

这个策略的特点是：

- 更接近连续笛卡尔跟踪
- 目标变化时响应直接
- IK 调用频率更高
- 在某些姿态附近更容易看到“每步都在修正”的控制风格

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

注意：`MinkIKSolver.solve()` 仍然只负责“求一个目标关节角”。
“一次求解后是否继续做关节插值执行”是在 `UnifiedMujocoEnv.step_operator_toward_target()`
这一层决定的，而不是在 solver 内部完成的。

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

stages:
  - name: pick_source
    param:
      pre_move:
        - position: [0.0, 0.0, 0.12]
          orientation: [-0.7071, 0.7071, 0.0, 0.0]
          reference: object_world
          max_linear_step: 0.02
          max_angular_step: 0.18
        - position: [0.0, 0.0, 0.006]
          orientation: [-0.7071, 0.7071, 0.0, 0.0]
          reference: object_world
          max_linear_step: 0.005
          max_angular_step: 0.08

operators:
  - name: arm
    ik:
      joint_control_mode: solve_once_interpolate
      joint_interp_speed: 0.05
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
| `control.cartesian_max_linear_step` | `0.0` | 默认笛卡尔位置分段步长上限（m/tick）。大于 0 时，末端位置按直线小步逼近 |
| `control.cartesian_max_angular_step` | `0.0` | 默认笛卡尔姿态分段步长上限（rad/tick）。大于 0 时，末端姿态按 SLERP 小步逼近 |
| `pose.max_linear_step` | `0.0` | 单个 waypoint 的笛卡尔位置分段步长。若 > 0，则覆盖 operator 默认值 |
| `pose.max_angular_step` | `0.0` | 单个 waypoint 的笛卡尔姿态分段步长。若 > 0，则覆盖 operator 默认值 |
| `joint_control_mode` | `solve_once_interpolate` | Joint 模式执行策略。可选 `solve_once_interpolate` 或 `per_step_ik` |
| `joint_interp_speed` | `0.05` | 当 `joint_control_mode=solve_once_interpolate` 时，每个 control step 允许的最大单关节位移上限（rad/step），系统据此自适应计算插值步数 |
| `n_iterations` | 300 | 每次 solve 的 mink 迭代步数。越大求解越精确，但越慢 |
| `dt` | 0.1 | 每个 IK 迭代的虚拟时间步（秒）。`n_iterations × dt` = 总积分时长 |
| `position_cost` | 1.0 | EEF 位置跟踪权重 |
| `orientation_cost` | 1.0 | EEF 姿态跟踪权重。增大可提高姿态精度 |
| `posture_cost` | 1e-4 | 关节姿态正则化权重。增大使关节更保守（不易跳变），但降低末端精度 |
| `max_joint_delta` | 0.8 | 单次 solve 允许的最大关节位移（rad）。防止奇异点附近的解跳变 |

### 参数调优指南

**机械臂运动太慢：**
- 增大 `control.cartesian_max_linear_step`（如 0.015），减少位置分段数
- 增大 `control.cartesian_max_angular_step`（如 0.2），减少姿态分段数
- 或者直接给远距离 waypoint 设置更大的 `max_linear_step` / `max_angular_step`
- 对于 `solve_once_interpolate`，增大 `joint_interp_speed`（如 0.1），让插值更快到终点
- 增大 `max_joint_delta`（如 1.2），允许每步走更远
- 降低 `update_freq`（如 50），增加每步的物理仿真时间，使 PD 控制器有更多时间跟踪

**经过奇异点时关节跳变 / 末端翻转：**
- 优先使用 `solve_once_interpolate`，减少连续重求解带来的分支抖动
- 减小 `max_joint_delta`（如 0.5），限制关节速度
- 增大 `posture_cost`（如 1e-3），使 IK 更倾向于保持当前关节构型
- 调整 keyframe 中的初始关节角，使 home 配置远离奇异区域

**末端姿态不准确：**
- 增大 `orientation_cost`（如 2.0）
- 增大 `n_iterations`（如 500），给更多迭代时间
- 减小 `posture_cost`（如 1e-5），放松关节约束

**IK 求解太慢（影响实时性）：**
- 使用 `solve_once_interpolate`，降低 IK 调用频率
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
