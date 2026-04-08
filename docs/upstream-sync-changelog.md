# Upstream Sync Changelog (v0.2.0 → v0.2.5)

> 同步时间: 2026-04-08
> 上游: OpenGHz/auto-atomic-operation
> 涉及 39 个 commit，102 个文件，+4915 / -555 行

---

## 破坏性变更汇总

| # | 变更 | 影响 |
|---|---|---|
| 1 | operators 从 List 变 Dict | 所有手动构建 EnvConfig 的代码 |
| 2 | 相机名重命名 | 所有引用相机名的代码和配置 |
| 3 | 默认分辨率 640x480 → 1280x720 | 显存/内存占用、下游图像处理 |
| 4 | 默认 enable_mask 改为 false | 依赖 mask 的流水线 |
| 5 | timeout_steps 默认值 600 → 100 | 原来能完成的任务可能提前超时 |
| 6 | EEF 四元数坐标系变更 | 所有旧 orientation 值不可直接使用 |
| 7 | pose_site 统一为 `eef_pose` | 旧的 `gripper`/`tcp_site` 引用 |
| 8 | joint state 观测键名和数据结构变更 | structured 模式下游解析代码 |
| 9 | structured 模式 color 键名变更 | `color/image_raw` → `video_encoded` |
| 10 | image data 编码 `.tolist()` → `.tobytes()` | structured 模式下游解析 |
| 11 | pose 数据从列表改为字典格式 | structured 模式 |
| 12 | float32 → float64 | 可能影响下游精度假设 |
| 13 | max_joint_delta 从 IK 移到 operator 顶层 | 配置文件和构建代码 |
| 14 | mujoco >= 3.4.0 最低版本要求 | 旧环境需升级 |

---

## 一、YAML 配置变更

### 1.1 operators: List → Dict (破坏性)

所有包含 `operators` 的配置文件结构发生变化：

```yaml
# 旧
operators:
  - name: arm
    arm_actuators: [...]

# 新
operators:
  arm:
    arm_actuators: [...]
```

`name` 字段不再需要显式设置，由字典 key 自动填充。

**影响文件**: `basis_franka.yaml`, `basis_mocap_eef.yaml`, `basis_p7_xf9600.yaml`

### 1.2 相机名重命名 (破坏性)

| 旧名称 | 新名称 |
|---|---|
| `hand_cam` | `wrist_cam` |
| `front_cam` | `env1_cam` |
| `side_cam` | `env2_cam` |

**影响文件**: `basis.yaml` 及所有引用相机名的下游代码

### 1.3 相机新增 `is_static` 字段

```yaml
cameras:
  wrist_cam:
    is_static: false    # 动态，每帧重新渲染 GS 背景
  env1_cam:
    is_static: true     # 静态，GS 背景渲染一次并缓存
  env2_cam:
    is_static: true
```

### 1.4 公共变量提取到 `common_vars.yaml` (新文件)

以下字段从 `basis.yaml` 移到 `aao_configs/common_vars.yaml`：

| 字段 | 旧默认值 | 新默认值 |
|---|---|---|
| `cam_width` | 640 | **1280** |
| `cam_height` | 480 | **720** |
| `enable_mask` | true | **false** |
| `enable_color` | true | true (不变) |
| `enable_depth` | true | true (不变) |
| `enable_heat_map` | true | true (不变) |

`basis.yaml` 通过 Hydra defaults 引用：

```yaml
defaults:
  - common_vars
```

### 1.5 新增 `heatmap_operations` 字段

```yaml
# basis.yaml
heatmap_operations: ["pick", "place", "push", "pull", "press"]
```

将热力图通道定义与 `operations` 列表解耦，需要单独配置。

### 1.6 EEF 输出名称重命名

| 文件 | 旧值 | 新值 |
|---|---|---|
| `basis_mocap_eef.yaml` | `eef_output_name: eef` | `eef_output_name: gripper` |
| `basis_p7_xf9600.yaml` | `eef_output_name: eef` | `eef_output_name: gripper` |

### 1.7 pose_site 统一 (破坏性)

| 文件 | 旧值 | 新值 |
|---|---|---|
| `basis_franka.yaml` | `pose_site: gripper` | `pose_site: eef_pose` |
| `basis_p7_xf9600.yaml` | `pose_site: tcp_site` | `pose_site: eef_pose` |

### 1.8 max_joint_delta 位置变更 (破坏性)

从 IK 参数提升到 operator 顶层：

```yaml
# 旧
task_operators:
  - name: arm
    ik:
      max_joint_delta: 1.8

# 新
task_operators:
  - name: arm
    max_joint_delta: 1.8
    ik:
      # max_joint_delta 已移除
```

### 1.9 timeout_steps 默认值变更 (破坏性)

```yaml
# 旧默认值: 600
# 新默认值: 100
# 原来在配置中显式设置的 timeout_steps 行被移除
```

### 1.10 EEF 四元数方向值全面变更 (破坏性)

由于 pose_site 从 `gripper`/`tcp_site` 统一到 `eef_pose`，所有任务配置中的四元数需要重新计算：

| 文件 | 旧 orientation (xyzw) | 新 orientation (xyzw) |
|---|---|---|
| `arrange_flowers.yaml` | `[-0.707, 0.707, 0, 0]` | `[0, 0.707, 0, 0.707]` |
| `cup_on_coaster.yaml` | `[-0.707, 0.707, 0, 0]` | `[0, 0.707, 0, 0.707]` |
| `pick_and_place.yaml` | `[-0.707, 0.707, 0, 0]` | `[0, 0.707, 0, 0.707]` |
| `pick_and_place_franka.yaml` | `[-0.707, 0.707, 0, 0]` | `[-0.707, 0, 0.707, 0]` |
| `press_button_basis.yaml` | `[-0.707, 0.707, 0, 0]` | `[0, 0.707, 0, 0.707]` |
| `close_drawer.yaml` | `[-0.5, 0.5, 0.5, 0.5]` | `[0.5, 0.5, 0.5, 0.5]` |
| `hang_toothbrush_cup.yaml` | `[0.5, 0.5, -0.5, -0.5]` | `[0, 0, -1, 0]` |

同时 `rotation` (RPY euler) 被替换为 `orientation` (quaternion xyzw)。

### 1.11 其他配置变更

| 文件 | 变更 |
|---|---|
| `arrange_flowers.yaml` | 新增 `update_freq: 15` |
| `open_hinge_door.yaml` | 新增 `update_freq: 10` |
| `close_hinge_door.yaml` | 新增 `update_freq: 10` |
| `close_drawer.yaml` | `mask_objects`: `["handle", "drawer"]` → `["handle"]` |
| `press_green_button.yaml` | `object_name`: `green_blue` → `button_green` |
| `basis_mocap_eef.yaml` | 移除 `tactile_prefixes: ["left_", "right_"]` |

### 1.12 新增配置文件

| 文件 | 用途 |
|---|---|
| `common_vars.yaml` | 公共变量（分辨率、开关等） |
| `basis_xf9600.yaml` | XF9600 夹爪基础配置 |
| `basis_airbot_play_xf9600.yaml` | AIRBOT Play + XF9600 配置 |
| `open_door_airbot_play.yaml` | AIRBOT Play 开门任务 |
| `env/gl.yaml` | 图形库环境变量配置 (EGL/CUDA) |
| `test/test.yaml` | 测试用配置 |

---

## 二、Python API 变更

### 2.1 EnvConfig.operators 类型变更 (破坏性)

```python
# 旧
operators: List[OperatorBinding] = Field(default_factory=list)
# 新
operators: Dict[str, OperatorBinding] = Field(default_factory=dict)
```

遍历方式: `for op in self._operators:` → `for op in self._operators.values():`

新增 `populate_operator_names` model validator 自动从字典 key 填充 `name`。

### 2.2 MujocoControlConfig.timeout_steps 默认值 (破坏性)

```python
# 旧
timeout_steps: int = 600
# 新
timeout_steps: int = 100
```

### 2.3 Tolerance 支持 per-axis

```python
# 旧
position: float = 0.01
# 新
position: Union[float, List[float]] = 0.01
```

标量时使用 L2 norm 判定，三元素列表时使用 per-axis 判定。

### 2.4 max_joint_delta 移至 operator handler (破坏性)

- `MinkIKSolver` / `P7AnalyticalIKSolver` 中的 `max_joint_delta` 参数被**移除**
- Clamping 逻辑移到 `UnifiedMujocoEnv._clamp_joint_delta()` 静态方法
- `MujocoOperatorHandler` 新增 `max_joint_delta: float = 0.35`

### 2.5 观测键名变更 (structured 模式, 破坏性)

| 类别 | 旧键名 | 新键名 |
|---|---|---|
| 彩色图 | `.../color/image_raw` | `.../video_encoded` |
| 深度图 | `.../aligned_depth_to_color/image_raw` | `.../depth/image_raw` |
| Camera prefix 提取 | `cam_name.split("_")[0]` | `cam_name.split("_")[-2]` |

### 2.6 Joint state 观测变更 (structured 模式, 破坏性)

```python
# 旧: enc/ 和 action/ 两个键，数据相同
obs["enc/{limb}/joint_state"]    = {"data": {"position": ..., "velocity": ..., "effort": ctrl}}
obs["action/{limb}/joint_state"] = {"data": {"position": ..., "velocity": ..., "effort": ctrl}}

# 新: measurement 和 action 分离
obs["{limb}/joint_state"] = {
    "data": {
        "position": qpos.tolist(),
        "velocity": qvel.tolist(),
        "effort": actuator_force.tolist(),   # 数据源变更: ctrl → actuator_force
    }
}
obs["action/{limb}/joint_state"] = {
    "data": {
        "position": ctrl.tolist(),
        "velocity": [],    # 空
        "effort": [],      # 空
    }
}
```

### 2.7 Pose 数据格式变更 (structured 模式, 破坏性)

```python
# 旧
"position": [x, y, z]
"orientation": [x, y, z, w]

# 新
"position": {"x": float, "y": float, "z": float}
"orientation": {"x": float, "y": float, "z": float, "w": float}
```

### 2.8 Image data 编码变更 (structured 模式, 破坏性)

```python
# 旧
"data": data.tolist()
# 新
"data": np.ascontiguousarray(data).tobytes()
```

### 2.9 float32 → float64

大量 `np.asarray(x, dtype=np.float32)` 调用改为 `np.asarray(x)` (默认 float64)。涉及:
- `_OperatorState` 中的 target_pos/target_quat
- `world_to_base()` / `base_to_world()` 输入输出
- Wrench/force/torque 数据
- 所有 `BatchedUnifiedMujocoEnv` 批量操作

### 2.10 `_create_image_data` → `create_image_data`

私有函数变为公开导出。

### 2.11 KeyCreator 类 (新增)

统一管理观测键名前缀。structured 模式前缀为 `/robot/`，非 structured 模式为空。

### 2.12 IK 失败时抛出 RuntimeError

`update_operator_home_eef()` 中 IK 求解失败不再静默跳过，而是 raise RuntimeError。

---

## 三、新增功能

### 3.1 per-waypoint 容差覆盖

```python
class WaypointToleranceConfig(BaseModel, extra="forbid"):
    position: Optional[Union[float, List[float]]] = None
    orientation: Optional[float] = None
```

`PoseControlConfig` 新增 `tolerance` 字段。

### 3.2 per-waypoint 随机化

`PoseControlConfig` 新增 `randomization` 字段，支持在每个 waypoint 级别定义随机偏移。

### 3.3 ctrl 插值

`EnvConfig` 新增 `ctrl_interpolation: bool = False`。启用后在子步之间线性插值 ctrl 信号，防止 PD overshoot。

### 3.4 interests 字段

`EnvConfig` 新增 `interests: Tuple[List[str], List[str]] = ([], [])`，初始化时自动调用 `set_interest_objects_and_operations()`。

### 3.5 ExecutionSummary 扩展

新增字段: `env_completion_steps`, `env_completion_time_sec`, `completed_stage_info`。

### 3.6 Joint-mode operator 物理基座移动

`override_operator_base_pose()` 现在对 joint-mode operator 会实际移动 MuJoCo 中的物理 body。

### 3.7 AIRBOT Play KDL IK solver (新文件)

- `auto_atom/backend/mjc/ik/airbot_kdl_ik_solver.py`
- `auto_atom/backend/mjc/ik/third_party_ik/arm_kdl.py`
- `auto_atom/backend/mjc/ik/third_party_ik/p7_arm_analytical_ik.py`

---

## 四、依赖变更

| 项目 | 旧值 | 新值 |
|---|---|---|
| `version` | 0.2.0 | 0.2.5 |
| `mujoco` 最低版本 | 无要求 | >= 3.4.0 |
| `PyOpenGL_accelerate` | 无 | 新增 (mujoco optional) |
| ruff 排除 | 无 | `third_party_ik/*` |
