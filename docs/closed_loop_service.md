# Closed-Loop WS Service

这份说明只讲这个 fork 里新增的闭环服务层，以及 `WorldModel_3d` 客户端是怎么调用它的。

底层执行任务、判断 stage 成败、抓取 observation、应用低层动作，仍然是 `auto-atomic-operation` 原有的 `PolicyEvaluator`、`MujocoTaskBackend` 和 MuJoCo env。

## 这份 fork 里新增/修改了什么

本次闭环相关改动主要有三部分：

1. 新增 WebSocket 服务端：
   - `auto_atom/runner/policy_server.py`
   - 作用是把 `PolicyEvaluator` 封成可被外部客户端调用的服务

2. 补了一个 heatmap / interest-object 的小 patch：
   - `auto_atom/policy_eval.py`
   - `auto_atom/runtime.py`
   - 作用是当前 stage 还没真正激活时，也先把下一阶段的目标 object / operation 注册到 backend，避免启动阶段拿不到对应 mask / heatmap

3. 客户端不在这个仓库：
   - 客户端、observation adapter、闭环评测脚本在 `WorldModel_3d/closed_loop_eval/`

## 服务端职责

`policy_server.py` 做的事情不是重写仿真器，而是把现有 AAO runtime 封成一个 WebSocket 会话服务。

每个连接对应一个 `SimulatorSession`，它负责：

- `init`
  - 加载 Hydra task config
  - 创建 `PolicyEvaluator`
  - 记录 `env.get_info()`
  - 初始化录像器和 oracle
- `reset`
  - reset evaluator
  - 抓首帧 observation
  - 返回 `observation/update/info`
- `step`
  - 接收一个 action chunk
  - 只推进 `num_steps`
  - 每一步都抓 observation 和 update
  - 需要时写视频和数组
- `expert_action`
  - 基于当前 stage 返回 oracle chunk
- `close`
  - 收尾并 finalize 当前 episode

## 协议

连接建立后，服务端会先发一帧 metadata：

```python
{
  "service": "AAOClosedLoopService",
  "version": 1,
  "supported_endpoints": [
    "init",
    "reset",
    "step",
    "expert_action",
    "close",
  ],
  "supported_action_formats": [
    "cartesian_absolute",
    "joint_absolute",
  ],
}
```

后续所有请求和响应都是：

- WebSocket binary frame
- `msgpack_numpy` 编码

## 服务端请求格式

### 1. `init`

```python
{
  "endpoint": "init",
  "config_name": "pick_and_place",
  "overrides": ["task.seed=42"],
  "action_format": "cartesian_absolute",
  "operator_name": "arm",
  "recording": {
    "enabled": True,
    "output_root": "/abs/path/to/output",
    "video_camera": "front_cam",
    "save_mp4": True,
    "save_arrays": True,
    "fps": 30,
    "episode_name": "pick_and_place_ep000_seed42",
  },
}
```

说明：

- 如果 `overrides` 里没有 `env.batch_size=1`，服务端会自动补上
- 如果 `overrides` 里没有 `++env.viewer.disable=true`，服务端也会自动补上

返回：

```python
{
  "status": "ok",
  "config_name": "...",
  "overrides": [...],
  "action_format": "cartesian_absolute",
  "info": {...},
}
```

其中 `info` 来自 `backend.env.get_info()`。

### 2. `reset`

```python
{
  "endpoint": "reset",
  "episode_name": "pick_and_place_ep000_seed42",
}
```

返回：

```python
{
  "status": "ok",
  "observation": {...},
  "update": {...},
  "info": {...},
  "episode_dir": "/abs/path/to/episode_dir",
}
```

注意：

- 这里返回的是仿真器原始 observation
- 不是模型直接吃的 payload
- 模型输入格式转换在 `WorldModel_3d` 客户端侧完成

### 3. `expert_action`

```python
{
  "endpoint": "expert_action",
  "horizon": 8,
  "action_format": "cartesian_absolute",
}
```

返回：

```python
{
  "status": "ok",
  "payload": {
    "action_format": "cartesian_absolute",
    "action_horizons": [1, 2, 3, 4, 5, 6, 7, 8],
    "actions": float32[8, 7],
  },
}
```

### 4. `step`

```python
{
  "endpoint": "step",
  "action": {
    "action_format": "cartesian_absolute",
    "action_horizons": [1, 2, 3, 4, 5, 6, 7, 8],
    "actions": [
      [x1, y1, z1, roll1, pitch1, yaw1, gripper1],
      [x2, y2, z2, roll2, pitch2, yaw2, gripper2],
      ...
      [x8, y8, z8, roll8, pitch8, yaw8, gripper8],
    ],
  },
  "num_steps": 5,
}
```

返回：

```python
{
  "status": "ok",
  "observations": [...],   # 长度通常为 num_steps
  "step_updates": [...],
  "update": {...},         # 最后一步的 TaskUpdate
  "applied_steps": 5,
  "done": [False],
  "success": [None],
  "summary": None,         # 任务结束时才有
}
```

## 动作格式

### `cartesian_absolute`

单步动作为：

```python
[x, y, z, roll, pitch, yaw, gripper]
```

含义：

- `x, y, z`: 末端绝对位置，单位米
- `roll, pitch, yaw`: 绝对欧拉角，单位弧度
- `gripper`: 夹爪绝对值

服务端内部会把 `roll/pitch/yaw` 转成 quaternion，再调用：

```python
env.apply_pose_action(operator_name, position, quat, gripper)
```

### `joint_absolute`

服务端会直接调用：

```python
env.apply_joint_action(operator_name, tick, env_mask=...)
```

## 录像和数据落盘

只要 `recording.enabled=True`，服务端就会在 episode 目录里写：

- `summary.json`
- `trace.json.gz`
- `low_dim_trace.npz`
- `<video_camera>.mp4`
- `camera_arrays.npz`，前提是 `save_arrays=True`

其中 `camera_arrays.npz` 会按相机拆出：

- `<camera>.rgb`
- `<camera>.depth`
- `<camera>.mask`
- `<camera>.heat_map`

## `WorldModel_3d` 客户端是怎么调这个服务的

客户端代码在另一个仓库：

- `WorldModel_3d/closed_loop_eval/sim_service_client.py`
- `WorldModel_3d/closed_loop_eval/run_closed_loop_eval.py`
- `WorldModel_3d/closed_loop_eval/observation_adapter.py`

调用顺序是：

1. 连接 WebSocket 服务
2. 发 `init`
3. 发 `reset`
4. 把 `reset` 返回的原始 observation 转成模型输入
5. 让模型或 mock 模型输出 `8` 帧 action chunk
6. 发 `step(num_steps=5)`
7. 把 `step` 返回的 5 帧 observation 再转成新的模型输入
8. 循环直到 `done=true`

当前默认验证方式不是接真实模型，而是：

- 客户端先校验三相机 payload 结构
- 再调用服务端 `expert_action`
- 用 oracle action 验证整条闭环链路是否打通

## 模型输入转换不在服务端做

服务端返回的 observation 是 AAO 原始观测键，例如：

- `front_cam/color/image_raw`
- `front_cam/aligned_depth_to_color/image_raw`
- `front_cam/mask/image_raw`
- `front_cam/mask/heat_map`
- `hand_cam/...`
- `side_cam/...`

客户端 adapter 再把它们整理成模型侧的三相机 sample：

- `selected_cameras`
- `timestamps_ns`
- `robot_state`
- `cameras.front_cam.rgb_window`
- `cameras.front_cam.depth_window_m`
- `cameras.front_cam.mask_window`
- `cameras.front_cam.raw_heatmap_window`
- `cameras.front_cam.legacy_heatmap_window`
- 三路相机同理

另外，刚 reset 时只有首帧，客户端会把首帧复制成 `history_length=5` 的启动窗口。

## 最小调用示例

先启动服务端：

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
./.venv/bin/python -m auto_atom.runner.policy_server --host 127.0.0.1 --port 8765 --log-level INFO
```

然后客户端调用：

```python
from closed_loop_eval.sim_service_client import SimulatorServiceClient

with SimulatorServiceClient("ws://127.0.0.1:8765") as c:
    c.init(
        config_name="pick_and_place",
        overrides=["task.seed=42"],
        action_format="cartesian_absolute",
        operator_name="arm",
        recording={
            "enabled": False,
            "output_root": "/tmp/unused",
            "video_camera": "front_cam",
            "save_mp4": False,
            "save_arrays": False,
            "fps": 30,
            "episode_name": "probe_pick_and_place",
        },
    )
    reset_response = c.reset(episode_name="probe_pick_and_place")
    action_payload = c.expert_action(horizon=8, action_format="cartesian_absolute")
    step_response = c.step(action_payload=action_payload, num_steps=5)
```

## 排查时最值得先看哪里

如果是服务端 reset 前就失败，先看：

1. `policy_server.py`
   - `SimulatorSession.initialize()`
   - `SimulatorSession.reset()`
2. `PolicyEvaluator.reset()` / `get_observation()`
3. `backend.env.capture_observation()`

如果是动作应用后失败，先看：

1. `policy_server.py`
   - `_action_applier()`
2. 任务 stage 的 success / failure 条件
3. `trace.json.gz` 里的 `failure_category` 和 `failure_reason`
