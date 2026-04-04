# WorldModel 闭环评测接入说明

> 文档日期：2026-04-04
>
> 本文档记录这次在 `auto-atomic-operation` 中完成的 WorldModel 闭环中间层改动，以及仿真器服务端、闭环 runner、模型服务端三者的实际使用方式。

---

## 1. 这次改了什么

本次闭环逻辑最终放在 `auto-atomic-operation`，不再放在模型仓库里。

新增或修改的关键文件：

- `auto_atom/worldmodel_eval/__init__.py`
  - 闭环中间层包入口
- `auto_atom/worldmodel_eval/observation_adapter.py`
  - 把仿真器 observation 转成模型输入
  - 负责首帧不足历史长度时的复制补齐
  - 负责 heatmap 顺序对齐
- `auto_atom/worldmodel_eval/sim_service_client.py`
  - `rpyc` 仿真器客户端
  - 把模型输出的 `[x, y, z, roll, pitch, yaw, gripper]` 转成仿真器 `update()` 需要的 `position + quaternion + gripper`
- `auto_atom/worldmodel_eval/model_clients.py`
  - `PayloadValidatingHoldModelClient`
    - 校验模型输入 payload 是否满足约定
    - 可在没有真实模型时返回假的 8 步 absolute cartesian action
  - `WebSocketModelClient`
    - 用 binary websocket + msgpack 和真实模型服务交互
- `auto_atom/worldmodel_eval/episode_recorder.py`
  - 保存 `client_trace.json.gz`
  - 保存 `summary.json`
  - 保存 `camera_arrays.npz`
  - 把三路相机横向拼接成 `multicam.mp4`
- `auto_atom/runner/worldmodel_closed_loop_eval.py`
  - 真正的闭环 runner
  - 实现 `模型出 8 步 / 仿真只走前 5 步 / 再拿新 observation 回喂模型`
  - 支持批量任务、批量 episode、成功率统计
  - 支持显式 `--max-updates`
- `examples/worldmodel_closed_loop_eval.py`
  - runner example 入口
- `examples/policy_eval_server.py`
  - 现在支持 `--gpu`
  - 会自动设置 headless EGL 环境变量
  - 会把当前 Python 的 `bin` 目录 prepend 到 `PATH`，保证 `gsplat` JIT 时能找到 `ninja`
- `pyproject.toml`
  - 增加 `imageio / msgpack / msgpack-numpy / pillow / rpyc / websockets`
  - 在 `gs` extras 里增加 `ninja`
- `tests/test_worldmodel_closed_loop.py`
  - 增加 bootstrap history、payload 校验、success 统计逻辑测试

---

## 2. 当前闭环结构

当前结构是三段式：

1. 仿真器服务端
   - `auto-atomic-operation/examples/policy_eval_server.py`
   - 基于 `rpyc`
   - 暴露 `from_config / get_info / reset / get_observation / update / summarize`

2. 闭环中间层
   - `auto_atom/runner/worldmodel_closed_loop_eval.py`
   - 负责 observation 聚合、格式转换、action chunk 拆步执行、录视频、存 trace、算成功率

3. 模型服务端
   - 当前通过 websocket binary + msgpack 交互
   - 中间层可以接 mock，也可以接真实模型

职责边界：

- 仿真器服务端只负责一步 `update(action)`
- 中间层负责 receding horizon
- 模型只负责“给定 observation，输出未来 8 步 action chunk”

---

## 3. 闭环 runner 的实际逻辑

默认逻辑：

- `horizon = 8`
- `stride = 5`
- `history_frames = 5`
- `max_updates = 500`

单个 episode 的运行过程：

1. 中间层连接 `rpyc` 仿真器服务
2. `from_config(task)` 初始化场景
3. `get_info()` 拿相机和内参信息
4. `reset()`
5. `get_observation()` 拿首帧
6. 如果历史帧不足 5 帧，就把首帧复制到 5 帧
7. 构造模型 payload
8. 请求模型，拿到 8 步 absolute cartesian action
9. 只执行前 5 步
10. 每一步都：
    - `update(action_i)`
    - `get_observation()`
    - 记录三路相机、depth、mask、heatmap、robot_state、action、update
11. 若任务完成则停止；否则继续下一轮 chunk
12. 如果总步数达到 `max_updates`，直接判失败

成功判定规则：

- 只有 `summary.final_done` 全为 `True`
- 且 `summary.final_success` 全为明确的 `True`
- 才记为 success

到 `max_updates` 截止但未完成，不算成功。

---

## 4. 模型输入输出协议

### 4.1 中间层发给模型的 payload

发送方式：

- binary websocket frame
- `msgpack` + `msgpack_numpy`

当前中间层发出的核心字段：

```python
{
    "endpoint": "infer",
    "observation/exterior_image_0_left_history": uint8[5, H, W, 3],
    "observation/exterior_image_0_left": uint8[H, W, 3],
    "observation/exterior_depth_0": float32[H, W],
    "observation/camera_intrinsics": float32[3, 3, 3],
    "observation/heatmaps": float32[5, H, W],
    "observation/heatmap_keys": [
        "pick", "place", "push", "pull", "press"
    ],
    "observation/cartesian_position": float32[6],
    "observation/gripper_position": float32[1],
    "observation/robot_state/cartesian_position": float32[6],
    "observation/robot_state/gripper_position": float32[1],
}
```

说明：

- `history` 不够时会用首帧补齐到 5 帧
- `camera_intrinsics` 目前是把主相机的 `3x3` 内参复制 3 份，形状固定 `(3,3,3)`
- `heatmap` 顺序固定为：
  - `pick, place, push, pull, press`
- 虽然三路相机都会完整记录下来，但当前发给模型的是选定主相机的一路 RGBD + heatmap

### 4.2 模型返回给中间层的 payload

当前中间层要求模型返回：

```python
{
    "action_format": "cartesian_absolute",
    "action_horizons": int32[8],   # [1,2,3,4,5,6,7,8]
    "actions": float32[8, 7],
}
```

其中每一行为：

```python
[x, y, z, roll, pitch, yaw, gripper]
```

语义：

- absolute cartesian pose
- 位置单位米
- 姿态单位弧度
- `gripper` 为 absolute gripper value

### 4.3 中间层如何把模型 action 送进仿真器

仿真器 `update()` 不直接吃 `(7,)` action。

中间层会把：

```python
[x, y, z, roll, pitch, yaw, gripper]
```

转换成：

```python
{
    "position": np.ndarray([x, y, z]),
    "orientation": quaternion_xyzw_from_rpy,
    "gripper": np.ndarray([gripper]),
}
```

然后逐步调用 `evaluator.update(remote_action)`。

---

## 5. 仿真器服务端怎么起

推荐命令：

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
source ~/.config/cuda-env.sh
./.venv/bin/python examples/policy_eval_server.py \
  --host 127.0.0.1 \
  --port 18861 \
  --gpu 1
```

说明：

- `--gpu` 会自动设置 headless EGL 所需环境变量
- 这条命令已经在本机做过真实验证
- 如果不指定 `--gpu`，则只设置 EGL 模式，不做 GPU pinning

### 5.1 这次验证后保留的环境变量结论

用户之前给的变量如下：

```bash
export CUDA_DEVICE=3
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export EGL_VISIBLE_DEVICES=$CUDA_DEVICE
export MUJOCO_EGL_DEVICE_ID=$CUDA_DEVICE
export EGL_PLATFORM=device
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
export NVIDIA_DRIVER_CAPABILITIES=all
```

本机实测结论：

- 必需：
  - `MUJOCO_GL=egl`
- 建议保留：
  - `PYOPENGL_PLATFORM=egl`
  - `EGL_PLATFORM=device`
- 建议在多 GPU 机器上保留：
  - `CUDA_VISIBLE_DEVICES`
  - `EGL_VISIBLE_DEVICES`
  - `MUJOCO_EGL_DEVICE_ID`
- 本机这轮验证里不是必需：
  - `__EGL_VENDOR_LIBRARY_FILENAMES`
  - `LD_PRELOAD`
  - `NVIDIA_DRIVER_CAPABILITIES`

所以当前服务端入口默认保留的是：

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
EGL_PLATFORM=device
CUDA_VISIBLE_DEVICES=<gpu>
EGL_VISIBLE_DEVICES=<gpu>
MUJOCO_EGL_DEVICE_ID=<gpu>
```

### 5.2 为什么还要 prepend Python bin 到 PATH

GS 场景在第一次走 `gsplat` 的 CUDA JIT 时，需要找到 `ninja` 可执行文件。

仅仅把 `ninja` 装进 `.venv` 不够：

- 如果服务端是直接 `./.venv/bin/python script.py`
- 但进程 `PATH` 里没有 `.venv/bin`
- Torch JIT 仍然会报找不到 `ninja`

所以 `examples/policy_eval_server.py` 现在会自动把当前 Python 所在目录 prepend 到 `PATH`。

---

## 6. 闭环 runner 怎么用

### 6.1 先用 mock 验证链路

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
./.venv/bin/python -m auto_atom.runner.worldmodel_closed_loop_eval \
  --sim-uri rpyc://127.0.0.1:18861 \
  --model-mode mock_validate \
  --tasks cup_on_coaster_gs,arrange_flowers_gs,wipe_the_table_gs \
  --episodes-per-task 1 \
  --max-updates 500 \
  --stride 5 \
  --history-frames 5 \
  --save-arrays
```

### 6.2 接真实模型 websocket 服务

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
./.venv/bin/python -m auto_atom.runner.worldmodel_closed_loop_eval \
  --sim-uri rpyc://127.0.0.1:18861 \
  --model-mode ws \
  --model-uri ws://127.0.0.1:6000 \
  --tasks cup_on_coaster_gs \
  --max-updates 500 \
  --save-arrays
```

如果模型服务协议没对齐，这里会直接报 payload / action format 错误。

---

## 7. 输出目录长什么样

runner 输出目录结构：

```text
outputs/worldmodel_closed_loop_eval/
  aggregate_summary.json
  episodes/
    <task>_ep000_seed42/
      summary.json
      client_trace.json.gz
      camera_arrays.npz
      multicam.mp4
```

各文件用途：

- `aggregate_summary.json`
  - 全部 episode 的成功率汇总
- `summary.json`
  - 当前 episode 的最终状态
- `client_trace.json.gz`
  - 每一步的 model input 摘要、action、update、robot_state
- `camera_arrays.npz`
  - 三路相机的 `rgb / depth / mask / heat_map`
- `multicam.mp4`
  - 三路相机横向拼接视频

---

## 8. 已做过的真实验证

本机在 2026-04-04 已做过如下真实 smoke：

- 服务端：
  - `examples/policy_eval_server.py --gpu 1`
- 客户端：
  - `python -m auto_atom.runner.worldmodel_closed_loop_eval`
  - `--model-mode mock_validate`
  - `--tasks cup_on_coaster_gs`
  - `--max-updates 5`

结果：

- `rpyc` 服务连通
- GS 场景可以初始化
- `get_info / reset / get_observation / update / summarize` 全链路通
- `multicam.mp4 / camera_arrays.npz / client_trace.json.gz / summary.json` 均落盘
- 因 `max_updates=5` 提前截止，所以 episode 被正确记为失败

保留产物：

- `/tmp/aao_wm_closed_loop_smoke_ok4`

---

## 9. 当前已知边界

1. 中间层已经按“新版模型协议”准备好了 websocket payload，但当前 `WorldModel_3d` 仓库里 checked-in 的旧模型服务不一定已经对齐这个协议。
2. 当前模型输入仍然只发送单主相机到模型；三路相机完整数据会记录，但不会全部送模型。
3. 非 GS 任务这轮没有继续作为重点验证对象；当前优先保证 GS 任务链路通。

