# PolicyEvaluator 仿真器服务端接入说明

> 文档日期：2026-04-05
>
> 本文档只记录 `auto-atomic-operation` 里需要保留的内容：仿真器服务端本身，以及它对外暴露的远程调用接口。WorldModel 闭环中间层不再放在本仓库。

---

## 1. 仓库边界

当前职责边界如下：

- `auto-atomic-operation`
  - 只负责启动 MuJoCo / GS 仿真
  - 只负责通过 `rpyc` 暴露 `PolicyEvaluator` 服务端
- `WorldModel_3d`
  - 负责闭环中间层
  - 负责 observation 到模型 payload 的转换
  - 负责模型 websocket client
  - 负责录像、trace 落盘、批量评测和成功率统计

中间层的正式位置是：

- `WorldModel_3d/closed_loop_eval/`

---

## 2. 本仓库保留的关键文件

- `examples/policy_eval_server.py`
  - 服务端启动入口
  - 支持 `--gpu`
  - 会在导入 `auto_atom` 前设置 headless EGL 环境变量
  - 会把当前 Python 的 `bin` 目录 prepend 到 `PATH`，保证 `gsplat` JIT 时能找到 `ninja`
- `auto_atom/ipc/service.py`
  - `rpyc` 服务实现
- `auto_atom/ipc/client.py`
  - 远程客户端封装，供 `WorldModel_3d/closed_loop_eval/sim_service_client.py` 调用

---

## 3. 服务端对外 API

`examples/policy_eval_server.py` 启动后，对外提供的是 `RemotePolicyEvaluator` 兼容接口。

客户端初始化：

- `ping()`
- `from_config(config_name, overrides)`
- `from_yaml(path)`

每个 episode 的主要调用顺序：

1. `from_config(...)`
2. `get_info()`
3. `reset()`
4. `get_observation()`
5. 多次 `update(action)`
6. `summarize(max_updates=..., updates_used=..., elapsed_time_sec=...)`

只读数据：

- `records`
- `stage_plans`
- `batch_size`

其中：

- `get_info()`
  - 返回场景级信息，包括相机名、分辨率、内参、外参等
  - 中间层应优先使用这里的相机参数，不要自己猜
- `get_observation()`
  - 返回当前一步的完整 observation dict
  - RGB、depth、mask、heatmap、robot state 都从这里取
- `update(action)`
  - 是一步动作推进，不负责 horizon / stride
  - receding horizon 由 `WorldModel_3d` 中间层负责

---

## 4. `update()` 期望的动作格式

服务端当前由中间层逐步调用，每次喂一步动作：

```python
{
    "position": np.ndarray([x, y, z], dtype=np.float32),
    "orientation": np.ndarray([qx, qy, qz, qw], dtype=np.float32),
    "gripper": np.ndarray([gripper], dtype=np.float32),
}
```

说明：

- 中间层拿到模型输出的单步 `cartesian_absolute`
  - `[x, y, z, roll, pitch, yaw, gripper]`
- 然后把 `roll/pitch/yaw` 转成 quaternion
- 再逐步调用 `update()`

也就是说，仿真器服务端不需要理解模型的 8-step chunk；它只处理一步动作。

---

## 5. 推荐启动方式

推荐命令：

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
source ~/.config/cuda-env.sh
./.venv/bin/python examples/policy_eval_server.py \
  --host 127.0.0.1 \
  --port 18861 \
  --gpu 1
```

如果你使用 `uv` 管理环境，也可以先同步环境再启动：

```bash
cd /DATA/disk1/zoyo/auto-atomic-operation
uv sync
source ~/.config/cuda-env.sh
uv run python examples/policy_eval_server.py --host 127.0.0.1 --port 18861 --gpu 1
```

---

## 6. 这次验证后保留的环境变量结论

用户最初提供了这组变量：

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

本机实测后保留结论：

- 必需：
  - `MUJOCO_GL=egl`
- 推荐保留：
  - `PYOPENGL_PLATFORM=egl`
  - `EGL_PLATFORM=device`
- 多 GPU pinning 推荐保留：
  - `CUDA_VISIBLE_DEVICES`
  - `EGL_VISIBLE_DEVICES`
  - `MUJOCO_EGL_DEVICE_ID`
- 本轮实测中不是必需：
  - `__EGL_VENDOR_LIBRARY_FILENAMES`
  - `LD_PRELOAD`
  - `NVIDIA_DRIVER_CAPABILITIES`

当前这些逻辑已经固化在：

- `examples/policy_eval_server.py`

---

## 7. 录像和 trace 不在本仓库做

当前服务端只负责：

- 场景初始化
- 一步步推进
- 返回 observation / summary / records

录像和 trace 由 `WorldModel_3d/closed_loop_eval` 负责：

- 每步抓取三路相机 RGB
- 横向拼接成 `multicam.mp4`
- 保存 `client_trace.json.gz`
- 保存 `camera_arrays.npz`
- 保存 `summary.json`
- 汇总 `aggregate_summary.json`

---

## 8. 当前结论

这次保留在 `auto-atomic-operation` 的改动只有两类：

- 仿真器服务端能力增强
  - `get_info()` 暴露
  - `policy_eval_server.py --gpu`
  - headless EGL 启动修复
- 服务端所需依赖
  - `rpyc`
  - GS 场景需要的 `ninja` 等依赖

WorldModel 闭环中间层不应继续留在本仓库。
