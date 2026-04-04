"""WebSocket service for policy-evaluation-based closed-loop simulation."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import imageio.v3 as iio
import numpy as np
import websockets.sync.server

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
python_bin_dir = str(Path(sys.executable).parent)
os.environ["PATH"] = (
    python_bin_dir
    if not os.environ.get("PATH")
    else f"{python_bin_dir}:{os.environ['PATH']}"
)

from auto_atom import PolicyEvaluator, TaskUpdate, load_task_file_hydra
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.runtime import PrimitiveAction, TaskFlowBuilder, TaskRunner
from auto_atom.utils.pose import (
    euler_to_quaternion,
    quaternion_angular_distance,
    quaternion_to_rpy,
)

logger = logging.getLogger(__name__)

try:
    import msgpack
    import msgpack_numpy as msgpack_numpy_lib

    msgpack_numpy_lib.patch()

    class _MsgpackCompat:
        @staticmethod
        def packb(obj: Any) -> bytes:
            return msgpack.packb(obj, default=msgpack_numpy_lib.encode)

        @staticmethod
        def unpackb(data: bytes) -> Any:
            return msgpack.unpackb(
                data,
                object_hook=msgpack_numpy_lib.decode,
                raw=False,
            )

        class Packer:
            def pack(self, obj: Any) -> bytes:
                return msgpack.packb(obj, default=msgpack_numpy_lib.encode)

    msgpack_numpy = _MsgpackCompat()
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "policy_server.py requires msgpack and msgpack-numpy."
    ) from exc


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return getattr(value, "value")
    return value


def _is_low_dim_value(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim <= 2
    if isinstance(value, np.generic):
        return True
    if isinstance(value, (bool, int, float, str)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        return arr.ndim <= 1
    if isinstance(value, dict):
        return all(_is_low_dim_value(v) for v in value.values())
    return False


def _extract_low_dim_observation(obs: dict[str, dict]) -> dict[str, dict]:
    low_dim: dict[str, dict] = {}
    excluded_suffixes = (
        "/color/image_raw",
        "/aligned_depth_to_color/image_raw",
        "/mask/image_raw",
        "/mask/heat_map",
        "/tactile/point_cloud2",
    )
    for key, payload in obs.items():
        if key.endswith(excluded_suffixes):
            continue
        data = payload.get("data")
        if not _is_low_dim_value(data):
            continue
        low_dim[key] = {
            "data": _to_jsonable(data),
            "t": _to_jsonable(payload.get("t")),
        }
    return low_dim


def _iter_low_dim_leaf_items(
    low_dim_step: dict[str, dict],
) -> list[tuple[str, object, object]]:
    items: list[tuple[str, object, object]] = []
    for key, payload in low_dim_step.items():
        data = payload.get("data")
        t = payload.get("t")
        if isinstance(data, dict):
            for field_name, field_value in data.items():
                items.append((f"{key}/{field_name}", field_value, t))
        else:
            items.append((key, data, t))
    return items


def _build_low_dim_npz_payload(
    low_dim_observations: list[dict[str, dict]],
) -> dict[str, np.ndarray]:
    leaf_keys = sorted(
        {
            leaf_key
            for step in low_dim_observations
            for leaf_key, _, _ in _iter_low_dim_leaf_items(step)
        }
    )
    payload: dict[str, np.ndarray] = {"low_dim_keys": np.asarray(leaf_keys, dtype=str)}
    for idx, leaf_key in enumerate(leaf_keys):
        values: list[np.ndarray] = []
        times: list[float] = []
        for step in low_dim_observations:
            leaf_items = {k: (v, t) for k, v, t in _iter_low_dim_leaf_items(step)}
            if leaf_key not in leaf_items:
                raise ValueError(f"Missing low-dimensional key '{leaf_key}' in trace.")
            value, t = leaf_items[leaf_key]
            values.append(np.asarray(value, dtype=np.float32).reshape(-1))
            t_scalar = t[0] if isinstance(t, (list, np.ndarray)) else t
            times.append(float(t_scalar))
        payload[f"low_dim_data__{idx}"] = np.stack(values).astype(np.float32)
        payload[f"low_dim_t__{idx}"] = np.asarray(times, dtype=np.float64)
    return payload


def _serialize_task_update(update: TaskUpdate) -> dict[str, Any]:
    return {
        "stage_index": np.asarray(update.stage_index, dtype=np.int64),
        "stage_name": list(update.stage_name),
        "status": np.asarray([str(s.value) for s in update.status], dtype=object),
        "done": np.asarray(update.done, dtype=bool),
        "success": np.asarray(update.success, dtype=object),
        "details": [_to_jsonable(d) for d in update.details],
        "phase": list(update.phase),
        "phase_step": np.asarray(update.phase_step, dtype=np.int64),
    }


def _serialize_summary(summary: Any) -> dict[str, Any]:
    return {
        "total_stages": int(summary.total_stages),
        "max_updates": summary.max_updates,
        "updates_used": int(summary.updates_used),
        "elapsed_time_sec": float(summary.elapsed_time_sec),
        "completed_stage_count": np.asarray(summary.completed_stage_count, dtype=np.int64),
        "final_stage_index": np.asarray(summary.final_stage_index, dtype=np.int64),
        "final_stage_name": list(summary.final_stage_name),
        "final_status": np.asarray(
            [str(s.value) for s in summary.final_status],
            dtype=object,
        ),
        "final_done": np.asarray(summary.final_done, dtype=bool),
        "final_success": np.asarray(summary.final_success, dtype=object),
        "records": [_to_jsonable(asdict(record)) for record in summary.records],
    }


def _normalize_single_env(data: Any) -> Any:
    if isinstance(data, np.ndarray) and data.shape[:1] == (1,):
        return data[0]
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def _get_current_gripper_command(env: Any, operator_name: str) -> np.ndarray:
    single_env = env.envs[0]
    eef_aidx = single_env._op_eef_aidx[operator_name]
    if len(eef_aidx) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.asarray(single_env.data.ctrl[eef_aidx], dtype=np.float32).copy()


def _first_timestamp(obs: dict[str, dict]) -> Any:
    for payload in obs.values():
        if "t" in payload:
            return payload["t"]
    return np.zeros(1, dtype=np.float64)


@dataclass
class EpisodeRecorderConfig:
    enabled: bool = False
    output_root: str = ""
    video_camera: str = "front_cam"
    save_mp4: bool = True
    save_arrays: bool = False
    fps: int = 30
    episode_name: str = ""


class EpisodeRecorder:
    def __init__(self, config: EpisodeRecorderConfig) -> None:
        self.config = config
        self.episode_dir: Optional[Path] = None
        self.frames: list[np.ndarray] = []
        self.low_dim_observations: list[dict[str, dict]] = []
        self.raw_steps: list[dict[str, Any]] = []
        self.tick_actions: list[np.ndarray] = []
        self.camera_buffers: dict[str, dict[str, list[np.ndarray]]] = {}

    def reset(self, task_name: str) -> Optional[Path]:
        if not self.config.enabled:
            self.episode_dir = None
            self.frames = []
            self.low_dim_observations = []
            self.raw_steps = []
            self.tick_actions = []
            self.camera_buffers = {}
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        episode_name = self.config.episode_name or f"{task_name}_{timestamp}"
        self.episode_dir = Path(self.config.output_root).expanduser().resolve() / episode_name
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.frames = []
        self.low_dim_observations = []
        self.raw_steps = []
        self.tick_actions = []
        self.camera_buffers = {}
        return self.episode_dir

    def record_observation(
        self,
        observation: dict[str, dict],
        update: TaskUpdate,
        action: Optional[np.ndarray],
        step_index: int,
    ) -> None:
        if not self.config.enabled:
            return
        self.low_dim_observations.append(_extract_low_dim_observation(observation))
        self.raw_steps.append(
            {
                "step_index": step_index,
                "action": _to_jsonable(action) if action is not None else None,
                "update": _serialize_task_update(update),
            }
        )
        if action is not None:
            self.tick_actions.append(np.asarray(action, dtype=np.float32))

        frame_key = f"{self.config.video_camera}/color/image_raw"
        frame_payload = observation.get(frame_key)
        if frame_payload is not None:
            frame = np.asarray(_normalize_single_env(frame_payload["data"]), dtype=np.uint8)
            self.frames.append(frame)

        if self.config.save_arrays:
            for key, payload in observation.items():
                data = payload.get("data")
                if key.endswith("/color/image_raw"):
                    cam = key.split("/")[0]
                    self.camera_buffers.setdefault(cam, {}).setdefault("rgb", []).append(
                        np.asarray(_normalize_single_env(data), dtype=np.uint8)
                    )
                elif key.endswith("/aligned_depth_to_color/image_raw"):
                    cam = key.split("/")[0]
                    self.camera_buffers.setdefault(cam, {}).setdefault("depth", []).append(
                        np.asarray(_normalize_single_env(data), dtype=np.float32)
                    )
                elif key.endswith("/mask/heat_map"):
                    cam = key.split("/")[0]
                    self.camera_buffers.setdefault(cam, {}).setdefault("heat_map", []).append(
                        np.asarray(_normalize_single_env(data), dtype=np.uint8)
                    )
                elif key.endswith("/mask/image_raw"):
                    cam = key.split("/")[0]
                    self.camera_buffers.setdefault(cam, {}).setdefault("mask", []).append(
                        np.asarray(_normalize_single_env(data), dtype=np.uint8)
                    )

    def finalize(self, task_name: str, summary: Optional[dict[str, Any]]) -> Optional[Path]:
        if not self.config.enabled or self.episode_dir is None:
            return None

        summary_path = self.episode_dir / "summary.json"
        trace_path = self.episode_dir / "trace.json.gz"
        low_dim_npz_path = self.episode_dir / "low_dim_trace.npz"
        video_path = self.episode_dir / f"{self.config.video_camera}.mp4"

        with summary_path.open("w", encoding="utf-8") as fp:
            json.dump(
                _to_jsonable(
                    {
                        "task_name": task_name,
                        "summary": summary,
                        "episode_dir": str(self.episode_dir),
                        "num_frames": len(self.frames),
                        "num_tick_actions": len(self.tick_actions),
                    }
                ),
                fp,
                indent=2,
            )

        with gzip.open(trace_path, "wt", encoding="utf-8") as fp:
            json.dump(
                _to_jsonable(
                    {
                        "task_name": task_name,
                        "low_dim_observations": self.low_dim_observations,
                        "tick_actions": self.tick_actions,
                        "steps": self.raw_steps,
                    }
                ),
                fp,
            )

        if self.low_dim_observations:
            npz_payload = _build_low_dim_npz_payload(self.low_dim_observations)
            if self.tick_actions:
                npz_payload["tick_actions"] = np.stack(self.tick_actions).astype(np.float32)
            np.savez_compressed(low_dim_npz_path, **npz_payload)

        if self.frames and self.config.save_mp4:
            iio.imwrite(
                video_path,
                self.frames,
                fps=self.config.fps,
                codec="libx264",
                quality=8,
            )

        if self.config.save_arrays and self.camera_buffers:
            arrays_path = self.episode_dir / "camera_arrays.npz"
            flat_payload: dict[str, np.ndarray] = {}
            for camera_name, streams in self.camera_buffers.items():
                for stream_name, values in streams.items():
                    if values:
                        flat_payload[f"{camera_name}.{stream_name}"] = np.stack(values)
            if flat_payload:
                np.savez_compressed(arrays_path, **flat_payload)

        return self.episode_dir


class DemoOraclePlanner:
    """Repeat the current config-derived primitive action as a model chunk."""

    def __init__(self, evaluator: PolicyEvaluator, operator_name: str) -> None:
        self.evaluator = evaluator
        self.operator_name = operator_name
        self.builder = TaskFlowBuilder()
        self.current_stage_index: Optional[int] = None
        self.current_plan = None
        self.actions: list[PrimitiveAction] = []
        self.action_index: int = 0

    def reset(self) -> None:
        self.current_stage_index = None
        self.current_plan = None
        self.actions = []
        self.action_index = 0

    def sync_with_update(self, update: TaskUpdate) -> None:
        stage_index = int(update.stage_index[0]) if len(update.stage_index) > 0 else -1
        done = bool(update.done[0]) if len(update.done) > 0 else True
        if done or stage_index < 0:
            self.reset()
            return

        if self.current_stage_index != stage_index:
            self.current_stage_index = stage_index
            self.current_plan = self.evaluator.stage_plans[stage_index]
            self.actions, _ = self.builder.build_actions(
                self.current_plan.stage,
                self.current_plan.last_orientation_before,
            )
            self.action_index = 0

        self._advance_completed_actions()

    def _advance_completed_actions(self) -> None:
        if self.current_plan is None:
            return
        while self.action_index < len(self.actions) - 1:
            action = self.actions[self.action_index]
            if not self._is_action_completed(action):
                return
            self.action_index += 1

    def _is_action_completed(self, action: PrimitiveAction) -> bool:
        context = self.evaluator._require_context()
        backend = context.backend
        operator = backend.get_operator_handler(self.operator_name)
        target = backend.get_object_handler(self.current_plan.stage.object)
        if action.kind == "pose" and action.pose is not None:
            resolved = TaskRunner._resolve_pose_command(
                env_index=0,
                operator=operator,
                pose=action.pose,
                target=target,
                backend=backend,
                action=action,
            )
            current_eef = operator.get_end_effector_pose().select(0)
            pos_error = float(
                np.linalg.norm(current_eef.position[0] - np.asarray(resolved.position))
            )
            ori_error = float(
                quaternion_angular_distance(
                    current_eef.orientation[0],
                    np.asarray(resolved.orientation, dtype=np.float32),
                )
            )
            tolerance = operator.control.tolerance
            return (
                pos_error <= float(tolerance.position)
                and ori_error <= float(tolerance.orientation)
            )
        if action.kind == "eef" and action.eef is not None:
            if action.eef.close:
                if self.current_plan.stage.object:
                    return bool(
                        backend.is_object_grasped(
                            self.operator_name,
                            self.current_plan.stage.object,
                        )[0]
                    )
                return bool(backend.is_operator_grasping(self.operator_name)[0])
            return not bool(backend.is_operator_grasping(self.operator_name)[0])
        return False

    def _current_action(self) -> Optional[PrimitiveAction]:
        if not self.actions:
            return None
        return self.actions[min(self.action_index, len(self.actions) - 1)]

    def build_chunk(self, horizon: int, action_format: str) -> dict[str, Any]:
        action = self._current_action()
        if action is None or self.current_plan is None:
            dims = 7 if action_format == "cartesian_absolute" else 1
            actions = np.zeros((horizon, dims), dtype=np.float32)
            return {
                "action_format": action_format,
                "action_horizons": np.arange(1, horizon + 1, dtype=np.int32),
                "actions": actions,
            }

        tick = self._build_single_tick(action, action_format)
        chunk = np.repeat(tick.reshape(1, -1), horizon, axis=0).astype(np.float32)
        return {
            "action_format": action_format,
            "action_horizons": np.arange(1, horizon + 1, dtype=np.int32),
            "actions": chunk,
        }

    def _build_single_tick(
        self,
        action: PrimitiveAction,
        action_format: str,
    ) -> np.ndarray:
        context = self.evaluator._require_context()
        backend = context.backend
        env = backend.env
        operator = backend.get_operator_handler(self.operator_name)
        target = backend.get_object_handler(self.current_plan.stage.object)

        if action_format == "joint_absolute":
            single_env = env.envs[0]
            return np.asarray(single_env.data.ctrl[: single_env.model.nu], dtype=np.float32)

        if action.kind == "pose" and action.pose is not None:
            resolved = TaskRunner._resolve_pose_command(
                env_index=0,
                operator=operator,
                pose=action.pose,
                target=target,
                backend=backend,
                action=action,
            )
            gripper = _get_current_gripper_command(env, self.operator_name)
            rpy = quaternion_to_rpy(np.asarray(resolved.orientation, dtype=np.float32))
            return np.concatenate(
                [
                    np.asarray(resolved.position, dtype=np.float32),
                    np.asarray(rpy, dtype=np.float32),
                    np.asarray(gripper[:1], dtype=np.float32),
                ]
            )

        if action.kind == "eef" and action.eef is not None:
            current_pose = operator.get_end_effector_pose().select(0)
            rpy = quaternion_to_rpy(current_pose.orientation[0])
            gripper_target = np.asarray(
                [float(operator._eef_target(action.eef))],
                dtype=np.float32,
            )
            return np.concatenate(
                [
                    np.asarray(current_pose.position[0], dtype=np.float32),
                    np.asarray(rpy, dtype=np.float32),
                    gripper_target,
                ]
            )

        raise ValueError(f"Unsupported oracle primitive action: {action.kind}")


class SimulatorSession:
    def __init__(self, config_dir: Path, default_action_format: str = "cartesian_absolute") -> None:
        self.config_dir = config_dir
        self.default_action_format = default_action_format
        self.evaluator: Optional[PolicyEvaluator] = None
        self.task_file = None
        self.task_name: str = ""
        self.operator_name: str = "arm"
        self.updates_used: int = 0
        self.episode_start_time: float = 0.0
        self.info_cache: Optional[dict[str, Any]] = None
        self.oracle: Optional[DemoOraclePlanner] = None
        self.recorder_config = EpisodeRecorderConfig()
        self.recorder = EpisodeRecorder(self.recorder_config)
        self.episode_finished: bool = True

    def initialize(
        self,
        *,
        config_name: str,
        overrides: list[str],
        operator_name: str,
        recording: dict[str, Any],
        action_format: str,
    ) -> dict[str, Any]:
        self.close()

        self.default_action_format = action_format or self.default_action_format
        self.operator_name = operator_name or "arm"
        self.recorder_config = EpisodeRecorderConfig(**recording)
        self.recorder = EpisodeRecorder(self.recorder_config)

        resolved_overrides = list(overrides)
        if not any(item.startswith("env.batch_size=") for item in resolved_overrides):
            resolved_overrides.append("env.batch_size=1")
        if not any("env.viewer.disable" in item for item in resolved_overrides):
            resolved_overrides.append("++env.viewer.disable=true")

        self.task_file = load_task_file_hydra(
            config_name=config_name,
            config_dir=self.config_dir,
            overrides=resolved_overrides,
        )
        self.evaluator = PolicyEvaluator(
            action_applier=self._action_applier,
            observation_getter=self._observation_getter,
        ).from_config(self.task_file)
        self.task_name = config_name
        backend = self._backend()
        self.info_cache = backend.env.get_info()
        self.oracle = DemoOraclePlanner(self.evaluator, self.operator_name)
        return {
            "status": "ok",
            "config_name": config_name,
            "overrides": resolved_overrides,
            "action_format": self.default_action_format,
            "info": self.info_cache,
        }

    def reset(self, episode_name: str = "") -> dict[str, Any]:
        evaluator = self._require_evaluator()
        self._finalize_partial_episode()
        self.updates_used = 0
        self.episode_start_time = time.perf_counter()
        if episode_name:
            self.recorder_config.episode_name = episode_name
        episode_dir = self.recorder.reset(self.task_name)
        self.episode_finished = False
        update = evaluator.reset()
        observation = evaluator.get_observation()
        assert self.oracle is not None
        self.oracle.reset()
        self.oracle.sync_with_update(update)
        self.recorder.record_observation(
            observation=observation,
            update=update,
            action=None,
            step_index=0,
        )
        return {
            "status": "ok",
            "observation": observation,
            "update": _serialize_task_update(update),
            "info": self.info_cache,
            "episode_dir": str(episode_dir) if episode_dir is not None else "",
        }

    def step(self, action_payload: dict[str, Any], num_steps: int) -> dict[str, Any]:
        evaluator = self._require_evaluator()
        actions = np.asarray(action_payload["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        fmt = str(action_payload.get("action_format", self.default_action_format))

        max_steps = min(int(num_steps), int(actions.shape[0]))
        per_step_observations: list[dict[str, dict]] = []
        per_step_updates: list[dict[str, Any]] = []
        final_update: Optional[TaskUpdate] = None
        summary = None

        for idx in range(max_steps):
            tick_action = {
                "action_format": fmt,
                "action": actions[idx],
            }
            final_update = evaluator.update(tick_action)
            self.updates_used += 1
            obs = evaluator.get_observation()
            per_step_observations.append(obs)
            per_step_updates.append(_serialize_task_update(final_update))
            self.recorder.record_observation(
                observation=obs,
                update=final_update,
                action=actions[idx],
                step_index=self.updates_used,
            )
            assert self.oracle is not None
            self.oracle.sync_with_update(final_update)
            if bool(np.all(final_update.done)):
                elapsed = time.perf_counter() - self.episode_start_time
                summary = evaluator.summarize(
                    final_update,
                    updates_used=self.updates_used,
                    elapsed_time_sec=elapsed,
                )
                self.recorder.finalize(self.task_name, _serialize_summary(summary))
                self.episode_finished = True
                break

        if final_update is None:
            raise RuntimeError("step() received an empty action payload.")

        return {
            "status": "ok",
            "observations": per_step_observations,
            "step_updates": per_step_updates,
            "update": _serialize_task_update(final_update),
            "applied_steps": max_steps,
            "done": np.asarray(final_update.done, dtype=bool),
            "success": np.asarray(final_update.success, dtype=object),
            "summary": _serialize_summary(summary) if summary is not None else None,
        }

    def expert_action(self, horizon: int, action_format: str) -> dict[str, Any]:
        self._require_evaluator()
        assert self.oracle is not None
        return {
            "status": "ok",
            "payload": self.oracle.build_chunk(horizon, action_format),
        }

    def close(self) -> None:
        self._finalize_partial_episode()
        if self.evaluator is not None:
            try:
                self.evaluator.close()
            finally:
                self.evaluator = None
                self.task_file = None
                self.task_name = ""
                self.info_cache = None
                self.oracle = None
                self.episode_finished = True

    def _require_evaluator(self) -> PolicyEvaluator:
        if self.evaluator is None:
            raise RuntimeError("Session is not initialized. Call endpoint='init' first.")
        return self.evaluator

    def _backend(self) -> MujocoTaskBackend:
        evaluator = self._require_evaluator()
        context = evaluator._require_context()
        backend = context.backend
        if not isinstance(backend, MujocoTaskBackend):
            raise TypeError(
                "policy_server currently only supports MujocoTaskBackend sessions."
            )
        return backend

    def _finalize_partial_episode(self) -> None:
        if self.episode_finished or self.evaluator is None:
            return
        if self.recorder.episode_dir is None:
            self.episode_finished = True
            return
        elapsed = max(time.perf_counter() - self.episode_start_time, 0.0)
        summary = self.evaluator.summarize(
            updates_used=self.updates_used,
            elapsed_time_sec=elapsed,
        )
        self.recorder.finalize(self.task_name, _serialize_summary(summary))
        self.episode_finished = True

    def _observation_getter(self, context: Any) -> dict[str, dict]:
        backend = context.backend
        return backend.env.capture_observation()

    def _action_applier(self, context: Any, action: Any, env_mask: Optional[np.ndarray] = None) -> None:
        if action is None:
            return None

        backend = context.backend
        env = backend.env
        fmt = str(action.get("action_format", self.default_action_format))
        tick = np.asarray(action["action"], dtype=np.float32).reshape(-1)

        if fmt == "cartesian_absolute":
            if tick.shape[0] < 6:
                raise ValueError(
                    "cartesian_absolute expects at least 6 dims: x,y,z,roll,pitch,yaw[,gripper]."
                )
            position = tick[:3]
            rpy = tick[3:6]
            quat = np.asarray(
                euler_to_quaternion(tuple(float(v) for v in rpy)),
                dtype=np.float32,
            )
            gripper = tick[6:7] if tick.shape[0] >= 7 else None
            env.apply_pose_action(
                self.operator_name,
                position,
                quat,
                gripper,
                env_mask=env_mask,
            )
            return None

        if fmt == "joint_absolute":
            env.apply_joint_action(
                self.operator_name,
                tick,
                env_mask=env_mask,
            )
            return None

        raise ValueError(f"Unsupported action_format '{fmt}'.")


class PolicyEvalService:
    def __init__(self, host: str, port: int, config_dir: Path) -> None:
        self.host = host
        self.port = port
        self.config_dir = config_dir

    def serve(self) -> None:
        packer = msgpack_numpy.Packer() if hasattr(msgpack_numpy, "Packer") else None

        def pack(obj: Any) -> bytes:
            if packer:
                return packer.pack(obj)
            return msgpack_numpy.packb(obj)

        def handler(ws: Any) -> None:
            session = SimulatorSession(config_dir=self.config_dir)
            ws.send(
                pack(
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
                )
            )

            try:
                while True:
                    try:
                        data = ws.recv()
                    except Exception:
                        logger.info("Client disconnected.")
                        break

                    if isinstance(data, str):
                        continue
                    request = msgpack_numpy.unpackb(data)
                    endpoint = str(request.get("endpoint", "")).strip()
                    try:
                        if endpoint == "init":
                            response = session.initialize(
                                config_name=str(request["config_name"]),
                                overrides=list(request.get("overrides", [])),
                                operator_name=str(request.get("operator_name", "arm")),
                                recording=dict(request.get("recording", {})),
                                action_format=str(
                                    request.get("action_format", "cartesian_absolute")
                                ),
                            )
                        elif endpoint == "reset":
                            response = session.reset(
                                episode_name=str(request.get("episode_name", "")),
                            )
                        elif endpoint == "step":
                            response = session.step(
                                action_payload=dict(request["action"]),
                                num_steps=int(request.get("num_steps", 1)),
                            )
                        elif endpoint == "expert_action":
                            response = session.expert_action(
                                horizon=int(request.get("horizon", 8)),
                                action_format=str(
                                    request.get("action_format", "cartesian_absolute")
                                ),
                            )
                        elif endpoint == "close":
                            session.close()
                            response = {"status": "ok"}
                        else:
                            raise ValueError(
                                f"Unsupported endpoint '{endpoint}'."
                            )
                    except Exception as exc:  # pragma: no cover
                        logger.exception("Service request failed: %s", exc)
                        response = {
                            "status": "error",
                            "error": str(exc),
                            "endpoint": endpoint,
                        }
                    ws.send(pack(response))
            finally:
                session.close()

        logger.info("Starting AAO closed-loop service on ws://%s:%d", self.host, self.port)
        with websockets.sync.server.serve(
            handler,
            host=self.host,
            port=self.port,
            compression=None,
            max_size=None,
        ) as server:
            server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="AAO closed-loop WebSocket service")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "aao_configs"),
        help="Path to aao_configs directory.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service = PolicyEvalService(
        host=args.host,
        port=args.port,
        config_dir=Path(args.config_dir).resolve(),
    )
    service.serve()


if __name__ == "__main__":
    main()
