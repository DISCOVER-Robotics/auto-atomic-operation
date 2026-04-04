"""Observation adapter from AAO observations to model-facing payloads."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional

import numpy as np

MODEL_HEATMAP_ORDER = (
    "pick",
    "place",
    "push",
    "pull",
    "press",
)


def _squeeze_single_env(data: Any) -> Any:
    if isinstance(data, np.ndarray) and data.shape[:1] == (1,):
        return data[0]
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def _extract_data(observation: dict[str, dict], key: str) -> Optional[Any]:
    payload = observation.get(key)
    if payload is None:
        return None
    return _squeeze_single_env(payload.get("data"))


def _infer_timestamp_ns(observation: dict[str, dict]) -> float:
    for payload in observation.values():
        value = payload.get("t")
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            return float(value[0])
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, list) and value:
            return float(value[0])
    return 0.0


def _coerce_gripper_scalar(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((1,), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return array[:1].astype(np.float32)


@dataclass
class SimFrame:
    timestamp_ns: float
    cameras: dict[str, dict[str, Any]]
    robot_state: dict[str, Any]
    raw_observation: dict[str, dict]


class ObservationWindowAdapter:
    def __init__(
        self,
        sim_info: dict[str, Any],
        *,
        selected_cameras: list[str],
        history_frames: int,
        task_operations: Optional[list[str]] = None,
        model_camera: Optional[str] = None,
    ) -> None:
        self.sim_info = sim_info
        self.selected_cameras = list(selected_cameras)
        self.history_frames = int(history_frames)
        self.task_operations = list(task_operations or sim_info.get("operations", []))
        self.model_camera = (
            model_camera or (self.selected_cameras[0] if self.selected_cameras else "")
        )
        if self.model_camera and self.model_camera not in self.selected_cameras:
            raise ValueError(
                f"model_camera '{self.model_camera}' must be included in "
                f"selected_cameras={self.selected_cameras}."
            )
        self.frames: Deque[SimFrame] = deque(maxlen=max(self.history_frames, 1))

    def reset(self) -> None:
        self.frames.clear()

    def extend(self, observations: list[dict[str, dict]]) -> None:
        for observation in observations:
            self.frames.append(self.parse_observation(observation))

    def latest_frame(self) -> SimFrame:
        if not self.frames:
            raise RuntimeError("No observations available in the history window.")
        return self.frames[-1]

    def parse_observation(self, observation: dict[str, dict]) -> SimFrame:
        cameras: dict[str, dict[str, Any]] = {}
        for camera_name in self.selected_cameras:
            heatmap = _extract_data(observation, f"{camera_name}/mask/heat_map")
            heatmaps: dict[str, np.ndarray] = {}
            if heatmap is not None:
                heatmap = np.asarray(heatmap)
                if (
                    heatmap.ndim == 3
                    and self.task_operations
                    and heatmap.shape[-1] == len(self.task_operations)
                ):
                    for idx, op_name in enumerate(self.task_operations):
                        heatmaps[op_name] = np.asarray(
                            heatmap[..., idx],
                            dtype=np.float32,
                        )

            cameras[camera_name] = {
                "rgb": _extract_data(observation, f"{camera_name}/color/image_raw"),
                "depth_m": _extract_data(
                    observation,
                    f"{camera_name}/aligned_depth_to_color/image_raw",
                ),
                "mask": _extract_data(observation, f"{camera_name}/mask/image_raw"),
                "intrinsics": self._camera_intrinsics(camera_name),
                "extrinsics": self._camera_extrinsics(camera_name),
                "heatmaps": heatmaps,
            }

        robot_state = {
            "arm_joint_position": _extract_data(observation, "arm/joint_state/position"),
            "eef_joint_position": _extract_data(observation, "eef/joint_state/position"),
            "eef_position": _extract_data(observation, "arm/pose/position"),
            "eef_orientation_xyzw": _extract_data(observation, "arm/pose/orientation"),
            "eef_rotation_rpy": _extract_data(observation, "arm/pose/rotation"),
            "target_eef_position": _extract_data(observation, "action/arm/pose/position"),
            "target_eef_orientation_xyzw": _extract_data(
                observation,
                "action/arm/pose/orientation",
            ),
        }

        return SimFrame(
            timestamp_ns=_infer_timestamp_ns(observation),
            cameras=cameras,
            robot_state=robot_state,
            raw_observation=observation,
        )

    def build_model_input(self) -> dict[str, Any]:
        if not self.frames:
            raise RuntimeError("No observations available in the history window.")

        actual_history = list(self.frames)
        history = self._padded_history()
        primary_camera = self.model_camera or self.selected_cameras[0]
        if primary_camera not in self.selected_cameras:
            raise RuntimeError(
                f"Primary model camera '{primary_camera}' is not available."
            )

        camera_payloads: dict[str, dict[str, Any]] = {}
        for camera_name in self.selected_cameras:
            rgb_ref = self._camera_hw(camera_name) + (3,)
            depth_ref = self._camera_hw(camera_name)
            rgb_window = self._stack_optional_window(
                history,
                camera_name,
                "rgb",
                dtype=np.uint8,
                fallback_shape=rgb_ref,
            )
            depth_window = self._stack_optional_window(
                history,
                camera_name,
                "depth_m",
                dtype=np.float32,
                fallback_shape=depth_ref,
            )
            mask_window = self._stack_optional_window(
                history,
                camera_name,
                "mask",
                dtype=np.uint8,
                fallback_shape=depth_ref,
            )
            intrinsics_window = np.stack(
                [
                    self._coerce_intrinsics(frame.cameras[camera_name]["intrinsics"])
                    for frame in history
                ],
                axis=0,
            )
            extrinsics_window = np.stack(
                [
                    self._coerce_extrinsics(frame.cameras[camera_name]["extrinsics"])
                    for frame in history
                ],
                axis=0,
            )
            raw_heatmap_window = self._stack_task_heatmap_window(
                history,
                camera_name,
                depth_ref,
            )
            model_heatmap_window = self._stack_model_heatmap_window(
                history,
                camera_name,
                depth_ref,
            )
            heatmaps = history[-1].cameras[camera_name]["heatmaps"]

            camera_payloads[camera_name] = {
                "rgb_window": rgb_window,
                "depth_window_m": depth_window,
                "mask_window": mask_window,
                "intrinsics_window": intrinsics_window,
                "intrinsics": intrinsics_window[-1],
                "extrinsics_window": extrinsics_window,
                "extrinsics_window_inv": np.linalg.inv(extrinsics_window),
                "extrinsics": extrinsics_window[-1],
                "extrinsics_inv": np.linalg.inv(extrinsics_window[-1]),
                "heatmaps": heatmaps,
                "raw_heatmap_window": raw_heatmap_window,
                "raw_heatmap_keys": list(self.task_operations),
                "model_heatmap_window": model_heatmap_window,
                "model_heatmap_keys": list(MODEL_HEATMAP_ORDER),
                "heatmap_window": model_heatmap_window,
            }

        primary = camera_payloads[primary_camera]
        robot_state = history[-1].robot_state
        cartesian_position = self._current_cartesian_position(robot_state)
        gripper_position = _coerce_gripper_scalar(robot_state.get("eef_joint_position"))
        camera_intrinsics = np.repeat(
            np.asarray(primary["intrinsics"], dtype=np.float32)[None, :, :],
            3,
            axis=0,
        )

        return {
            "history_length": len(history),
            "available_history_length": len(actual_history),
            "selected_cameras": list(self.selected_cameras),
            "model_camera": primary_camera,
            "task_operations": list(self.task_operations),
            "timestamps_ns": np.asarray(
                [frame.timestamp_ns for frame in history],
                dtype=np.float64,
            ),
            "cameras": camera_payloads,
            "robot_state": robot_state,
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "worldmodel_ws_payload": {
                "endpoint": "infer",
                "observation/exterior_image_0_left_history": np.asarray(
                    primary["rgb_window"],
                    dtype=np.uint8,
                ),
                "observation/exterior_image_0_left": np.asarray(
                    primary["rgb_window"][-1],
                    dtype=np.uint8,
                ),
                "observation/exterior_depth_0": np.asarray(
                    primary["depth_window_m"][-1],
                    dtype=np.float32,
                ),
                "observation/camera_intrinsics": camera_intrinsics.astype(np.float32),
                "observation/heatmaps": np.asarray(
                    primary["model_heatmap_window"][-1],
                    dtype=np.float32,
                ),
                "observation/heatmap_keys": list(MODEL_HEATMAP_ORDER),
                "observation/cartesian_position": cartesian_position,
                "observation/gripper_position": gripper_position,
                "observation/robot_state/cartesian_position": cartesian_position,
                "observation/robot_state/gripper_position": gripper_position,
            },
        }

    def _camera_hw(self, camera_name: str) -> tuple[int, int]:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        color_info = info.get("camera_info", {}).get("color", {})
        depth_info = info.get("camera_info", {}).get("depth", {})
        height = int(color_info.get("height", depth_info.get("height", 480)))
        width = int(color_info.get("width", depth_info.get("width", 640)))
        return (height, width)

    def _camera_intrinsics(self, camera_name: str) -> Optional[np.ndarray]:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        color_k = info.get("camera_info", {}).get("color", {}).get("k")
        depth_k = info.get("camera_info", {}).get("depth", {}).get("k")
        k = color_k if color_k is not None else depth_k
        if k is None:
            return None
        return np.asarray(k, dtype=np.float32).reshape(3, 3)

    def _camera_extrinsics(self, camera_name: str) -> Optional[np.ndarray]:
        info = self.sim_info.get("cameras", {}).get(camera_name, {})
        extr = info.get("camera_extrinsics")
        if not extr:
            return None
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.asarray(
            extr.get("rotation_matrix", np.eye(3)),
            dtype=np.float32,
        ).reshape(3, 3)
        matrix[:3, 3] = np.asarray(
            extr.get("translation", np.zeros(3)),
            dtype=np.float32,
        ).reshape(3)
        return matrix

    @staticmethod
    def _coerce_intrinsics(intrinsics: Optional[np.ndarray]) -> np.ndarray:
        if intrinsics is None:
            return np.zeros((3, 3), dtype=np.float32)
        return np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)

    @staticmethod
    def _coerce_extrinsics(extrinsics: Optional[np.ndarray]) -> np.ndarray:
        if extrinsics is None:
            return np.eye(4, dtype=np.float32)
        return np.asarray(extrinsics, dtype=np.float32).reshape(4, 4)

    def _padded_history(self) -> list[SimFrame]:
        history = list(self.frames)
        if not history:
            raise RuntimeError("No observations available in the history window.")
        target_length = max(self.history_frames, 1)
        if len(history) >= target_length:
            return history[-target_length:]
        pad_count = target_length - len(history)
        return [history[0]] * pad_count + history

    def _stack_optional_window(
        self,
        history: list[SimFrame],
        camera_name: str,
        field: str,
        *,
        dtype: np.dtype,
        fallback_shape: tuple[int, ...],
    ) -> np.ndarray:
        values: list[np.ndarray] = []
        zero = np.zeros(fallback_shape, dtype=dtype)
        for frame in history:
            value = frame.cameras[camera_name].get(field)
            if value is None:
                values.append(zero.copy())
            else:
                values.append(np.asarray(value, dtype=dtype))
        return np.stack(values, axis=0)

    def _stack_task_heatmap_window(
        self,
        history: list[SimFrame],
        camera_name: str,
        ref_hw: tuple[int, int],
    ) -> np.ndarray:
        return np.stack(
            [
                self._stack_task_heatmaps(frame.cameras[camera_name]["heatmaps"], ref_hw)
                for frame in history
            ],
            axis=0,
        )

    def _stack_model_heatmap_window(
        self,
        history: list[SimFrame],
        camera_name: str,
        ref_hw: tuple[int, int],
    ) -> np.ndarray:
        return np.stack(
            [
                self._stack_model_heatmaps(frame.cameras[camera_name]["heatmaps"], ref_hw)
                for frame in history
            ],
            axis=0,
        )

    def _stack_task_heatmaps(
        self,
        heatmaps: dict[str, np.ndarray],
        ref_hw: tuple[int, int],
    ) -> np.ndarray:
        if not self.task_operations:
            height, width = ref_hw
            return np.zeros((0, height, width), dtype=np.float32)

        ordered: list[np.ndarray] = []
        for op_name in self.task_operations:
            if op_name in heatmaps:
                ordered.append(np.asarray(heatmaps[op_name], dtype=np.float32))
            else:
                ordered.append(np.zeros(ref_hw, dtype=np.float32))
        return np.stack(ordered, axis=0)

    @staticmethod
    def _stack_model_heatmaps(
        heatmaps: dict[str, np.ndarray],
        ref_hw: tuple[int, int],
    ) -> np.ndarray:
        ordered: list[np.ndarray] = []
        for op_name in MODEL_HEATMAP_ORDER:
            if op_name in heatmaps:
                ordered.append(np.asarray(heatmaps[op_name], dtype=np.float32))
            else:
                ordered.append(np.zeros(ref_hw, dtype=np.float32))
        return np.stack(ordered, axis=0)

    @staticmethod
    def _current_cartesian_position(robot_state: dict[str, Any]) -> np.ndarray:
        position = robot_state.get("eef_position")
        rotation = robot_state.get("eef_rotation_rpy")
        position_array = (
            np.asarray(position, dtype=np.float32).reshape(-1)[:3]
            if position is not None
            else np.zeros((3,), dtype=np.float32)
        )
        rotation_array = (
            np.asarray(rotation, dtype=np.float32).reshape(-1)[:3]
            if rotation is not None
            else np.zeros((3,), dtype=np.float32)
        )
        if position_array.shape[0] < 3:
            position_array = np.pad(position_array, (0, 3 - position_array.shape[0]))
        if rotation_array.shape[0] < 3:
            rotation_array = np.pad(rotation_array, (0, 3 - rotation_array.shape[0]))
        return np.concatenate([position_array[:3], rotation_array[:3]], axis=0).astype(
            np.float32
        )
