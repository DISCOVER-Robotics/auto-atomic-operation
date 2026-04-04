"""Episode recording utilities for closed-loop evaluation."""

from __future__ import annotations

import gzip
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .observation_adapter import MODEL_HEATMAP_ORDER, SimFrame


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class EpisodeRecorder:
    def __init__(
        self,
        *,
        episode_dir: Path,
        episode_name: str,
        task_name: str,
        camera_names: list[str],
        fps: int,
        save_arrays: bool,
        save_video: bool = True,
    ) -> None:
        self.episode_dir = episode_dir
        self.episode_name = episode_name
        self.task_name = task_name
        self.camera_names = list(camera_names)
        self.fps = int(fps)
        self.save_arrays = bool(save_arrays)
        self.save_video = bool(save_video)

        self.video_frames: list[np.ndarray] = []
        self.trace_steps: list[dict[str, Any]] = []
        self.camera_buffers: dict[str, dict[str, list[np.ndarray]]] = {
            camera: {
                "rgb": [],
                "depth": [],
                "mask": [],
                "heat_map": [],
            }
            for camera in self.camera_names
        }

    def record(
        self,
        *,
        step_index: int,
        chunk_index: int,
        chunk_step_index: int,
        sim_frame: SimFrame,
        update: Any,
        action_cartesian: Optional[np.ndarray],
        remote_action: Optional[dict[str, Any]],
    ) -> None:
        self.video_frames.append(self._compose_multicam_frame(sim_frame))

        for camera_name in self.camera_names:
            camera = sim_frame.cameras.get(camera_name, {})
            rgb = self._coerce_rgb(camera.get("rgb"))
            depth = self._coerce_depth(camera.get("depth_m"), rgb.shape[:2])
            mask = self._coerce_mask(camera.get("mask"), rgb.shape[:2])
            heat_map = self._ordered_heatmaps(camera.get("heatmaps", {}), rgb.shape[:2])

            self.camera_buffers[camera_name]["rgb"].append(rgb)
            self.camera_buffers[camera_name]["depth"].append(depth)
            self.camera_buffers[camera_name]["mask"].append(mask)
            self.camera_buffers[camera_name]["heat_map"].append(heat_map)

        self.trace_steps.append(
            {
                "step_index": int(step_index),
                "chunk_index": int(chunk_index),
                "chunk_step_index": int(chunk_step_index),
                "timestamp_ns": float(sim_frame.timestamp_ns),
                "action_cartesian": None
                if action_cartesian is None
                else np.asarray(action_cartesian, dtype=np.float32),
                "remote_action": remote_action,
                "update": update,
                "robot_state": sim_frame.robot_state,
            }
        )

    def finalize(
        self,
        *,
        summary: Any,
        records: list[Any],
        metadata: dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        summary_payload = {
            "task_name": self.task_name,
            "episode_name": self.episode_name,
            "episode_dir": str(self.episode_dir),
            "num_video_frames": len(self.video_frames),
            "num_recorded_steps": len(self.trace_steps),
            "camera_names": list(self.camera_names),
            "heatmap_order": list(MODEL_HEATMAP_ORDER),
            "summary": summary,
            "records": records,
            "error": error,
        }
        with (self.episode_dir / "summary.json").open("w", encoding="utf-8") as fp:
            json.dump(to_jsonable(summary_payload), fp, indent=2)

        trace_payload = {
            "metadata": metadata,
            "steps": self.trace_steps,
            "summary": summary,
            "records": records,
            "error": error,
        }
        with gzip.open(
            self.episode_dir / "client_trace.json.gz",
            "wt",
            encoding="utf-8",
        ) as fp:
            json.dump(to_jsonable(trace_payload), fp)

        if self.save_arrays:
            arrays_payload: dict[str, np.ndarray] = {}
            for camera_name, streams in self.camera_buffers.items():
                for stream_name, values in streams.items():
                    if values:
                        arrays_payload[f"{camera_name}.{stream_name}"] = np.stack(
                            values,
                            axis=0,
                        )
            if arrays_payload:
                np.savez_compressed(self.episode_dir / "camera_arrays.npz", **arrays_payload)

        if self.save_video and self.video_frames:
            self._write_video(self.episode_dir / "multicam.mp4")

    def _compose_multicam_frame(self, sim_frame: SimFrame) -> np.ndarray:
        tiles: list[np.ndarray] = []
        tile_hw: Optional[tuple[int, int]] = None

        for camera_name in self.camera_names:
            camera = sim_frame.cameras.get(camera_name, {})
            rgb = self._coerce_rgb(camera.get("rgb"))
            if tile_hw is None:
                tile_hw = rgb.shape[:2]
            else:
                rgb = self._resize_rgb(rgb, tile_hw)
            tiles.append(self._annotate_rgb(rgb, camera_name))

        if not tiles:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return np.concatenate(tiles, axis=1)

    def _write_video(self, video_path: Path) -> None:
        with imageio.get_writer(
            str(video_path),
            fps=float(self.fps),
            macro_block_size=None,
        ) as writer:
            for frame in self.video_frames:
                writer.append_data(frame)

    @staticmethod
    def _coerce_rgb(value: Any) -> np.ndarray:
        if value is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = np.asarray(value, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        return rgb

    @staticmethod
    def _coerce_depth(value: Any, hw: tuple[int, int]) -> np.ndarray:
        if value is None:
            return np.zeros(hw, dtype=np.float32)
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _coerce_mask(value: Any, hw: tuple[int, int]) -> np.ndarray:
        if value is None:
            return np.zeros(hw, dtype=np.uint8)
        return np.asarray(value, dtype=np.uint8)

    @staticmethod
    def _ordered_heatmaps(
        heatmaps: dict[str, np.ndarray],
        hw: tuple[int, int],
    ) -> np.ndarray:
        ordered: list[np.ndarray] = []
        for op_name in MODEL_HEATMAP_ORDER:
            if op_name in heatmaps:
                ordered.append(np.asarray(heatmaps[op_name], dtype=np.float32))
            else:
                ordered.append(np.zeros(hw, dtype=np.float32))
        return np.stack(ordered, axis=0)

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
        height, width = target_hw
        if rgb.shape[:2] == (height, width):
            return rgb
        return np.asarray(
            Image.fromarray(rgb).resize((int(width), int(height)), Image.BILINEAR),
            dtype=np.uint8,
        )

    @staticmethod
    def _annotate_rgb(rgb: np.ndarray, camera_name: str) -> np.ndarray:
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 8, 8 + 11 * len(camera_name), 34), fill=(0, 0, 0))
        draw.text((12, 12), camera_name, fill=(255, 255, 255))
        return np.asarray(image, dtype=np.uint8)
