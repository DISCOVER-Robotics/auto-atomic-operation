"""Thin client for the AAO rpyc PolicyEvaluator service."""

from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np

from auto_atom.ipc import RemotePolicyEvaluator
from auto_atom.utils.pose import euler_to_quaternion


class SimulatorServiceClient:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.host, self.port = self._parse_uri(uri)
        self.evaluator: Optional[RemotePolicyEvaluator] = None
        self.metadata: Optional[dict[str, Any]] = None
        self.info: Optional[dict[str, Any]] = None
        self.task_name: str = ""

    def connect(self) -> dict[str, Any]:
        self.evaluator = RemotePolicyEvaluator(self.host, self.port)
        self.metadata = dict(self.evaluator.ping())
        return self.metadata

    def init(
        self,
        *,
        config_name: str,
        overrides: list[str],
        action_format: str = "cartesian_absolute",
    ) -> dict[str, Any]:
        if action_format != "cartesian_absolute":
            raise ValueError(
                "The remote PolicyEvaluator service currently only supports "
                "cartesian_absolute actions via apply_pose_action()."
            )
        evaluator = self._require_evaluator()
        resolved_overrides = self._normalize_overrides(overrides)
        evaluator.from_config(config_name, overrides=resolved_overrides)
        self.info = dict(evaluator.get_info())
        self.task_name = config_name
        return {
            "status": "ok",
            "config_name": config_name,
            "overrides": resolved_overrides,
            "action_format": action_format,
            "info": self.info,
        }

    def get_info(self) -> dict[str, Any]:
        evaluator = self._require_evaluator()
        self.info = dict(evaluator.get_info())
        return self.info

    def reset(self) -> Any:
        return self._require_evaluator().reset()

    def get_observation(self) -> dict[str, dict]:
        observation = self._require_evaluator().get_observation()
        if not isinstance(observation, dict):
            raise RuntimeError(
                f"Expected observation dict, got {type(observation).__name__}."
            )
        return observation

    def update_cartesian_action(
        self,
        action_row: np.ndarray,
    ) -> tuple[Any, dict[str, Any]]:
        action_row = np.asarray(action_row, dtype=np.float32).reshape(-1)
        if action_row.shape[0] < 7:
            raise ValueError(
                "cartesian_absolute action must contain 7 values: "
                "[x, y, z, roll, pitch, yaw, gripper]."
            )
        remote_action = {
            "position": np.asarray(action_row[:3], dtype=np.float32),
            "orientation": np.asarray(
                euler_to_quaternion(tuple(float(v) for v in action_row[3:6])),
                dtype=np.float32,
            ),
            "gripper": np.asarray(action_row[6:7], dtype=np.float32),
        }
        update = self._require_evaluator().update(remote_action)
        return update, remote_action

    def summarize(
        self,
        *,
        max_updates: Optional[int],
        updates_used: int,
        elapsed_time_sec: float,
    ) -> Any:
        return self._require_evaluator().summarize(
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
        )

    @property
    def records(self) -> list[Any]:
        return list(self._require_evaluator().records)

    @property
    def stage_plans(self) -> list[dict[str, Any]]:
        return list(self._require_evaluator().stage_plans)

    @property
    def batch_size(self) -> int:
        return int(self._require_evaluator().batch_size)

    def close(self) -> None:
        if self.evaluator is not None:
            self.evaluator.close()
            self.evaluator = None

    def _require_evaluator(self) -> RemotePolicyEvaluator:
        if self.evaluator is None:
            raise RuntimeError("SimulatorServiceClient is not connected.")
        return self.evaluator

    @staticmethod
    def _parse_uri(uri: str) -> tuple[str, int]:
        parsed = urlparse(uri)
        if parsed.scheme and parsed.scheme not in {"rpyc"}:
            raise ValueError(
                f"Unsupported simulator URI scheme '{parsed.scheme}'. "
                "Use rpyc://host:port."
            )
        if parsed.scheme:
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 18861
            return host, port
        if ":" in uri:
            host, port_str = uri.rsplit(":", 1)
            return host or "127.0.0.1", int(port_str)
        return uri or "127.0.0.1", 18861

    @staticmethod
    def _normalize_overrides(overrides: list[str]) -> list[str]:
        resolved = list(overrides)
        if not any(item.startswith("env.batch_size=") for item in resolved):
            resolved.append("env.batch_size=1")
        if not any("env.viewer.disable" in item for item in resolved):
            resolved.append("++env.viewer.disable=true")
        return resolved

    def __enter__(self) -> "SimulatorServiceClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
