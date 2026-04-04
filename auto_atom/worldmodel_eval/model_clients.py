"""Model client abstractions for closed-loop evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from .observation_adapter import MODEL_HEATMAP_ORDER


class BaseModelClient(ABC):
    @abstractmethod
    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        """Return a chunked low-level action payload."""

    def close(self) -> None:
        return None


class PayloadValidatingHoldModelClient(BaseModelClient):
    def __init__(
        self,
        *,
        horizon: int,
        payload_key: str = "worldmodel_ws_payload",
        expected_history: int = 5,
    ) -> None:
        self.horizon = int(horizon)
        self.payload_key = payload_key
        self.expected_history = int(expected_history)

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        payload = model_input.get(self.payload_key)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Missing payload dict at key '{self.payload_key}'."
            )
        self._validate_model_input(model_input, payload)

        cartesian = np.asarray(
            payload["observation/cartesian_position"],
            dtype=np.float32,
        ).reshape(-1)
        gripper = np.asarray(
            payload["observation/gripper_position"],
            dtype=np.float32,
        ).reshape(-1)
        if cartesian.shape[0] < 6:
            raise RuntimeError("cartesian_position must contain 6 values.")
        if gripper.size == 0:
            raise RuntimeError("gripper_position must contain at least 1 value.")

        action_row = np.concatenate([cartesian[:6], gripper[:1]], axis=0).astype(
            np.float32
        )
        actions = np.repeat(action_row[None, :], self.horizon, axis=0)
        return {
            "action_format": "cartesian_absolute",
            "action_horizons": np.arange(1, self.horizon + 1, dtype=np.int32),
            "actions": actions,
        }

    def _validate_model_input(
        self,
        model_input: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        selected_cameras = list(model_input.get("selected_cameras", []))
        if not selected_cameras:
            raise RuntimeError("At least one selected camera is required.")

        history_length = int(model_input.get("history_length", 0))
        available_history_length = int(
            model_input.get("available_history_length", history_length)
        )
        if history_length != self.expected_history:
            raise RuntimeError(
                f"history_length must be {self.expected_history}, got {history_length}."
            )
        if available_history_length <= 0 or available_history_length > history_length:
            raise RuntimeError(
                "available_history_length must satisfy 0 < available <= history."
            )

        timestamps_ns = np.asarray(model_input.get("timestamps_ns", []), dtype=np.float64)
        if timestamps_ns.shape != (history_length,):
            raise RuntimeError(
                "timestamps_ns must be a 1D array with length equal to history_length."
            )

        cameras = model_input.get("cameras")
        if not isinstance(cameras, dict):
            raise RuntimeError("model_input must contain a 'cameras' dict.")
        for camera_name in selected_cameras:
            if camera_name not in cameras:
                raise RuntimeError(f"Missing camera payload for '{camera_name}'.")
            camera = cameras[camera_name]
            self._require_history_window(camera, "rgb_window", history_length)
            self._require_history_window(camera, "depth_window_m", history_length)
            self._require_history_window(camera, "mask_window", history_length)
            self._require_history_window(camera, "intrinsics_window", history_length)
            self._require_history_window(camera, "extrinsics_window", history_length)
            self._require_history_window(camera, "model_heatmap_window", history_length)

        rgb_history = self._require_array(
            payload,
            "observation/exterior_image_0_left_history",
            dtype=np.uint8,
            ndim=4,
        )
        if rgb_history.shape[0] != history_length:
            raise RuntimeError(
                "observation/exterior_image_0_left_history must have shape "
                f"({history_length}, H, W, 3), got {rgb_history.shape}."
            )
        if rgb_history.shape[-1] != 3:
            raise RuntimeError("RGB history last dimension must be 3.")

        rgb_current = self._require_array(
            payload,
            "observation/exterior_image_0_left",
            dtype=np.uint8,
            ndim=3,
        )
        if rgb_current.shape != rgb_history.shape[1:]:
            raise RuntimeError(
                "Current RGB frame shape does not match RGB history frame shape."
            )
        if not np.array_equal(rgb_current, rgb_history[-1]):
            raise RuntimeError(
                "Current RGB frame must equal the last history RGB frame."
            )

        depth = self._require_array(
            payload,
            "observation/exterior_depth_0",
            dtype=np.float32,
            ndim=2,
        )
        if depth.shape != rgb_history.shape[1:3]:
            raise RuntimeError("Depth frame shape must match RGB height/width.")

        intrinsics = self._require_array(
            payload,
            "observation/camera_intrinsics",
            dtype=np.float32,
            ndim=3,
        )
        if intrinsics.shape != (3, 3, 3):
            raise RuntimeError(
                "camera_intrinsics must have shape (3, 3, 3), "
                f"got {intrinsics.shape}."
            )
        if not (np.array_equal(intrinsics[0], intrinsics[1]) and np.array_equal(intrinsics[0], intrinsics[2])):
            raise RuntimeError("camera_intrinsics must currently be three copies of the model camera intrinsics.")

        heatmaps = self._require_array(
            payload,
            "observation/heatmaps",
            dtype=np.float32,
            ndim=3,
        )
        if heatmaps.shape[0] != len(MODEL_HEATMAP_ORDER):
            raise RuntimeError(
                "observation/heatmaps must have leading dimension equal to "
                f"{len(MODEL_HEATMAP_ORDER)}, got {heatmaps.shape}."
            )
        if heatmaps.shape[1:] != depth.shape:
            raise RuntimeError("Heatmap H/W must match depth H/W.")

        heatmap_keys = payload.get("observation/heatmap_keys")
        if heatmap_keys is not None and list(heatmap_keys) != list(MODEL_HEATMAP_ORDER):
            raise RuntimeError(
                "observation/heatmap_keys must match "
                f"{list(MODEL_HEATMAP_ORDER)}, got {list(heatmap_keys)}."
            )

        cartesian = self._require_array(
            payload,
            "observation/cartesian_position",
            dtype=np.float32,
            ndim=1,
        )
        if cartesian.shape != (6,):
            raise RuntimeError(
                "observation/cartesian_position must have shape (6,), "
                f"got {cartesian.shape}."
            )
        alias_cartesian = self._require_array(
            payload,
            "observation/robot_state/cartesian_position",
            dtype=np.float32,
            ndim=1,
        )
        if not np.allclose(cartesian, alias_cartesian):
            raise RuntimeError("cartesian_position alias does not match.")

        gripper = np.asarray(
            payload.get("observation/gripper_position"),
            dtype=np.float32,
        ).reshape(-1)
        if gripper.size != 1:
            raise RuntimeError(
                "observation/gripper_position must be a scalar or shape (1,)."
            )
        alias_gripper = np.asarray(
            payload.get("observation/robot_state/gripper_position"),
            dtype=np.float32,
        ).reshape(-1)
        if alias_gripper.size != 1 or not np.allclose(gripper[:1], alias_gripper[:1]):
            raise RuntimeError("gripper_position alias does not match.")

        endpoint = payload.get("endpoint", "infer")
        if endpoint != "infer":
            raise RuntimeError(f"Unsupported endpoint '{endpoint}'.")

        if available_history_length == 1 and history_length > 1:
            repeated_rgb = np.repeat(rgb_history[:1], history_length, axis=0)
            if not np.array_equal(rgb_history, repeated_rgb):
                raise RuntimeError(
                    "Bootstrap RGB history padding must duplicate the first frame."
                )
            repeated_timestamps = np.repeat(timestamps_ns[:1], history_length)
            if not np.array_equal(timestamps_ns, repeated_timestamps):
                raise RuntimeError(
                    "Bootstrap timestamp history padding must duplicate the first frame timestamp."
                )

    @staticmethod
    def _require_history_window(
        camera: dict[str, Any],
        key: str,
        history_length: int,
    ) -> None:
        if key not in camera:
            raise RuntimeError(f"Camera payload missing required field '{key}'.")
        value = np.asarray(camera[key])
        if value.shape[0] != history_length:
            raise RuntimeError(
                f"Camera field '{key}' has history {value.shape[0]} "
                f"but expected {history_length}."
            )
        if value.size == 0:
            raise RuntimeError(f"Camera field '{key}' is empty.")

    @staticmethod
    def _require_array(
        payload: dict[str, Any],
        key: str,
        *,
        dtype: np.dtype,
        ndim: int,
    ) -> np.ndarray:
        if key not in payload:
            raise RuntimeError(f"Missing required payload field '{key}'.")
        array = np.asarray(payload[key], dtype=dtype)
        if array.ndim != ndim:
            raise RuntimeError(
                f"Payload field '{key}' must have rank {ndim}, got {array.shape}."
            )
        return array


class WebSocketModelClient(BaseModelClient):
    def __init__(self, uri: str, *, payload_key: str = "worldmodel_ws_payload") -> None:
        self.uri = uri
        self.payload_key = payload_key
        self.ws: Optional[Any] = None
        self.metadata: Optional[dict[str, Any]] = None

    def connect(self) -> dict[str, Any]:
        websockets_client = self._get_ws_client()
        self.ws = websockets_client.connect(
            self.uri,
            compression=None,
            max_size=None,
        )
        metadata = self.ws.recv()
        if isinstance(metadata, str):
            raise RuntimeError(f"Expected binary model metadata, got text: {metadata}")
        self.metadata = dict(self._unpack(metadata))
        return self.metadata

    def infer(self, model_input: dict[str, Any]) -> dict[str, Any]:
        if self.ws is None:
            self.connect()
        assert self.ws is not None
        payload = model_input.get(self.payload_key)
        if payload is None:
            payload = model_input.get("worldmodel_ws_payload", model_input)
        self.ws.send(self._pack(payload))
        response = self.ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Model server error: {response}")
        return self._normalize_action_payload(dict(self._unpack(response)))

    @staticmethod
    def _normalize_action_payload(response: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in response:
            raise RuntimeError("Model response does not contain 'actions'.")

        payload = dict(response)
        actions = np.asarray(payload["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        payload["actions"] = actions

        if "action_format" not in payload:
            if actions.shape[-1] == 7:
                payload["action_format"] = "cartesian_absolute"
            else:
                raise RuntimeError(
                    f"Cannot infer action_format from actions shape {actions.shape}."
                )

        if "action_horizons" not in payload:
            payload["action_horizons"] = np.arange(
                1,
                actions.shape[0] + 1,
                dtype=np.int32,
            )
        else:
            payload["action_horizons"] = np.asarray(
                payload["action_horizons"],
                dtype=np.int32,
            )

        return payload

    def close(self) -> None:
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    @staticmethod
    def _get_ws_client() -> Any:
        try:
            import websockets.sync.client as ws_client
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "websockets is required for WebSocketModelClient."
            ) from exc
        return ws_client

    @staticmethod
    def _pack(obj: Any) -> bytes:
        try:
            import msgpack
            import msgpack_numpy as msgpack_numpy_lib
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "msgpack and msgpack_numpy are required for WebSocketModelClient."
            ) from exc
        msgpack_numpy_lib.patch()
        return msgpack.packb(obj, default=msgpack_numpy_lib.encode)

    @staticmethod
    def _unpack(data: bytes) -> Any:
        try:
            import msgpack
            import msgpack_numpy as msgpack_numpy_lib
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "msgpack and msgpack_numpy are required for WebSocketModelClient."
            ) from exc
        msgpack_numpy_lib.patch()
        return msgpack.unpackb(
            data,
            object_hook=msgpack_numpy_lib.decode,
            raw=False,
        )
