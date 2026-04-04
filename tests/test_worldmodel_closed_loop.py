from __future__ import annotations

import numpy as np

from auto_atom.runtime import ExecutionSummary
from auto_atom.runner.worldmodel_closed_loop_eval import _summary_indicates_success
from auto_atom.worldmodel_eval import (
    MODEL_HEATMAP_ORDER,
    ObservationWindowAdapter,
    PayloadValidatingHoldModelClient,
)


def _make_sim_info(camera_names: list[str], height: int, width: int) -> dict:
    intrinsics = [100.0, 0.0, width / 2.0, 0.0, 100.0, height / 2.0, 0.0, 0.0, 1.0]
    return {
        "cameras": {
            camera_name: {
                "camera_info": {
                    "color": {"height": height, "width": width, "k": intrinsics},
                    "depth": {"height": height, "width": width, "k": intrinsics},
                },
                "camera_extrinsics": {
                    "translation": [0.0, 0.0, 0.0],
                    "rotation_matrix": np.eye(3).tolist(),
                },
            }
            for camera_name in camera_names
        }
    }


def _make_observation(
    camera_names: list[str],
    *,
    height: int,
    width: int,
    timestamp_ns: int,
) -> dict[str, dict]:
    observation: dict[str, dict] = {
        "arm/pose/position": {
            "data": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            "t": timestamp_ns,
        },
        "arm/pose/rotation": {
            "data": np.asarray([0.4, 0.5, 0.6], dtype=np.float32),
            "t": timestamp_ns,
        },
        "eef/joint_state/position": {
            "data": np.asarray([0.7], dtype=np.float32),
            "t": timestamp_ns,
        },
    }

    heat_map = np.zeros((height, width, len(MODEL_HEATMAP_ORDER)), dtype=np.float32)
    for index, _name in enumerate(MODEL_HEATMAP_ORDER):
        heat_map[..., index] = index + 1

    for cam_index, camera_name in enumerate(camera_names):
        rgb = np.full((height, width, 3), 10 + cam_index, dtype=np.uint8)
        depth = np.full((height, width), 0.01 * (cam_index + 1), dtype=np.float32)
        mask = np.full((height, width), cam_index, dtype=np.uint8)
        observation[f"{camera_name}/color/image_raw"] = {"data": rgb, "t": timestamp_ns}
        observation[f"{camera_name}/aligned_depth_to_color/image_raw"] = {
            "data": depth,
            "t": timestamp_ns,
        }
        observation[f"{camera_name}/mask/image_raw"] = {
            "data": mask,
            "t": timestamp_ns,
        }
        observation[f"{camera_name}/mask/heat_map"] = {
            "data": heat_map,
            "t": timestamp_ns,
        }

    return observation


def test_observation_adapter_bootstrap_history() -> None:
    camera_names = ["front_cam", "hand_cam", "side_cam"]
    sim_info = _make_sim_info(camera_names, height=4, width=6)
    observation = _make_observation(
        camera_names,
        height=4,
        width=6,
        timestamp_ns=123456789,
    )
    adapter = ObservationWindowAdapter(
        sim_info=sim_info,
        selected_cameras=camera_names,
        history_frames=5,
        task_operations=list(MODEL_HEATMAP_ORDER),
        model_camera="front_cam",
    )
    adapter.extend([observation])

    model_input = adapter.build_model_input()
    payload = model_input["worldmodel_ws_payload"]

    assert model_input["history_length"] == 5
    assert model_input["available_history_length"] == 1
    assert payload["observation/exterior_image_0_left_history"].shape == (5, 4, 6, 3)
    assert payload["observation/exterior_depth_0"].shape == (4, 6)
    assert payload["observation/camera_intrinsics"].shape == (3, 3, 3)
    assert payload["observation/heatmaps"].shape == (5, 4, 6)
    assert np.array_equal(
        payload["observation/exterior_image_0_left_history"][0],
        payload["observation/exterior_image_0_left_history"][-1],
    )
    assert np.array_equal(
        model_input["timestamps_ns"],
        np.repeat(model_input["timestamps_ns"][:1], 5),
    )
    assert np.allclose(
        payload["observation/heatmaps"][:, 0, 0],
        np.asarray([1, 2, 3, 4, 5], dtype=np.float32),
    )


def test_payload_validating_hold_model_client() -> None:
    camera_names = ["front_cam", "hand_cam", "side_cam"]
    sim_info = _make_sim_info(camera_names, height=4, width=6)
    observation = _make_observation(
        camera_names,
        height=4,
        width=6,
        timestamp_ns=123456789,
    )
    adapter = ObservationWindowAdapter(
        sim_info=sim_info,
        selected_cameras=camera_names,
        history_frames=5,
        task_operations=list(MODEL_HEATMAP_ORDER),
        model_camera="front_cam",
    )
    adapter.extend([observation])
    model_input = adapter.build_model_input()

    client = PayloadValidatingHoldModelClient(horizon=8, expected_history=5)
    payload = client.infer(model_input)

    assert payload["action_format"] == "cartesian_absolute"
    assert payload["actions"].shape == (8, 7)
    assert np.array_equal(payload["action_horizons"], np.arange(1, 9, dtype=np.int32))
    expected_row = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    assert np.allclose(payload["actions"], np.repeat(expected_row[None, :], 8, axis=0))


def test_summary_indicates_success_requires_done_and_true_success() -> None:
    not_done = ExecutionSummary(
        total_stages=2,
        max_updates=5,
        updates_used=5,
        completed_stage_count=np.asarray([0], dtype=np.int32),
        final_stage_index=np.asarray([0], dtype=np.int32),
        final_stage_name=np.asarray(["pick"], dtype=object),
        final_status=np.asarray(["running"], dtype=object),
        final_done=np.asarray([False], dtype=bool),
        final_success=np.asarray([None], dtype=object),
        elapsed_time_sec=1.0,
        records=[],
    )
    assert _summary_indicates_success(not_done) is False

    done_success = ExecutionSummary(
        total_stages=2,
        max_updates=5,
        updates_used=3,
        completed_stage_count=np.asarray([2], dtype=np.int32),
        final_stage_index=np.asarray([1], dtype=np.int32),
        final_stage_name=np.asarray(["place"], dtype=object),
        final_status=np.asarray(["succeeded"], dtype=object),
        final_done=np.asarray([True], dtype=bool),
        final_success=np.asarray([True], dtype=object),
        elapsed_time_sec=1.0,
        records=[],
    )
    assert _summary_indicates_success(done_success) is True
