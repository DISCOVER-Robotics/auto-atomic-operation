"""Closed-loop WorldModel-style evaluation on top of the remote PolicyEvaluator service."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from auto_atom.worldmodel_eval import (
    BaseModelClient,
    EpisodeRecorder,
    ObservationWindowAdapter,
    PayloadValidatingHoldModelClient,
    SimulatorServiceClient,
    WebSocketModelClient,
    to_jsonable,
)

DEFAULT_TASKS = (
    "cup_on_coaster_gs",
    "arrange_flowers_gs",
    "wipe_the_table_gs",
)


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _unique_operations(stage_plans: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for plan in stage_plans:
        operation = str(plan.get("operation", "")).strip()
        if not operation or operation in seen:
            continue
        seen.add(operation)
        ordered.append(operation)
    return ordered


def _select_cameras(sim_info: dict[str, Any], requested_cameras: list[str]) -> list[str]:
    available = list(sim_info.get("cameras", {}).keys())
    if requested_cameras:
        missing = [name for name in requested_cameras if name not in available]
        if missing:
            raise RuntimeError(
                f"Requested cameras {missing} are not available. "
                f"Available cameras: {available}"
            )
        return list(requested_cameras)
    return available


def _model_input_summary(model_input: dict[str, Any]) -> dict[str, Any]:
    ws_payload = dict(model_input.get("worldmodel_ws_payload", {}))
    summary: dict[str, Any] = {
        "history_length": int(model_input.get("history_length", 0)),
        "available_history_length": int(model_input.get("available_history_length", 0)),
        "selected_cameras": list(model_input.get("selected_cameras", [])),
        "model_camera": model_input.get("model_camera"),
        "task_operations": list(model_input.get("task_operations", [])),
        "timestamps_ns": np.asarray(model_input.get("timestamps_ns", []), dtype=np.float64),
        "worldmodel_ws_payload": {},
    }
    for key, value in ws_payload.items():
        if isinstance(value, np.ndarray):
            summary["worldmodel_ws_payload"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary["worldmodel_ws_payload"][key] = value
    return summary


def _make_model_client(
    *,
    model_mode: str,
    horizon: int,
    model_uri: str,
    model_payload_key: str,
    history_frames: int,
) -> BaseModelClient:
    if model_mode == "mock_validate":
        return PayloadValidatingHoldModelClient(
            horizon=horizon,
            payload_key=model_payload_key,
            expected_history=history_frames,
        )
    if model_mode == "ws":
        if not model_uri:
            raise ValueError("--model-uri is required when --model-mode=ws")
        client = WebSocketModelClient(model_uri, payload_key=model_payload_key)
        client.connect()
        return client
    raise ValueError(f"Unsupported model mode '{model_mode}'.")


def _aggregate_payload(episodes: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    success_count = sum(1 for episode in episodes if episode["success"])
    by_task: dict[str, dict[str, Any]] = {}
    task_names = sorted({episode["task_name"] for episode in episodes})
    for task_name in task_names:
        task_episodes = [episode for episode in episodes if episode["task_name"] == task_name]
        task_success = sum(1 for episode in task_episodes if episode["success"])
        by_task[task_name] = {
            "episodes": len(task_episodes),
            "successes": task_success,
            "success_rate": 0.0 if not task_episodes else task_success / len(task_episodes),
        }
    return {
        "output_root": str(output_root),
        "episodes": episodes,
        "overall": {
            "episodes": len(episodes),
            "successes": success_count,
            "success_rate": 0.0 if not episodes else success_count / len(episodes),
        },
        "by_task": by_task,
    }


def _summary_indicates_success(summary: Any) -> bool:
    final_done = np.asarray(summary.final_done, dtype=bool).reshape(-1)
    if final_done.size == 0 or not bool(np.all(final_done)):
        return False
    final_success = np.asarray(summary.final_success, dtype=object).reshape(-1)
    if final_success.size == 0:
        return False
    return all(
        bool(value) and value is not None
        for value in final_success.tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop WorldModel evaluation via the remote PolicyEvaluator service",
    )
    parser.add_argument("--sim-uri", type=str, default="rpyc://127.0.0.1:18861")
    parser.add_argument(
        "--model-mode",
        type=str,
        default="mock_validate",
        choices=("mock_validate", "ws"),
    )
    parser.add_argument("--model-uri", type=str, default="")
    parser.add_argument(
        "--model-payload-key",
        type=str,
        default="worldmodel_ws_payload",
        help="Model input payload key to send over WebSocket.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated Hydra config names.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--history-frames", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=128)
    parser.add_argument(
        "--max-updates",
        type=int,
        default=500,
        help="Hard limit on executed simulator steps per episode. Reaching it counts as failure.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--action-format",
        type=str,
        default="cartesian_absolute",
        choices=("cartesian_absolute",),
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="",
        help="Comma-separated camera names. Empty means all cameras from get_info().",
    )
    parser.add_argument(
        "--model-camera",
        type=str,
        default="",
        help="Primary camera name used to build the model payload.",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(Path.cwd() / "outputs" / "worldmodel_closed_loop_eval"),
    )
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument(
        "--save-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    tasks = _parse_csv(args.tasks)
    requested_cameras = _parse_csv(args.cameras)
    output_root = Path(args.output_root).resolve()
    episodes_root = output_root / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    aggregate_episodes: list[dict[str, Any]] = []

    for task_offset, task_name in enumerate(tasks):
        for episode_index in range(args.episodes_per_task):
            seed = args.seed_base + task_offset * args.episodes_per_task + episode_index
            episode_name = f"{task_name}_ep{episode_index:03d}_seed{seed}"
            episode_dir = episodes_root / episode_name
            overrides = list(args.override)
            overrides.append(f"task.seed={seed}")

            summary = None
            records: list[Any] = []
            error_text: str | None = None
            updates_used = 0
            selected_cameras: list[str] = []
            task_operations: list[str] = []
            metadata: dict[str, Any] = {
                "task_name": task_name,
                "episode_name": episode_name,
                "seed": seed,
                "sim_uri": args.sim_uri,
                "model_mode": args.model_mode,
                "model_uri": args.model_uri,
                "horizon": args.horizon,
                "stride": args.stride,
                "history_frames": args.history_frames,
                "max_updates": args.max_updates,
                "fps": args.fps,
                "action_format": args.action_format,
                "requested_cameras": requested_cameras,
                "model_camera_arg": args.model_camera,
                "overrides": overrides,
                "chunk_requests": [],
            }
            recorder: EpisodeRecorder | None = None

            try:
                with SimulatorServiceClient(args.sim_uri) as simulator_client:
                    model_client = _make_model_client(
                        model_mode=args.model_mode,
                        horizon=args.horizon,
                        model_uri=args.model_uri,
                        model_payload_key=args.model_payload_key,
                        history_frames=args.history_frames,
                    )
                    try:
                        sim_metadata = simulator_client.metadata or {}
                        init_response = simulator_client.init(
                            config_name=task_name,
                            overrides=overrides,
                            action_format=args.action_format,
                        )
                        sim_info = simulator_client.get_info()
                        stage_plans = simulator_client.stage_plans
                        task_operations = _unique_operations(stage_plans)
                        selected_cameras = _select_cameras(sim_info, requested_cameras)
                        model_camera = args.model_camera or (
                            selected_cameras[0] if selected_cameras else ""
                        )

                        metadata.update(
                            {
                                "sim_metadata": sim_metadata,
                                "init_response": init_response,
                                "sim_info": sim_info,
                                "stage_plans": stage_plans,
                                "task_operations": task_operations,
                                "selected_cameras": selected_cameras,
                                "model_camera": model_camera,
                            }
                        )

                        recorder = EpisodeRecorder(
                            episode_dir=episode_dir,
                            episode_name=episode_name,
                            task_name=task_name,
                            camera_names=selected_cameras,
                            fps=args.fps,
                            save_arrays=args.save_arrays,
                            save_video=args.save_video,
                        )

                        reset_update = simulator_client.reset()
                        initial_observation = simulator_client.get_observation()
                        adapter = ObservationWindowAdapter(
                            sim_info=sim_info,
                            selected_cameras=selected_cameras,
                            history_frames=args.history_frames,
                            task_operations=task_operations,
                            model_camera=model_camera,
                        )
                        adapter.extend([initial_observation])
                        metadata["reset_update"] = reset_update

                        recorder.record(
                            step_index=-1,
                            chunk_index=-1,
                            chunk_step_index=-1,
                            sim_frame=adapter.latest_frame(),
                            update=reset_update,
                            action_cartesian=None,
                            remote_action=None,
                        )

                        start_time = perf_counter()
                        rollout_done = bool(np.all(np.asarray(reset_update.done, dtype=bool)))
                        for chunk_index in range(args.max_chunks):
                            if rollout_done or updates_used >= args.max_updates:
                                break
                            model_input = adapter.build_model_input()
                            action_payload = model_client.infer(model_input)
                            actions = np.asarray(
                                action_payload.get("actions"),
                                dtype=np.float32,
                            )
                            if actions.ndim != 2 or actions.shape[1] != 7:
                                raise RuntimeError(
                                    "Model actions must have shape (T, 7), "
                                    f"got {actions.shape}."
                                )
                            if action_payload.get("action_format") != "cartesian_absolute":
                                raise RuntimeError(
                                    "Model action_format must be 'cartesian_absolute'."
                                )
                            remaining_updates = args.max_updates - updates_used
                            apply_steps = min(
                                args.stride,
                                actions.shape[0],
                                remaining_updates,
                            )
                            if apply_steps <= 0:
                                break

                            metadata["chunk_requests"].append(
                                {
                                    "chunk_index": chunk_index,
                                    "model_input": _model_input_summary(model_input),
                                    "action_payload": action_payload,
                                    "applied_steps": apply_steps,
                                }
                            )

                            for chunk_step_index in range(apply_steps):
                                update, remote_action = simulator_client.update_cartesian_action(
                                    actions[chunk_step_index]
                                )
                                updates_used += 1
                                observation = simulator_client.get_observation()
                                adapter.extend([observation])
                                recorder.record(
                                    step_index=updates_used - 1,
                                    chunk_index=chunk_index,
                                    chunk_step_index=chunk_step_index,
                                    sim_frame=adapter.latest_frame(),
                                    update=update,
                                    action_cartesian=actions[chunk_step_index],
                                    remote_action=remote_action,
                                )
                                rollout_done = bool(
                                    np.all(np.asarray(update.done, dtype=bool))
                                )
                                if rollout_done:
                                    break

                        elapsed_time_sec = perf_counter() - start_time
                        summary = simulator_client.summarize(
                            max_updates=args.max_updates,
                            updates_used=updates_used,
                            elapsed_time_sec=elapsed_time_sec,
                        )
                        records = simulator_client.records
                    finally:
                        model_client.close()
            except Exception as exc:
                error_text = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                metadata["traceback"] = traceback.format_exc()

            if recorder is not None:
                recorder.finalize(
                    summary=summary,
                    records=records,
                    metadata=metadata,
                    error=error_text,
                )
            else:
                episode_dir.mkdir(parents=True, exist_ok=True)
                with (episode_dir / "summary.json").open("w", encoding="utf-8") as fp:
                    json.dump(
                        {
                            "task_name": task_name,
                            "episode_name": episode_name,
                            "error": error_text,
                        },
                        fp,
                        indent=2,
                    )

            final_success = False
            if summary is not None:
                final_success = _summary_indicates_success(summary)

            aggregate_episodes.append(
                {
                    "task_name": task_name,
                    "episode_name": episode_name,
                    "seed": seed,
                    "success": final_success,
                    "error": error_text,
                    "updates_used": updates_used,
                    "selected_cameras": selected_cameras,
                    "task_operations": task_operations,
                    "episode_dir": str(episode_dir),
                    "summary": summary,
                }
            )

    aggregate_payload = _aggregate_payload(aggregate_episodes, output_root)
    with (output_root / "aggregate_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(to_jsonable(aggregate_payload), fp, indent=2)

    print(json.dumps(to_jsonable(aggregate_payload["overall"]), indent=2))


if __name__ == "__main__":
    main()
