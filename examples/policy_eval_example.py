"""Minimal PolicyEvaluator example with a mock random policy.

Demonstrates:
- Loading a task config via ``load_task_file_hydra`` (Hydra compose, supports defaults)
- Constructing a ``PolicyEvaluator`` with custom action_applier / observation_getter
- Running the full evaluation loop: reset → act → update → summarize

Usage::

    cd <project_root>
    python examples/policy_eval_example.py
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from auto_atom import (
    ExecutionContext,
    PolicyEvaluator,
    TaskUpdate,
    load_task_file_hydra,
)


# ---------------------------------------------------------------------------
# Mock action applier / observation getter
# ---------------------------------------------------------------------------
# The mock backend has no env.step() or env.capture_observation(), so we
# provide lightweight stubs that keep the loop running.


def my_action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    """Apply a batched action array to the environment (ctrl mode).

    This example uses **ctrl mode**: the action vector is written directly to
    MuJoCo's ``data.ctrl`` (one dimension per actuator — joint targets in
    radians).  This fully controls joint-mode robots, but for mocap robots it
    only drives the gripper (the arm requires pose-mode commands).

    For pose-mode control (EEF target position + orientation), use
    ``env.step_operator_toward_target()`` instead.  See ``docs/action_space.md``
    for a full comparison of ctrl vs. pose mode and the record/replay pipeline.
    """
    env = context.backend.env

    # Ensure action is a 2-D float64 array: (batch_size, action_dim)
    action = np.asarray(action, dtype=np.float64)
    if action.ndim == 1:
        action = action.reshape(1, -1)

    env.step(action, env_mask=env_mask)


def my_observation_getter(context: ExecutionContext) -> dict:
    """Capture observations from the environment.

    ``env.capture_observation()`` returns a dict keyed by sensor channel,
    e.g.::

        {
            "enc/arm/joint_state":  {"data": np.ndarray, "t": ...},
            "hand_cam/color":       {"data": np.ndarray, "t": ...},
            "front_cam/depth":      {"data": np.ndarray, "t": ...},
            ...
        }

    Each entry has ``"data"`` (batched along axis-0 when batch_size > 1)
    and ``"t"`` (timestamp).
    """
    return context.backend.env.capture_observation()


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------


class RandomPolicy:
    """Trivial policy that returns random 7-DoF actions."""

    def __init__(self, action_dim: int = 7, scale: float = 0.01) -> None:
        self.action_dim = action_dim
        self.scale = scale

    def reset(self) -> None:
        pass

    def act(
        self,
        observation: Any,
        update: TaskUpdate,
        evaluator: PolicyEvaluator,
    ) -> np.ndarray:
        return self.scale * np.random.randn(evaluator.batch_size, self.action_dim)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Load config (supports Hydra defaults merging)
    task_file = load_task_file_hydra("policy_eval_mock")

    # 2. Build policy
    policy = RandomPolicy()

    # 3. Build evaluator
    evaluator = PolicyEvaluator(
        action_applier=my_action_applier,
        observation_getter=my_observation_getter,
    ).from_config(task_file)

    # 4. Run evaluation loop
    max_updates = 20
    try:
        policy.reset()
        update = evaluator.reset()
        step = -1

        print(f"Task stages: {[s.name for s in task_file.task.stages]}")
        print(f"Batch size:  {evaluator.batch_size}")
        print(f"Max updates: {max_updates}")
        print()

        for step in range(max_updates):
            observation = evaluator.get_observation()
            action = policy.act(observation, update=update, evaluator=evaluator)
            update = evaluator.update(action)

            if step % 5 == 0 or update.done.all():
                print(
                    f"  step {step:3d} | "
                    f"stage={update.stage_name!r:30s} | "
                    f"done={update.done.tolist()} | "
                    f"success={update.success.tolist()}"
                )

            if update.done.all():
                break

        # 5. Summarize
        summary = evaluator.summarize(
            update, max_updates=max_updates, updates_used=step + 1
        )
        print()
        print("=== Execution Summary ===")
        print(f"  updates_used:          {summary.updates_used}")
        print(f"  completed_stage_count: {summary.completed_stage_count}")
        print(f"  final_stage_index:     {summary.final_stage_index}")
        print(f"  final_stage_name:      {summary.final_stage_name}")
        print(f"  final_done:            {summary.final_done}")
        print(f"  final_success:         {summary.final_success}")

        # 6. Records
        records = evaluator.records
        print()
        print(f"=== Execution Records ({len(records)}) ===")
        for record in records:
            print(f"  {record}")

    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
