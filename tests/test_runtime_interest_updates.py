from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runtime import ComponentRegistry, TaskRunner
from auto_atom.mock import MockSimulatorBackend, build_mock_backend


ENV_NAME = "mock_single_arm"


def build_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register_backend("mock", build_mock_backend)
    registry.register_env(ENV_NAME, {"kind": "mock_env"})
    return registry


def test_runner_propagates_interest_objects_and_operations() -> None:
    task_path = ROOT / "tests" / "runtime_interest_updates_task.yaml"
    task_path.write_text(
        f"""task:
  env_name: {ENV_NAME}
  simulator: mock
  seed: 7
  stages:
    - name: move_to_cup
      object: cup
      operation: move
      operator: arm_a
      param:
        pre_move:
          - position: [0.45, -0.10, 0.08]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world
    - name: move_to_tray
      object: tray
      operation: move
      operator: arm_a
      param:
        pre_move:
          - position: [0.10, 0.25, 0.16]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world

operators:
  - type: mock
    name: arm_a
    role: manipulator
"""
    )

    runner = TaskRunner(registry=build_registry()).from_yaml(task_path)

    try:
        update = runner.reset()
        assert update.stage_name == "move_to_cup"

        backend = runner._require_context().backend
        assert isinstance(backend, MockSimulatorBackend)
        assert backend.interest_updates == [{"objects": [], "operations": []}]

        update = runner.update()
        assert update.stage_name == "move_to_cup"
        assert backend.interest_updates[-1] == {
            "objects": ["cup"],
            "operations": ["move"],
        }

        while True:
            update = runner.update()
            if update.stage_name == "move_to_cup" and update.status.value == "succeeded":
                break

        update = runner.update()
        assert update.stage_name == "move_to_tray"
        assert backend.interest_updates[-1] == {
            "objects": ["tray"],
            "operations": ["move"],
        }

        while not update.done:
            update = runner.update()

        assert update.success is True
        assert backend.interest_updates[-1] == {"objects": [], "operations": []}
    finally:
        runner.close()
        task_path.unlink(missing_ok=True)
