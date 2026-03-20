"""Run the YAML-driven mock task flow example."""

from pathlib import Path
from auto_atom.mock import build_mock_backend
from auto_atom.runtime import ComponentRegistry, TaskRunner


ENV_NAME = "mock_single_arm"


def build_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register_backend("mock", build_mock_backend)
    registry.register_env(ENV_NAME, {"kind": "mock_env"})
    return registry


def main() -> None:
    config_path = Path(__file__).with_name("mock_task.yaml")
    runner = TaskRunner(registry=build_registry()).from_yaml(config_path)

    print("Reset task")
    print(runner.reset())
    print()

    while True:
        update = runner.update()
        print(update)
        if update.done:
            break

    print()
    print("Execution records:")
    for record in runner.records:
        print(record)

    runner.close()


if __name__ == "__main__":
    main()
