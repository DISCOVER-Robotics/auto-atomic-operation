"""Run a simple pick-and-place demo using the Mujoco backend."""

from pathlib import Path
from auto_atom.runtime import ComponentRegistry, TaskRunner


def main() -> None:
    config_path = Path(__file__).with_name("mujoco_pick_place_demo.yaml")
    ComponentRegistry.clear()
    runner = TaskRunner().from_yaml(config_path)

    try:
        print("Reset task")
        print(runner.reset())
        print()

        while True:
            update = runner.update()
            print(update)
            if update.done:
                break

        backend = runner._require_context().backend
        source_pose = backend.get_object_handler("source_block").get_pose()
        target_pose = backend.get_object_handler("target_pedestal").get_pose()

        print()
        print("Final poses:")
        print("source_block:", source_pose)
        print("target_pedestal:", target_pose)
        print()
        print("Execution records:")
        for record in runner.records:
            print(record)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
