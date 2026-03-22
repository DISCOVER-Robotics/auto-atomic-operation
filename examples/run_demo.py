"""Run a demo using the Mujoco backend.

Config files live in the ``mujoco/`` subdirectory. The default is
``mujoco/pick_and_place.yaml``. Any value can be overridden from the command
line via Hydra, e.g.::

    # Run the default pick-and-place demo
    python run_demo.py

    # Run the mock demo
    python run_demo.py --config-name mock

    # Override individual values
    python run_demo.py task.seed=0
    python run_demo.py task.randomization.source_block.x="[-0.05,0.05]"
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


@hydra.main(config_path="mujoco", config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)

    try:
        print("Reset task (randomization loaded from YAML)")
        print(runner.reset())
        print()

        backend = runner._require_context().backend
        source_pose = backend.get_object_handler("source_block").get_pose()
        print(f"source_block pose after reset: {source_pose}")
        print()

        while True:
            update = runner.update()
            print(update)
            if update.done:
                break

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
