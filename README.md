# auto-atomic-operation
An automated atomic operation framework based on Mujoco.

## YAML-driven mock example

Use the provided example to build a task runner from YAML and then drive it with `reset()` and repeated `update()` calls:

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python examples/run_mock_example.py
```

The runtime abstractions and configuration schema now both live in `auto_atom/`.
