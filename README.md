<div align="center">

<h1>Auto Atomic Operation</h1>

[![PyPI](https://img.shields.io/pypi/v/auto-atomic-operation)](https://pypi.org/project/auto-atomic-operation/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A YAML-driven atomic operation framework for robotic manipulation.

</div>

`auto-atomic-operation` lets you define robotic manipulation tasks — pick, place, push, pull, move, grasp, release — as declarative YAML files. A built-in state machine handles task sequencing, pose resolution, end-effector control, and execution tracking. A plugin-based backend system decouples task logic from the underlying hardware or simulator, making it easy to run the same task definition against a real robot, a physics simulator, or a lightweight mock for testing.

## Features

- **Hydra-powered task configuration** — describe multi-stage manipulation tasks in YAML; full [Hydra](https://hydra.cc) support means `_target_` instantiation, variable interpolation, and command-line overrides work out of the box
- **Flexible pose references** — specify targets relative to world, robot base, end-effector, or tracked objects
- **Pluggable backends** — connect any robot or simulator by implementing a small set of abstract interfaces
- **Pose randomization** — per-object position/orientation randomization with automatic collision avoidance on reset
- **Multi-arm support** — single-arm and dual-arm (left/right) topologies
- **Execution records** — detailed per-stage status, failure reasons, and timing after every run
- **Mujoco backend included** — a ready-to-use backend with RGB-D cameras, tactile sensors, force/torque, IMU, and joint state support

## Installation

Requires **Python 3.10+**.

```bash
# Core framework only
pip install -e .

# With the built-in Mujoco backend
pip install -e ".[mujoco]"
```

## Quick Start

### 1. Define a task in YAML

```yaml
# task.yaml
backend: auto_atom.mock.build_mock_backend

task:
  env_name: my_env
  stages:
    - name: pick_cup
      object: cup
      operation: pick
      operator: arm_a
      param:
        pre_move:
          - position: [0.45, -0.10, 0.08]
            rotation: [0.0, 1.57, 0.0]
            reference: object_world
        eef:
          close: true

    - name: place_on_shelf
      object: shelf
      operation: place
      operator: arm_a
      param:
        pre_move:
          - position: [0.10, 0.25, 0.16]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world
        eef:
          close: false

operators:
  - name: arm_a
    role: manipulator
```

### 2. Run the task

```python
from pathlib import Path
from auto_atom.runtime import ComponentRegistry, TaskRunner

ComponentRegistry.clear()
runner = TaskRunner().from_yaml(Path("task.yaml"))

runner.reset()
while True:
    update = runner.update()
    if update.done:
        break

for record in runner.records:
    print(record)

runner.close()
```

## YAML Configuration Reference

Task files are processed by [Hydra](https://hydra.cc) / OmegaConf, so the full Hydra feature set is available: `_target_` instantiation, `${key}` interpolation, structured configs, and command-line overrides (`key=value`).

A task file has four top-level keys:

| Key | Description |
|-----|-------------|
| `env` | Hydra `_target_` instantiation of the environment, registered via `ComponentRegistry` |
| `backend` | Dotted import path to the backend factory function |
| `task` | Task definition: `env_name`, `seed`, and a list of `stages` |
| `operators` | Named operators with assigned roles |

### Stage definition

```yaml
- name: <stage_name>          # Unique stage identifier
  object: <object_name>       # Target object
  operation: pick|place|push|pull|move|grasp|release
  operator: <operator_name>   # Which arm/robot executes this stage
  blocking: true              # Optional, default true
  param:
    pre_move:                 # Approach waypoints (list of pose configs)
      - position: [x, y, z]
        rotation: [rx, ry, rz]    # Euler angles (rad), or:
        orientation: [x, y, z, w] # Quaternion
        reference: world|base|eef|object|object_world|eef_world|auto
    post_move:                # Retreat waypoints (same format as pre_move)
    eef:
      close: true|false       # Gripper open/close
```

### Pose references

| Reference | Description |
|-----------|-------------|
| `world` | Fixed world frame |
| `base` | Robot base frame |
| `eef` | Current end-effector frame |
| `object` | Object frame (tracks object movement) |
| `object_world` | Object position with world orientation |
| `eef_world` | EEF position snapshot at command start |
| `auto` | Automatically determined from context |

## Examples

The [`examples/`](examples/) directory contains two runnable demos:

### Mock example (no robot or simulator required)

```bash
python examples/run_mock_example.py
```

Uses the in-memory mock backend — ideal for testing task logic in isolation.

### Mujoco pick-and-place demo

```bash
python examples/run_mujoco_pick_place_demo.py
```

A full pick-and-place task with RGB-D cameras, tactile sensors, and randomized object placement, running in the Mujoco physics simulator.

## Architecture

```
auto_atom/
├── framework.py        # Pydantic configuration models (YAML schema)
├── runtime.py          # Task execution engine (TaskRunner, TaskFlowBuilder)
├── mock.py             # In-memory mock backend for testing
├── basis/
│   └── mujoco_env.py   # UnifiedMujocoEnv — Mujoco wrapper with sensor suite
├── backend/
│   └── mjc/            # Mujoco backend (operators, objects, scene)
└── utils/
    └── pose.py         # Pose transforms and quaternion utilities
```

**Execution flow:**

1. Load and validate the YAML task file via Pydantic models
2. Instantiate the backend via the configured factory function
3. `TaskFlowBuilder` expands stages into primitive pose-move and EEF-control actions
4. `TaskRunner.reset()` initializes the scene (with optional randomization)
5. `TaskRunner.update()` advances one step of the active primitive action
6. After completion, `TaskRunner.records` holds per-stage execution history

## Implementing a Custom Backend

To integrate a new simulator or real robot, implement three abstract classes from `auto_atom.runtime`:

- `OperatorHandler` — arm movement (`move_to_pose`) and gripper control (`control_eef`)
- `ObjectHandler` — object pose queries and updates (`get_pose`, `set_pose`)
- `SceneBackend` — scene lifecycle, handler resolution, and randomization

See the **[Custom Backend Guide](docs/custom-backend.md)** for a step-by-step walkthrough with annotated code examples. [`auto_atom/mock.py`](auto_atom/mock.py) provides a minimal reference implementation.

## License

See [LICENSE](LICENSE).
