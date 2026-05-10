# Scene Composition

`auto_atom` composes the scene XML and the robot XML(s) at load time instead of authoring a separate per-robot demo file. Scene XMLs declare only task-specific geometry (tables, objects, cameras, sites); robot XMLs are pulled in via `env.robot_paths` and inlined into the scene's `<mujoco>` root with their asset paths absolutized, so robot and scene can use independent `meshdir` / `texturedir` settings.

## Why

Before this refactor, each robot needed its own `demo_<robot>.xml` (e.g. `demo_p7_xf9600.xml`, `demo_franka.xml`) duplicating the entire scene plus that robot's keyframe. Adding a new robot meant copy-pasting and re-tuning every task scene.

Now:

- One `assets/xmls/scenes/<task>/demo.xml` holds the scene only — no robot include, no `<key>` keyframe.
- Robots live under `assets/xmls/robots/` and are referenced by config.
- Home pose is described once in YAML as `env.initial_joint_positions` instead of an XML keyframe.

## Configuring `robot_paths`

Set `env.robot_paths` to the list of robot XMLs the task should compose with the scene. Order matters — earlier entries are inserted first. Most basis configs already do this for you:

```yaml
# aao_configs/basis_p7_xf9600.yaml
env:
  robot_paths:
    - ${assets_dir}/xmls/robots/p7_arm_with_xf9600.xml
  initial_joint_positions:
    joint1: 0.0
    joint2: -0.785
    # ... 7 P7 hinges + XF9600 gripper joints
```

When the scene XML already embeds its own robot (legacy monolithic scenes), leave `robot_paths` empty (the default in [`aao_configs/basis.yaml`](../../aao_configs/basis.yaml)). The loader takes a fast path and skips XML rewriting.

## Loader behaviour

`auto_atom.utils.scene_loader.load_scene` is the single entry point:

| Case                          | Behaviour                                                                        |
|-------------------------------|----------------------------------------------------------------------------------|
| `robot_paths=[]`              | `mujoco.MjModel.from_xml_path(scene_xml)` — no rewriting                         |
| `robot_paths=[r1, r2, ...]`   | Parse `scene_xml` and each robot XML, **recursively expand** every `<include>` inside the robot tree, **absolutize** all asset paths and `compiler` `meshdir` / `texturedir` / `assetdir` attributes against each robot file's own directory, then **inline** the robot's children directly into `<mujoco>`. The composed document is written to a temp sibling of the scene file, loaded, and deleted. |

The inline-with-absolutized-paths strategy means robot XMLs and scene XMLs can use different `meshdir` settings (e.g. `assets/meshes/p7_arm` vs `assets/meshes`) without one clobbering the other in MuJoCo's parse-time meshdir merge. Asset paths inside the robot are resolved before inlining, so the host scene's compiler context is never perturbed and the loader is independent of CWD. Relative `meshdir`, `texturedir`, and `<include>` references inside the *scene* itself continue to resolve as before, because the composed file is still written next to the scene XML.

## Home pose injection

Because the scene XML no longer carries a `<key>` keyframe, the runtime applies `env.initial_joint_positions` on every reset:

- Scalar joint → write directly into `data.qpos`.
- Multi-DOF joint (free / ball) → write the full `[x y z qw qx qy qz]` or `[qw qx qy qz]` vector.
- Equality-constrained passive joints (e.g. parallel-linkage gripper followers) are settled by stepping under zero gravity while pinning the configured scalar joints.

This is implemented in `MujocoBasis.reset()`. The same logic is mirrored in [`examples/view_scene.py`](../../examples/view_scene.py) so the viewer shows exactly what the runtime will see.

## Available robot XMLs

| Robot XML                                       | Description                                                |
|-------------------------------------------------|------------------------------------------------------------|
| `assets/xmls/robots/robotiq.xml`                | 6-DOF floating base + Robotiq 2F-85 gripper                |
| `assets/xmls/robots/panda_robotiq.xml`          | Franka Panda + Robotiq 2F-85                               |
| `assets/xmls/robots/p7_arm_with_xf9600.xml`     | 7-DOF P7 arm + XFG-9600 parallel-linkage gripper           |
| `assets/xmls/robots/p7_arm_with_g2p.xml`        | 7-DOF P7 arm + G2P (UMI parallel-linkage) gripper          |
| `assets/xmls/robots/airbot_play.xml`            | Airbot Play 6-DOF arm                                      |
| `assets/xmls/robots/airbot_play_with_xf9600.xml`| Airbot Play + XFG-9600                                     |
| `assets/xmls/robots/airbot_play_with_g2p.xml`   | Airbot Play + G2P gripper                                  |
| `assets/xmls/robots/airbot_g2p.xml`             | Standalone G2P gripper assembly (used by `*_with_g2p.xml`) |
| `assets/xmls/robots/xf9600_mocap.xml`           | Mocap-driven floating XFG-9600 gripper                     |
| `assets/xmls/robots/p7_arm_v3_with_umi_gripper_v3.xml` | 7-DOF P7 arm v3 + UMI gripper v3 (driven by an analytical IK solver) |
| `assets/xmls/robots/umi_gripper_v3.xml`         | Standalone UMI gripper v3 assembly (referenced by both the v3 arm and the mocap variant) |
| `assets/xmls/robots/umi_gripper_v3_mocap.xml`   | Mocap-driven floating UMI gripper v3                       |

Across all G2P variants the driven actuator joint is `eef_claw_joint`, the
finger pad geoms are `eef_left_finger_pad_upper` / `eef_right_finger_pad_upper`,
and `FingerDistanceMapper` is wired to those names. The `eef_*` prefix replaces
the legacy `xfg_*` prefix that earlier configs used; see the
`xfg_* → eef_*` notes in [Action Space](action_space.md) and
[EEF Mapper (Finger Distance)](../mujoco-backend/eef_mapper.md).

### Basis configs paired with each robot

Most tasks pick a robot via the basis config they extend:

| Basis config                          | Robot composition                                                            |
|---------------------------------------|------------------------------------------------------------------------------|
| `aao_configs/basis_xf9600.yaml`       | Generic XF9600 settings (extended by the configs below)                      |
| `aao_configs/basis_airbot_play_xf9600.yaml` | Airbot Play + XF9600                                                   |
| `aao_configs/basis_airbot_play_g2p.yaml`    | Airbot Play + G2P                                                      |
| `aao_configs/basis_p7_xf9600.yaml`    | P7 arm + XF9600                                                              |
| `aao_configs/basis_p7_g2p.yaml`       | P7 arm + G2P                                                                 |
| `aao_configs/basis_p7_xf9600_composable.yaml` | Composable variant of the P7 + XF9600 stack                          |
| `aao_configs/basis_p7_v3_umi_v3.yaml`         | P7 arm v3 + UMI gripper v3 (analytical IK via `P7V3AnalyticalIKSolver`) |
| `aao_configs/basis_mocap_eef_xf9600.yaml`     | Mocap-driven floating XF9600                                         |
| `aao_configs/basis_mocap_eef_umi_v3.yaml`     | Mocap-driven floating UMI gripper v3                                 |
| `aao_configs/basis_franka.yaml`       | Franka Panda + Robotiq                                                       |

Task configs that bind to one of the new robots include
`open_door_airbot_play_g2p`, `cup_on_coaster_airbot_p7`,
`arrange_flowers_gs_airbot_p7`, `wipe_the_table_gs_airbot_p7`, and
`press_{blue,green,pink}_button_airbot_p7`.

Tasks that bind to the **P7 v3 + UMI v3** stack:

- `cup_on_coaster_gs_airbot_p7` — joint-mode arm via `basis_p7_g2p`
- `cup_on_coaster_gs_airbot_p7_umi` — joint-mode arm via `basis_p7_v3_umi_v3`
- `open_door_p7_v3_umi_v3` — joint-mode arm via `basis_p7_v3_umi_v3`
- `pick_and_place_umi_v3` — mocap variant via `basis_mocap_eef_umi_v3`

## Iterating with the viewer

[`examples/view_scene.py`](../../examples/view_scene.py) (see [View Scene](../tools/view_scene.md)) is the fastest way to verify a `robot_paths` change: it composes the scene + robot, applies all home-pose / initial-pose / operator base overrides, and supports reload-on-edit so you don't need to restart Python after tweaking YAML or XML.

## Related

- [Action Space](action_space.md) — how operators map joints/sites to actions
- [View Scene](../tools/view_scene.md) — interactive viewer that mirrors the runtime composition
- [Custom Backend](../mujoco-backend/custom-backend.md) — backend factories that bind to the composed model
