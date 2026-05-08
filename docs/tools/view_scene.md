# View Scene

Launch the interactive MuJoCo viewer on a composed scene + robot, applying every YAML-defined home-pose and pose override before the first frame is shown.

Since scene XMLs no longer embed their robot include or `<key name="home">`, opening `assets/xmls/scenes/<task>/demo.xml` directly in the MuJoCo simulator shows just the empty scene. This script reads a Hydra task config, composes the scene with the robot(s) declared under `env.robot_paths`, applies `env.initial_joint_positions` plus `task.initial_pose` and `task_operators.<name>.initial_state.base_pose`, and hands the model to `mujoco.viewer.launch`.

**Script:** [examples/view_scene.py](../examples/view_scene.py)

## Usage

```bash
python examples/view_scene.py --config-name pick_and_place
python examples/view_scene.py --config-name open_door_airbot_play_back_gs
python examples/view_scene.py --config-name open_door_p7_ik
python examples/view_scene.py --debug --config-name open_door_p7_ik
```

The default config is `pick_and_place`. Any Hydra override can be appended after `--`:

```bash
python examples/view_scene.py --config-name open_door_p7_ik -- env.initial_joint_positions.joint1=0.5
```

## What it composes

For the chosen config, the script reads four override surfaces and applies them on top of the bare scene XML:

| Source                                            | What it sets                                                                  |
|---------------------------------------------------|-------------------------------------------------------------------------------|
| `env.model_path`                                  | Scene XML (no robot, no keyframe)                                             |
| `env.robot_paths`                                 | Robot XML(s) injected as `<include>` siblings under `<mujoco>` at load time   |
| `env.initial_joint_positions`                     | Per-joint home pose (mirrors `MujocoBasis.reset()`)                           |
| `task.initial_pose`                               | Per-body pose overrides (freejoint qpos for movable bodies, or `body_pos/quat` for static bodies) |
| `task_operators.<name>.initial_state.base_pose`   | Relocates each operator's `root_body` so the arm sits at the right world pose |

Equality-constrained passive joints (e.g. parallel-linkage gripper followers) are settled by stepping under zero gravity while pinning the configured scalar joints, matching the runtime backend reset.

Mocap bodies welded to a freejoint are synced onto their target pose so the arm doesn't snap on the first viewer step.

## Gaussian Splatting mode

When the chosen config carries an `env.gaussian_render` section with at least one body
gaussian or a background PLY (e.g. `open_door_airbot_play_gs`,
`stack_color_blocks_gs`), the script switches to a **passive** MuJoCo viewer
and opens a second OpenCV window titled
`GS view (synced with MuJoCo viewer)`.

The GS window re-renders the scene from the same free-camera pose as the
MuJoCo viewer every step, so orbit / pan / zoom in the MuJoCo viewer drives
the GS preview live. The window defaults to 640×480; use the OpenCV window
controls (or close the MuJoCo viewer) to exit.

Detection is content-based: the script looks for `env.gaussian_render` with
either `body_gaussians` or `background_ply` populated, so it works whether
the task uses the GS env target directly or composes GS into a non-GS env.

Multi-background configs (`background_ply` is list / glob / parts dict) are
previewed with **only the first resolved PLY** — the viewer is for verifying
geometry alignment, not for sweeping backgrounds.

## Reload workflow

Pick up edits without restarting Python:

- **GS mode** — press `R` in either window, or click the **Reload (R)** button
  in the top-right of the GS window. Both trigger a full reload.
- **Non-GS mode** — use the reload button on the MuJoCo viewer panel.

Reload re-reads:

- **YAML edits** to `env.initial_joint_positions`, `task.initial_pose`,
  `task_operators.<name>.initial_state.base_pose`, `env.robot_paths`, or
  `env.gaussian_render.*` — re-composed via Hydra from disk.
- **XML edits** to the scene or any robot XML — re-read by
  `auto_atom.utils.scene_loader.load_scene`.
- **PLY edits** in GS mode — body / background gaussians are reloaded from
  disk and a new `GSRendererMuJoCo` is built. The GS window shows
  `Loading Gaussian renderer...` and `Warming up GS render...` status frames
  during the rebuild; the first GS frame after a reload can take a few
  seconds while gaussian PLYs upload to GPU.

In GS mode the MuJoCo viewer window is closed and reopened around each
reload (because the underlying `MjModel` is replaced); this is normal.

This makes `view_scene.py` the fastest way to iterate on home pose, robot
injection, scene geometry, and Gaussian alignment side by side.

## Console output

On startup and on every reload the script prints:

```
[info] scene  : .../scenes/open_door/demo.xml
[info] robots : ['.../robots/p7_arm_with_xf9600.xml']
[info] home   : 9 joint override(s), 0 body pose(s), 1 operator base(s)
[info] model  : nq=23 nv=22 nu=8 nbody=14 ngeom=37  (robots=[...], ijp=9, body_pose=0, op_base=1)
```

Use these counters to confirm that the expected robot was injected and that all overrides were honoured.

Pass `--debug` before Hydra arguments to preflight-build the model and print
full loader tracebacks to the terminal. This is useful for errors that MuJoCo's
viewer would otherwise show only inside the GUI.

## Related

- [Scene Composition](../task-configuration/scene_composition.md) — how `env.robot_paths` injection works
- [Tune Randomization Extremes](tune_randomization_extremes.md) — same override surfaces, with randomization stepping
