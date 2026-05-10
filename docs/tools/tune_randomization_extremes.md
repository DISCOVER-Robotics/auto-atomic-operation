# Tune Randomization Extremes

Interactive tkinter inspector for verifying that configured randomization ranges keep objects and operators within a reasonable workspace. Opens the MuJoCo viewer alongside a control panel that lets you cycle through extreme randomization cases.

The inspector also reloads the YAML values that define the **default** state:
`task.initial_pose` and `task_operators.<name>.initial_state`. This makes it
useful for iterating on both the nominal operator/object placement and the
randomization ranges around that placement.

**Script:** [examples/tune_randomization_extremes.py](../examples/tune_randomization_extremes.py)

## Usage

```bash
python examples/tune_randomization_extremes.py
python examples/tune_randomization_extremes.py --config-name cup_on_coaster
python examples/tune_randomization_extremes.py --config-name arrange_flowers
```

The default config is `pick_and_place`. The script extracts the `task.randomization` section from the YAML config and builds a set of extreme cases.

## Control panel

The tkinter panel provides:

- **Randomization summary** -- shows all randomized targets with their axis ranges
- **Extreme case selector** -- dropdown to pick a case, with Prev/Next buttons
- **Apply / Reset Default** -- apply the selected case or return to the nominal pose
- **Random Sample** -- draw a fresh random sample uniformly from each configured range
- **Reload Randomization** -- re-read the YAML config from disk, apply updated defaults from `task.initial_pose` / `task_operators.<name>.initial_state`, then rebuild the randomization cases
- **Full Reload** -- rebuild the entire scene and backend from the current config
- **Current Poses** -- live display of each target's position, quaternion, and RPY

## Generated extreme cases

The inspector automatically generates these cases from the config:

| Case | Description |
|---|---|
| `default` | No randomization offset; all targets at nominal pose |
| `all-min` | Every randomized axis at its minimum value simultaneously |
| `all-max` | Every randomized axis at its maximum value simultaneously |
| `<target> <axis>=min` | Single target, single axis at minimum; everything else at default |
| `<target> <axis>=max` | Single target, single axis at maximum; everything else at default |

This covers all corners of the randomization space. If any case pushes an object outside the workspace, off the table, or into collision, the range should be tightened.

`default` is the current YAML-defined baseline after initial pose/state
overrides have been applied. Operators with `initial_state` are shown even if
they do not have a randomization range, so you can edit the YAML directly and
press **Reload Randomization** to inspect the updated nominal base, EEF pose,
and gripper control.

## Workflow

1. Configure `task.randomization` in your YAML config (see [Pose Randomization](../task-configuration/randomization.md))
2. Run this inspector to visually verify the extremes
3. Use `all-min` and `all-max` to check the worst-case simultaneous offsets
4. Step through per-axis cases to identify which specific axis causes problems
5. Use `Random Sample` to spot-check typical randomized states
6. Edit your YAML `initial_pose`, operator `initial_state`, or randomization ranges and press `Reload Randomization` to iterate without restarting

Use **Full Reload** when you add new object/operator names that were not
registered when the scene was first opened.

## Joint-limit proximity warnings

The inspector force-enables `UnifiedMujocoEnv.set_joint_limit_warning_enabled(True)`
on startup, even though the env's default is off. Surfacing borderline IK
solutions is the entire point of this tool, so the env will log a `WARNING`
whenever an IK-solved joint angle lands within ~0.05 rad (≈ 2.9°) of a hard
joint limit. The log line includes the operator, joint name, current angle
(rad + deg), distance to limit, and the limit value.

Use these warnings as a signal that:

- The randomization range is letting the arm reach poses that just barely
  satisfy IK and would behave poorly in production — tighten the per-axis
  range until the warnings stop, or
- The default `initial_state.base_pose` / `initial_state.eef_pose` itself is
  already close to a limit and any randomization on top will push past it,
  so the home pose should be re-tuned.

Each (joint, side) only warns once per entry into the danger band; the warning
re-arms after the joint moves at least 0.10 rad back from the limit, so the
log will not flap when a solution sits right at the boundary.

See [IK Control § 关节限位接近告警](../ik-motion-control/ik_control.md) for the
full warning-system contract.
