# Action Space

The framework provides two levels of control for driving the robot. Which one
to use depends on the action representation your policy outputs.

## Two Control Levels

| Level | API | What it commands | When to use |
|-------|-----|------------------|-------------|
| **Ctrl** | `env.step(action)` | Raw actuator targets (`data.ctrl`) | Policy outputs joint angles + gripper |
| **Pose** | `env.step_operator_toward_target(op, pos_b, quat_b)` | EEF target pose in base frame | Policy outputs Cartesian poses |

Both levels advance the physics simulation by one control step.

---

## Ctrl Mode — `env.step(action)`

Writes the action vector directly into MuJoCo's `data.ctrl`, clips to
`actuator_ctrlrange`, then steps the simulation.

```python
# Simplified from mujoco_env.py
data.ctrl[:n] = clip(action[:n], ctrlrange)
mj_step(model, data)
```

Each dimension corresponds to one `<actuator>` in the MuJoCo XML, in
declaration order. All actuators in this project use `<position>`, so each
dimension is a **target joint angle** in radians.

### Mocap Robot (basis_mocap_eef / Robotiq 2F-85)

| ctrl index | actuator name      | meaning                                             |
|:----------:|--------------------|-----------------------------------------------------|
| 0          | `fingers_actuator` | Gripper open/close angle, `ctrlrange=[0, 0.82]` rad |

**action_dim = 1** — gripper only.

Arm motion in mocap mode is **not** controlled through `data.ctrl`. It is
driven by writing `data.mocap_pos` / `data.mocap_quat` directly (see Pose
Mode below). Therefore `env.step(action)` alone **cannot move the arm** for
a mocap robot.

### Joint-Mode Robot (basis_p7_xf9600 / Panda + XFG-9600)

| ctrl index | actuator name    | meaning                        |
|:----------:|------------------|--------------------------------|
| 0          | `joint1`         | Arm joint 1 target angle (rad) |
| 1          | `joint2`         | Arm joint 2 target angle (rad) |
| 2          | `joint3`         | Arm joint 3 target angle (rad) |
| 3          | `joint4`         | Arm joint 4 target angle (rad) |
| 4          | `joint5`         | Arm joint 5 target angle (rad) |
| 5          | `joint6`         | Arm joint 6 target angle (rad) |
| 6          | `joint7`         | Arm joint 7 target angle (rad) |
| 7          | `xfg_claw_joint` | Gripper open/close angle (rad) |

**action_dim = 8** — 7 arm joints + 1 gripper.

`env.step(action)` fully controls both arm and gripper for joint-mode robots.

### Batched Step

`BatchedUnifiedMujocoEnv.step()` expects shape `(batch_size, action_dim)`:

```python
env.step(action, env_mask=mask)
# action: np.ndarray, shape (B, action_dim)
# env_mask: optional np.ndarray[bool], shape (B,)
```

Each row `action[i]` is applied to replica `envs[i]`. Replicas where
`env_mask[i] == False` are skipped.

---

## Pose Mode — `env.step_operator_toward_target()`

Commands a target EEF pose (position + quaternion) in the **operator base
frame**. The env handles the low-level conversion internally:

```python
env.step_operator_toward_target(
    op_name,           # operator name, e.g. "arm"
    target_pos_b,      # np.ndarray, shape (B, 3), position in base frame
    target_quat_b,     # np.ndarray, shape (B, 4), quaternion xyzw in base frame
    env_mask=mask,     # optional bool array, shape (B,)
)
```

### Internal behavior per robot type

**Joint-mode robot** — solves IK each step, writes resulting joint angles to
`data.ctrl[arm_aidx]`, then calls `env.step()`:

```
target EEF pose (base frame)
  → IK solver → joint angles
  → data.ctrl[arm_aidx] = joint_angles
  → mj_step()
```

Two IK strategies are available (configured per operator):
- `per_step_ik` — re-solves every step from current `qpos`
- `solve_once_interpolate` — solves once, interpolates joint trajectory

**Mocap robot** — converts EEF pose to mocap body pose in world frame, writes
to `data.mocap_pos` / `data.mocap_quat`, then calls `env.update()`:

```
target EEF pose (base frame)
  → base_to_world transform
  → subtract tool offset → mocap body pose (world frame)
  → data.mocap_pos[mocap_id] = pos
  → data.mocap_quat[mocap_id] = quat
  → mj_step()
```

### Gripper control in pose mode

`step_operator_toward_target` only controls the arm. Gripper must be set
separately by writing to `data.ctrl[eef_aidx]` (or using the recorded ctrl
values from the demo).

---

## How Actuator Indices Are Resolved

The YAML config declares actuator names per operator:

```yaml
env:
  operators:
    - name: arm
      arm_actuators: [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
      eef_actuators: [xfg_claw_joint]
```

For a mocap robot, `arm_actuators` is empty:

```yaml
env:
  operators:
    - name: arm
      arm_actuators: []
      eef_actuators: [fingers_actuator]
```

At startup, `_resolve_actuator_indices()` maps each name to its index in
`model.actuator` via `mj_name2id`. The resulting index arrays
(`_op_arm_aidx`, `_op_eef_aidx`) are used throughout the framework to
read/write specific slices of `data.ctrl` and `data.qpos`.

---

## Recorded Demo Data

`record_demo.py` captures **both** control levels at every step:

| NPZ key | content | shape |
|---------|---------|-------|
| `actions` | `data.ctrl[:nu]` — raw actuator values | `(T, B, nu)` |
| `action/{op}/pose/position` | EEF target position in base frame | `(T, 3)` |
| `action/{op}/pose/orientation` | EEF target quaternion (xyzw) in base frame | `(T, 4)` |

The `actions` key stores whatever was in `data.ctrl` at each step, which
includes arm joints (if joint-mode) and gripper. For mocap robots, arm joints
are absent from `data.ctrl`, so `actions` only contains the gripper.

The pose keys (`action/{op}/pose/*`) are captured from the operator's
internal `target_pos_in_base` / `target_quat_in_base` state, and are
available for **both** robot types.

---

## Replay via Policy

`replay_demo.py` supports two replay modes that correspond to the two
control levels:

### `mode="ctrl"` — direct actuator replay

```python
actions = np.load("demo.npz")["actions"]  # (T, B, nu)
for i in range(len(actions)):
    env.step(actions[i])
```

- **Joint robot**: fully reproduces arm + gripper motion
- **Mocap robot**: only reproduces gripper; arm stays at initial pose

### `mode="pose"` — pose-level replay

```python
pose_trace = load_pose_trace(demo_data)  # list of {op: {position, orientation}}
for i in range(len(pose_trace)):
    _apply_pose_targets(backend, pose_trace[i], actions[i])
```

Internally, `_apply_pose_targets()` does per-operator:
- **Joint robot**: IK solve from recorded EEF pose → write `ctrl[arm_aidx]`
- **Mocap robot**: convert EEF pose → write `mocap_pos/quat`
- Gripper values are taken from the `actions` array (ctrl)

This mode **works for both robot types** and is the recommended approach when
you want a unified replay pipeline.

### Choosing a mode for your policy

| Policy output | Robot type | Recommended mode |
|--------------|------------|-----------------|
| Joint angles + gripper | Joint-mode | ctrl |
| EEF pose + gripper | Joint-mode | pose (IK solved internally) |
| EEF pose + gripper | Mocap | pose (only viable option for arm) |
