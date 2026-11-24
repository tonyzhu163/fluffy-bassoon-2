# State lane: does putting Kinova3 proprioception in the source convention change what frozen π₀.₅ does?

Lane of the [proposal-shift program](../frozen-vla-proposal-shift-20261124-r1/PLAN.md). Written November 24 before any measurement or adapter result in this directory. Branches from harness commit `5fd6c426a8ea1ab2b75fb54d45c3d95357f66d7a`.

## Question

The frozen policy's state input is grip-site position (3), a rotation vector computed from `robot0_eef_quat` (3) and `robot0_gripper_qpos` (2). On the drawer-and-bowl layouts, how does that vector on Kinova3 differ from the one a source arm produces, and does mapping it to the source convention rescue the six Kinova3 timeouts where dropping it does not?

## Prediction

1. Measurement (archived records, CPU). Position and gripper readings on Kinova3 sit within millimetres of the source arm's at matched resets, because the benchmark aligns the grip site and Kinova3 carries the Panda gripper. The rotation differs by convention: Kinova3 reports the quaternion in the hemisphere opposite to Panda's, so the packed rotation vector has a negative X component near −π where the source data is on the positive branch. Every native Kinova3 run so far packed state with `canonicalize_quat=False`.
2. Native (not runnable here). If that branch difference is what moves proposals, `state-mapped` changes the proposal at every replan and rescues Kinova3 failures beyond the untreated repeat's discordance, and `state-dropped` rescues fewer. `state-shuffled` keeps the target convention and breaks the time alignment; it should not rescue.

## What counts against it

- Measurement: Kinova3 quaternions are already in the source hemisphere and every state channel is within the source range. Then `state-mapped` is the identity and the lane has nothing to test.
- Native: `state-mapped` equals `state-dropped` (the program's criterion), or equals `raw` within the repeat's discordance, or loses the known successes 16, 31, 34, 35, 39 or Panda 0–3.
- No numeric gate. Rescues, losses and paired intervals are reported against `raw`, `repeat` and `state-dropped`.

## Prior work checked

- [FINDINGS](../../docs/FINDINGS.md) line 117: on UR5e with frozen π₀, flipping the quaternion to the downward-gripper hemisphere moved `rot_x` from −4.0…−5.7 to +0.034 and ended a 0/96 saturation. `ur5e_motor.to_openpi_observation(canonicalize_quat=True)` holds that rule. It has not been applied on Kinova3: [native transfer](../frozen-vla-native-transfer-20261123-r1/native_rollout.py), [progress](../frozen-transfer-progress-20261124-r1/run.py), [source view](../frozen-vla-source-view-20261123-r1/motor.py) and the shared harness all pack with the raw sign.
- [SmolVLA state](../frozen-vla-factory-20261123-r1/state/RESULT.md): every source example has positive rotation-vector X (checkpoint minimum +0.353 rad); the IIWA logger was already on that branch, the UR5e logger was not.
- [SmolVLA state transfer](../frozen-vla-factory-20261123-r1/state-transfer/RESULT.md): a joint-to-pose map fit on the other arm gave 0.2770 action RMSE, dropping state 0.1001, 15 target pairs 0.0141. Forward pass only. That adapter regressed pose from joints; here the public observation already holds the pose, so the map is a convention change plus measured fixed offsets.
- [Factory](../frozen-vla-factory-20261123-r1/RESULT.md): proprioception moves frozen proposals. No native state arm has been run.

## Protocol

1. `measure.py` (CPU, archived records only) → `measurement.json`.
   - Kinova3 public state at five times on layouts 14 and 15: [source-view frames](../frozen-vla-source-view-20261123-r1/evidence/cpu01).
   - Kinova3 and Panda grip-site position and rotation along native rollouts of `libero_goal/3` layouts 14 and 15 (and the other archived tasks): [pose replays](../frozen-transfer-pose-20261124-r1/evidence/replay).
   - Any other archived record that holds source-arm quaternions or gripper readings; list what was found and what is missing.
   - Report reset position difference, geodesic rotation difference, the rotation vector under the raw and source sign rules, and gripper reading range.
2. `adapter.py` exposes `ADAPTERS`, all through the `observation` hook only (public observation in, public observation out; prompt, noise, choice and commands untouched):
   - `state-mapped`: quaternion sign moved to the source hemisphere; position and gripper corrected only by fixed offsets the measurement supports, using the 15-control calibration frames where a target-side quantity is needed.
   - `state-dropped`: the same constant state every replan. Source mean rather than zero: a zero vector puts the grip site 1 m below the table and the rotation on neither branch, which tests an out-of-range state and not the absence of state. The source mean normalizes to the centre of what the policy was trained on.
   - `state-shuffled`: the episode's own target-convention states, drawn from other replans by a seeded permutation of the history. Images, prompt and noise stay current.
3. `test_adapter.py` on the harness CPU fixture: only the state digest changes at a replan, images and commands are not written, mapped is an equivalent rotation, dropped is constant, shuffled returns states seen earlier in the episode, panel discovery finds the three names.
4. `run.sh`: wraps the harness launch for the three adapters on one RTX 4090. Not run here.

## Evidence kinds

Archived-record measurement and CPU fixture only. No policy forward pass, no native reward. Nothing here is task success.
