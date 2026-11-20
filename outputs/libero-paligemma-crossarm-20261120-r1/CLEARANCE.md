# Sawyer cabinet clearance on BASE Goal 5

On Sawyer pose 0, the native trace reported contact between the elbow geometry `robot0_right_l4_2` and `wooden_cabinet_1_g9` at controls 40, 65 and 84. The finger pads touched each other. The tool center stalled at 0.937 m and the lowest finger point at 0.927 m, 27 mm above the 0.900 m table. The elbow contact gives a reason to change the approach orientation. Pad-to-pad contact alone gives no evidence of a grip on the plate.

Construct `ClearanceExpert(session, scene)` once after the normal Sawyer reset. It uses the in-process native simulator to check candidate tool tilts and yaws at descent and push waypoints using inverse kinematics on scratch MuJoCo data. Its `.scores` list holds the candidate diagnostics, including elbow clearance. Its `.chosen` field holds the winning score dictionary or `None` if all candidates fail. Inspect both before issuing a control. `.points` holds the plate destination in world XY and `.path` holds the selected grip-site waypoints on a feasible reset and `.phase` identifies the current execution stage. Each call to `action(scene)` uses the latest scene to produce one seven-component native OSC/gripper command.

The scratch scene copies the settled joint and mocap positions and models closed finger slides. For each candidate, the planner transforms the compiled finger vertices to find the lowest contact patch and offsets the grip site so that patch descends over the plate center. Positive tilt points the fingers toward the negative-Y cabinet. The ten candidates include a vertical control, a reverse tilt, and several yaw angles. The path runs from the reset through a raised approach, two descent heights, center seating, inner-rim engagement, an intermediate push and the goal-side rim at the destination.

Bounded Jacobian IK starts each candidate at the same reset arm configuration and warm-starts each waypoint from its predecessor. The planner samples robot–cabinet contacts on each connecting joint-space segment at intervals of at most 0.15 rad per joint. It rejects a candidate on any sampled cabinet contact, nonpositive elbow clearance, position error above 4 mm or angular error above 0.045 rad at any waypoint. Among feasible candidates, it prefers the largest minimum link-4 clearance, then the smallest normalized worst pose error. Reported elbow distances saturate at 0.10 m. Geometry is evaluated against the cabinet while the plate stays at its reset pose during planning. Actual finger contact and plate transport depend on live dynamics.

## Collect one Linux attempt

Use the qualified Linux environment and run from this run's code directory. Invoke the simulator interpreter directly so the expert can access the live MuJoCo model. Set `OUT` to a fresh, existing attempt directory on shared storage. The example uses released pose 0 and collector seed 20261120. [Linux setup](DEMOS.md) · [Session](sim.py) · [Tape writer](demos.py)

```bash
"${CROSSARM_SIM_PYTHON:?}" -Bu - "${OUT:?}" <<'PY'
from pathlib import Path
import sys

from clearance_expert import ClearanceExpert
from demos import save_tape
from sim import Session

body, pose, seed = "Sawyer", 0, 20261120
path = Path(sys.argv[1]) / body / f"{pose}.npz"
with Session(body, task_id=5) as session:
    obs = session.reset(pose, seed)
    scene = session.expert_scene()
    expert = ClearanceExpert(session, scene)
    # The constructor prints the ranked candidates and selection as JSON.
    import json
    (path.parent.parent / "plan.json").write_text(json.dumps(
        dict(scores=expert.scores, chosen=expert.chosen), indent=2))
    if expert.chosen is None:
        raise SystemExit("No feasible path for this reset")

    prompt = obs["prompt"]
    frames, actions, motion, rewards = [], [], [], []
    trace = []
    for tick in range(session.reset_info["max_steps"]):
        action = expert.action(scene)
        next_obs, measured, reward, done, success = session.step(action)
        scene = session.expert_scene()
        trace.append(dict(step=tick + 1, phase=expert.phase, reward=reward,
                          elbow_pair_gap_m=expert.measured_elbow_gap(),
                          tcp=scene["tcp"], finger_low=scene["finger_low"],
                          plate=scene["plate"], plate_contact=scene["contact"]))
        frames.append(obs)
        actions.append(action)
        motion.append(measured)
        rewards.append(reward)
        if done or success or tick + 1 == session.reset_info["max_steps"]:
            (path.parent.parent / "trace.json").write_text(json.dumps(trace))
        if done and success and reward > 0:
            save_tape(path, frames, actions, motion, rewards, next_obs["state"],
                      prompt, body, pose, seed)
            print("saved", path)
            break
        if done or success:
            raise SystemExit(f"Rejected at {expert.phase}: {done=} {success=} {reward=}")
        obs = next_obs
    else:
        raise SystemExit(f"Horizon reached in {expert.phase} without native success")
PY
```

The recorded collision and stall are measured outcomes of the earlier rollout. A winning score predicts clearance only at the sampled poses. Whether the live controller clears the cabinet between them, makes finger–plate contact and pushes the plate into the goal remains unverified. The native terminal success flag and positive reward establish task completion.
