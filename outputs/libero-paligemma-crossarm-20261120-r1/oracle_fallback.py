"""Qualify the released Goal 5 action tape and track its measured motion on source arms."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from demos import json_write, save_tape
from sim import Session


ARCHIVE_SHA = "a288a781a2d76368530d0d95e0e0a9a9895937a365f2909c16d7b57e28eeed90"
BDDL_SHA = "c2db20071f58fd43fe029e5bf9176f3c7371ee62cadc311f2b8682c089174c0d"


def read_demo(path, number):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_SHA:
        raise ValueError("Expected the pinned libero_goal/task_5/BASE/episodes.hdf5")
    with h5py.File(path, "r") as archive:
        group = archive[f"data/demo_{number}"]
        source = json.loads(group.attrs["source_metadata"])
        if not (group.attrs["success"] and group["rewards"][-1] > 0 and group["dones"][-1]):
            raise ValueError("Reference lacks a successful native ending")
        return (source, group["actions"][:], group["obs/robot0_eef_pos"][:],
                group["obs/robot0_eef_quat_xyzw"][:])


def panda_reference(path, number):
    """Replay once at the recorded Panda reset. Geometry is absent from the release."""
    src, actions, recorded_pos, recorded_quat = read_demo(path, number)
    pose, seed = int(src["init_state_id"]), int(src["seed"])
    with Session("Panda", task_id=5) as session:
        obs = session.reset(pose, seed)
        lineage = session.reset_info["environment_reproducibility"]
        if lineage["bddl_sha256"] != BDDL_SHA or obs["prompt"] != src["instruction"]:
            raise ValueError("Released tape and live task differ")
        states, plates = [obs["state"]], [session.expert_scene()["plate"]]
        for i, action in enumerate(actions):
            obs, _, reward, done, success = session.step(action)
            states.append(obs["state"])
            plates.append(session.expert_scene()["plate"])
            if done:
                break
        if not (success and reward > 0 and i + 1 == len(actions)):
            raise RuntimeError(f"Released Panda replay ended at {i + 1}, reward={reward}, success={success}")
    states = np.asarray(states)
    # Released observations follow their corresponding command. Split quaternion
    # signs represent the same rotation, so compare physical angular distances.
    xyz_err = np.linalg.norm(states[1:, :3] - recorded_pos, axis=1)
    rot_err = (Rotation.from_quat(states[1:, 3:7]) *
               Rotation.from_quat(recorded_quat).inv()).magnitude()
    evidence = dict(demo=number, pose=pose, seed=seed, steps=len(actions), native_reward=float(reward),
                    native_success=bool(success), replay_max_xyz_error_m=float(xyz_err.max()),
                    replay_max_angle_error_rad=float(rot_err.max()), reset=lineage)
    return dict(states=states, plates=np.asarray(plates), actions=actions, evidence=evidence)


class TrajectoryExpert:
    """Track demonstrated OSC goals using the live tool and plate on every call."""

    def __init__(self, reference, scene):
        self.ref = reference
        self.i = 0
        self.wait_controls = 0
        self.initial_offset = np.subtract(scene["plate"][:2], reference["plates"][0, :2])
        travel = reference["plates"][-1, :2] - reference["plates"][0, :2]
        self.travel = travel
        self.travel_sq = max(float(travel @ travel), 1e-8)
        self.trace = []

    def action(self, obs, scene):
        i = self.i
        state = obs["state"]
        source = self.ref["states"][i]
        plate = self.ref["plates"][i, :2]
        progress = np.clip((plate - self.ref["plates"][0, :2]) @ self.travel / self.travel_sq, 0, 1)
        # Shift the approach to the target plate; keep the demonstrated terminal
        # destination fixed. Use current plate lag to preserve the rim contact.
        offset = (1 - progress) * self.initial_offset
        slip = np.asarray(scene["plate"][:2]) - (plate + offset)
        target = source[:3].copy()
        target[:2] += offset + np.clip(slip, -0.015, 0.015)
        dp = target - state[:3]
        rotation = Rotation.from_quat(source[3:7]) * Rotation.from_quat(state[3:7]).inv()
        dr = rotation.as_rotvec()
        pos_error, angle_error = np.linalg.norm(dp), np.linalg.norm(dr)
        # Impedance control has a residual tracking error near the cabinet.
        # Keep moving along the demonstrated approach despite a small static error.
        self.wait_controls += 1
        ready = ((pos_error <= 0.030 and angle_error <= 0.17) or
                 (self.wait_controls >= 4 and pos_error <= 0.06 and angle_error <= 0.30))
        a = self.ref["actions"][i].copy()
        if ready:
            # Compose the released world-axis orientation increment with the
            # measured frame error. Translation and gripper retain feedforward.
            a[:3] += 2.0 * dp / 0.05
            self.wait_controls = 0
            a[3:6] = (Rotation.from_rotvec(0.5 * a[3:6]) * rotation).as_rotvec() / 0.5
            self.i = min(i + 1, len(self.ref["actions"]) - 1)
        else:
            # A slow body catches the pre-action waypoint before time advances.
            a[:3] = 2.0 * dp / 0.05
            a[3:6] = dr / 0.5
            if i:
                a[6] = self.ref["actions"][i - 1, 6]
        self.trace.append(dict(index=i, advanced=bool(ready), xyz_error_m=float(pos_error),
                               angle_error_rad=float(angle_error), contact=bool(scene["contact"]),
                               plate=scene["plate"], plate_lag_xy_m=slip.tolist()))
        return np.clip(a, -1, 1).astype(np.float32)


def collect(reference, output, body, pose, seed, raw=False, force_tracking=False):
    frames, actions, motions, rewards = [], [], [], []
    with Session(body, task_id=5) as session:
        obs = session.reset(pose, seed)
        scene = session.expert_scene()
        tracker = TrajectoryExpert(reference, scene)
        lineage = session.reset_info["environment_reproducibility"]
        donor = reference["evidence"]["reset"]
        replay = raw or (not force_tracking and body == "Panda" and pose == donor["init_state_id"] and
                  lineage["initial_state_fingerprint"] == donor["initial_state_fingerprint"])
        route = "released_actions" if replay else "closed_loop_tracking"
        for _ in range(session.reset_info["max_steps"]):
            frames.append(obs)
            if replay:
                i = len(actions)
                a = reference["actions"][i]
                tracker.trace.append(dict(index=i, plate=scene["plate"], contact=scene["contact"]))
            else:
                a = tracker.action(obs, scene)
            obs, motion, reward, done, success = session.step(a)
            actions.append(a)
            motions.append(motion)
            rewards.append(reward)
            if done or (replay and len(actions) == len(reference["actions"])):
                break
            scene = session.expert_scene()
        accepted = bool(done and success and reward > 0)
        if accepted:
            save_tape(output / body / f"{pose}.npz", frames, actions, motions, rewards,
                      obs["state"], obs["prompt"], body, pose, seed)
        row = dict(body=body, pose=pose, seed=seed, steps=len(actions), accepted=accepted,
                   native_success=bool(success), terminal_reward=float(reward), done=bool(done),
                   reset=session.reset_info, route=route, reference_demo=reference["evidence"]["demo"])
        if accepted:
            row["file"] = f"{body}/{pose}.npz"
        json_write(output / f"{body}-{pose}-{route}-trace.json",
                   dict(attempt=row, trace=tracker.trace, actions=np.asarray(actions).tolist(), rewards=rewards))
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--demo", type=int, choices=range(30), default=22)
    parser.add_argument("--matching", action="store_true", help="Use a verified published Panda example of the same training pose when available")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--bodies", nargs="+", choices=("Panda", "Sawyer"), default=["Panda", "Sawyer"])
    parser.add_argument("--seed", type=int, required=True, help="Use the parent bank's seed base plus pose")
    args = parser.parse_args()
    if not all(0 <= pose < 50 and 0 <= args.seed + pose < 2**32 for pose in args.poses):
        parser.error("Use released rows 0..49 and valid uint32 target seeds")
    if len(set(args.poses)) != len(args.poses) or len(set(args.bodies)) != len(args.bodies):
        parser.error("Duplicate target would overwrite a qualification tape")
    args.output.mkdir(parents=True, exist_ok=False)
    index = dict(suite="libero_goal", task_id=5, case="BASE", archive_sha256=ARCHIVE_SHA,
                 demo=args.demo, poses=args.poses, seed_base=args.seed, bodies=args.bodies, attempts=[])
    if args.matching and any(p not in (*range(16), 24, 25, 26, 27) for p in args.poses):
        parser.error("Production reference selection is restricted to registered source demonstration poses")
    matching = {}
    if args.matching:
        with h5py.File(args.archive, "r") as archive:
            for num in range(len(archive["data"])):
                metadata = json.loads(archive[f"data/demo_{num}"].attrs["source_metadata"])
                key = int(metadata["init_state_id"])
                if key in args.poses:
                    matching.setdefault(key, num)
    try:
        ref = panda_reference(args.archive, args.demo)
    except Exception as exc:
        index["reference_error"] = str(exc)
        json_write(args.output / "index.json", index)
        raise
    index["reference"] = ref["evidence"]
    json_write(args.output / "index.json", index)
    references = {args.demo: ref}
    index["references"] = {args.demo: ref["evidence"]}
    for pose in args.poses:
        num = matching.get(pose, args.demo)
        if num not in references:
            try:
                references[num] = panda_reference(args.archive, num)
                index["references"][num] = references[num]["evidence"]
            except RuntimeError as exc:
                index.setdefault("reference_misses", {})[num] = str(exc)
                references[num] = ref
        donor = references[num]
        for body in args.bodies:
            # A matched demonstration can provide commands directly. If those
            # commands miss the goal, use measured feedback on a fresh reset.
            try_raw = args.matching and pose == donor["evidence"]["pose"]
            for raw in ((True, False) if try_raw else (False,)):
                trial_seed = donor["evidence"]["seed"] if try_raw else args.seed + pose
                row = collect(donor, args.output, body, pose, trial_seed, raw=raw,
                              force_tracking=args.matching and not raw)
                index["attempts"].append(row)
                json_write(args.output / "index.json", index)
                print(f"{body}/{pose} {row['route']} accepted={row['accepted']} controls={row['steps']}", flush=True)
                if row["accepted"]:
                    break
    return int(sum(r["accepted"] for r in index["attempts"]) != len(args.poses) * len(args.bodies))


if __name__ == "__main__":
    raise SystemExit(main())
