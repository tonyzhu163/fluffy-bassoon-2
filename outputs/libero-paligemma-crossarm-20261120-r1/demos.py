"""Collect successful on-body geometric demonstrations for BASE libero_goal/5."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from sim import PRIOR, Session
from rim_expert import RimExpert


COLLECTION_BODIES = ("Panda", "Sawyer", "UR5e")
ROUTES = ("direct", "around_cheese")


class PushExpert:
    def __init__(self, scene, route):
        # Aim inside the far/right portion of the named region. This gives the
        # plate clearance from the cheese between its start and the region.
        goal = np.array(scene["goal"][:2]) + 0.55 * np.array(scene["goal_half"][:2])
        self.points = [goal]
        if route == "around_cheese":
            obstacle = np.array(scene["obstacle_max"][:2])
            corner = np.maximum(obstacle + scene["plate_radius"] + 0.008, [scene["plate"][0], goal[1]])
            self.points.insert(0, corner)
        self.phase = "lift"
        self.reseats = 0
        self.ticks = 0
        self.best = float("inf")
        self.stalled = 0

    def phase_to(self, phase):
        self.phase = phase
        self.ticks = 0
        self.best = float("inf")
        self.stalled = 0

    def action(self, s):
        plate = np.asarray(s["plate"][:2])
        tcp = np.asarray(s["tcp"])
        r = s["plate_radius"]
        delta = self.points[0] - plate
        distance = np.linalg.norm(delta)
        if len(self.points) > 1 and distance < 0.015:
            self.points.pop(0)
            self.phase_to("lift")
            delta = self.points[0] - plate
            distance = np.linalg.norm(delta)
        direction = delta / max(distance, 1e-8)
        rear = plate - (r + 0.021) * direction
        clearance = s["finger_low"] - s["table_z"]
        # Place the lowest collision point within the vertical span of the rim.
        working_height = float(np.clip(0.24 * (s["rim_z"] - s["table_z"]), 0.0015, 0.004))
        hover = 0.18
        offset = tcp[:2] - plate
        side_error = np.linalg.norm(offset - (offset @ direction) * direction)
        if self.phase == "lift" and clearance >= hover - 0.012:
            self.phase_to("align")
        if self.phase == "align" and np.linalg.norm(rear - tcp[:2]) < 0.012:
            self.phase_to("lower")
        if (self.phase == "lower" and clearance <= working_height + 0.003 and
                np.linalg.norm(rear - tcp[:2]) < 0.016):
            self.phase_to("push")
        if self.phase == "push":
            if distance < self.best - 0.003:
                self.best, self.stalled = distance, 0
            else:
                self.stalled += 1
            if offset @ direction > 0 or side_error > 0.7 * r or self.stalled >= 32:
                self.reseats += 1
                if self.reseats > 2:
                    raise Stalled("Plate stalled or finger slipped after two reseats")
                self.phase_to("lift")
        self.ticks += 1
        if self.phase != "push" and self.ticks > 125:
            raise Stalled(f"Failed to reach {self.phase} waypoint in 125 controls")
        xy = rear
        z = hover
        if self.phase == "lift":
            xy = tcp[:2]
        elif self.phase == "lower":
            z = working_height
        elif self.phase == "push":
            # Recompute the trailing edge every control. The positive advance into
            # the rim maintains pressure as the object translates or slips.
            xy = plate - (r - 0.040) * direction
            z = working_height
        dp = np.r_[xy - tcp[:2], z - clearance]
        # Shortest tilt of the tool's local +Z towards the tabletop. Yaw is free.
        tilt = np.cross(s["tool_z"], [0.0, 0.0, -1.0])
        command = np.r_[np.clip(1.5 * dp / 0.05, -1, 1),
                        np.clip(tilt / 0.5, -0.5, 0.5), 1.0]
        return command.astype(np.float32)


class Stalled(RuntimeError):
    pass


def json_write(path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_tape(path, frames, actions, motion, rewards, terminal_state, prompt, body, pose, seed):
    arrays = {key: np.stack([obs[key] for obs in frames]) for key in ("rgb", "wrist", "state")}
    arrays.update(actions=np.stack(actions), motion=np.stack(motion), rewards=np.asarray(rewards, np.float32),
                  terminal_state=terminal_state, prompt=np.asarray(prompt), body=np.asarray(body),
                  pose=pose, seed=seed, native_success=True)
    n = len(actions)
    for key in ("rgb", "wrist"):
        if arrays[key].shape != (n, 256, 256, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f"Invalid {key} tape")
    for key, width in (("state", 8), ("actions", 7), ("motion", 7)):
        a = arrays[key]
        if a.shape != (n, width) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f"Invalid {key} tape")
    if not np.allclose(np.linalg.norm(arrays["state"][:, 3:7], axis=1), 1.0, atol=1e-5):
        raise ValueError("Non-unit hand quaternion")
    if np.any(abs(arrays["actions"]) > 1) or any(rewards[:-1]) or rewards[-1] <= 0 or not prompt.strip():
        raise ValueError("Invalid successful native control tape")
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".part")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def collect_attempt(root, body, pose, seed, route, limit):
    record = dict(body=body, pose=pose, seed=seed, route=route, native_success=False, accepted=False)
    frames, actions, motion, rewards, trace = [], [], [], [], []
    stem = f"{body}-{pose}-{route}"
    failure = None
    scene = None
    try:
        with Session(body, task_id=5) as env:
            obs = env.reset(pose, seed)
            record["reset"] = env.reset_info
            scene = env.expert_scene()
            if route == "clearance":
                from clearance_expert import ClearanceExpert

                expert = ClearanceExpert(env, scene)
                record.update(candidates=expert.scores, selected=expert.chosen)
                if expert.chosen is None:
                    raise Stalled("No collision-free accurate source path found by the bounded IK search")
            else:
                expert = RimExpert(scene) if route == "inner_rim" else PushExpert(scene, route)
            record["waypoints_xy"] = [p.tolist() for p in expert.points]
            prompt = obs["prompt"]
            horizon = min(limit, env.reset_info["max_steps"])
            done = False
            for _ in range(horizon):
                a = expert.action(scene)
                next_obs, measured, reward, done, success = env.step(a)
                trace.append(dict(phase=expert.phase, **scene))
                frames.append(obs)
                actions.append(a)
                motion.append(measured)
                rewards.append(reward)
                obs = next_obs
                if obs["prompt"] != prompt:
                    raise ValueError("BASE instruction changed mid-episode")
                scene = env.expert_scene()
                if success:
                    record["native_success"] = True
                    if not done or reward <= 0:
                        raise ValueError("Native success, terminal flag and reward disagree")
                    break
                if done:
                    break
            record["done"] = done
            if not record["native_success"]:
                raise Stalled("Native termination without success" if done else "Collection step limit")
            path = Path(body) / f"{pose}.npz"
            save_tape(root / path, frames, actions, motion, rewards, obs["state"], prompt, body, pose, seed)
            record.update(accepted=True, file=str(path))
    except Stalled as exc:
        failure = str(exc)
    except Exception:
        failure = traceback.format_exc()
        record["error"] = True
    record.update(steps=len(actions), native_return=float(sum(rewards)), terminal_reward=rewards[-1] if rewards else None,
                  phase_counts=dict(Counter(t["phase"] for t in trace)), contact_steps=sum(t["contact"] for t in trace))
    if scene is not None:
        record["final_scene"] = scene
    if trace:
        record["initial_scene"] = trace[0]
        record["plate_travel_xy_m"] = float(np.linalg.norm(np.subtract(scene["plate"][:2], trace[0]["plate"][:2])))
    if failure:
        path = Path("failures") / f"{stem}.json"
        (root / path).parent.mkdir(exist_ok=True)
        record.update(failure=failure, evidence=str(path))
        # Keep failed controls and geometry inspectable outside the accepted NPZ bank.
        json_write(root / path, dict(attempt=record, pre_action_scene=trace, actions=[a.tolist() for a in actions],
                                     motion=[m.tolist() for m in motion], rewards=rewards,
                                     pre_action_state=[o["state"].tolist() for o in frames]))
    return record


def split_values(value, convert):
    values = [convert(part.strip()) for part in value.split(",")]
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Duplicate entries")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poses", type=lambda x: split_values(x, int), default=[0, 1])
    parser.add_argument("--bodies", type=lambda x: split_values(x, str), default=["Panda", "Sawyer"])
    parser.add_argument("--task-id", type=int, choices=[5], default=5)
    parser.add_argument("--seed", type=int, default=20261120, help="BASE seed for pose 0; each pose adds its index")
    parser.add_argument("--routes", type=lambda x: split_values(x, str), default=ROUTES)
    parser.add_argument("--max-steps", type=int, default=300, help="Per-attempt ceiling, capped by native horizon")
    args = parser.parse_args()
    if not set(args.routes) <= {*ROUTES, "inner_rim", "clearance"}:
        parser.error("Unknown demonstration route")
    if any(body not in COLLECTION_BODIES for body in args.bodies):
        parser.error(f"Collection bodies are {COLLECTION_BODIES}. IIWA is reserved.")
    if not all(0 <= p < 50 for p in args.poses) or args.max_steps < 1:
        parser.error("Expected released pose IDs 0..49 and a positive step limit")
    if args.seed < 0 or args.seed + max(args.poses) >= 2**32:
        parser.error("BASE seed plus pose must fit uint32")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    attempts = []
    code = [Path(__file__).with_name("rim_expert.py"), Path(__file__).resolve(), Path(__file__).with_name("sim.py"), PRIOR / "arm_env.py", PRIOR / "sim_reset.py"]
    index = dict(task_id=args.task_id, suite="libero_goal", case="BASE", poses=args.poses, bodies=args.bodies,
                 seed_base=args.seed, max_steps=args.max_steps, command=sys.argv, routes=args.routes,
                 code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code}, attempts=attempts)
    json_write(root / "index.json", index)
    requested = len(args.bodies) * len(args.poses)
    for body in args.bodies:
        for pose in args.poses:
            for route in args.routes:
                record = collect_attempt(root, body, pose, args.seed + pose, route, args.max_steps)
                attempts.append(record)
                json_write(root / "index.json", index)
                print(f"{body}/{pose} {route} accepted={record['accepted']} controls={record['steps']}", flush=True)
                if record["accepted"] or record.get("error"):
                    break
            quality = dict(requested=requested, accepted=sum(r["accepted"] for r in attempts),
                           attempts=len(attempts), native_successes=sum(r["native_success"] for r in attempts),
                           by_body={b: dict(requested=len(args.poses),
                                            accepted=sum(r["accepted"] for r in attempts if r["body"] == b),
                                            first_pass=sum(r["accepted"] for r in attempts
                                                           if r["body"] == b and r["route"] == args.routes[0]))
                                    for b in args.bodies})
            json_write(root / "quality.json", quality)
    return 0 if quality["accepted"] == requested else 1


if __name__ == "__main__":
    raise SystemExit(main())
