"""Search Sawyer outer-edge pushes and collect native Panda/Sawyer Goal 5 tapes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from functools import partial
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "libero-paligemma-crossarm-20261120-r1"
sys.path.insert(0, str(PREVIOUS))


def parse_layouts(spec):
    values = []
    for group in spec.split(","):
        bounds = [int(x) for x in group.split("-")]
        if len(bounds) == 1:
            values.extend(bounds)
        elif len(bounds) == 2 and bounds[1] >= bounds[0]:
            values.extend(range(bounds[0], bounds[1]+1))
        else:
            raise argparse.ArgumentTypeError(f"Bad layout range {group!r}")
    if len(set(values)) != len(values) or not all(0 <= p <= 31 for p in values):
        raise argparse.ArgumentTypeError("Use unique train 0..27 or offline-dev 28..31 layouts")
    return values


def write_json(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(root, row):
    # The body processes share one motor-readable index. Each finished attempt
    # is immediately visible even while the other process is rendering.
    with (root / "index.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        index = json.loads((root / "index.json").read_text())
        if "official_demo" in row:
            donor = row["official_demo"]
            index["references"][str(donor["demo"])] = donor
        index["attempts"].append(row)
        write_json(root / "index.json", index)
    print(f"{row['body']} layout={row['pose']} seed={row['seed']} {row['route']} "
          f"steps={row['steps']} reward={row['terminal_reward']} {row['outcome']}", flush=True)


class CameraFree:
    """Use the original ArmEnv physics step with proprioceptive observations."""

    def __init__(self, arm):
        self.arm = arm

    def reset(self, pose, seed):
        from sim import _PushGeometry
        packet = self.arm.reset(pose, seed)
        self.geometry = _PushGeometry(self.arm._env)
        self.reset_info = {k: packet[k] for k in ("environment_reproducibility", "model_identity", "max_steps")}
        self.reset_info["renderer"] = []
        return self.observation(packet)

    @staticmethod
    def observation(packet):
        import numpy as np
        raw = packet["observation"]
        q = raw["robot0_gripper_qpos"]
        state = np.r_[raw["robot0_eef_pos"], raw["robot0_eef_quat_xyzw"], q[0]-q[1]]
        return dict(prompt=packet["language_instruction"], state=state.astype(np.float32))

    def step(self, action):
        p = self.arm.step(action)
        return self.observation(p), p["sensor_motion"], p["reward"], p["terminated"] or p["truncated"], p["success"]

    def expert_scene(self):
        return self.geometry.read()


@contextmanager
def session(body, camera):
    from sim import Session, _arm_class
    if camera:
        with Session(body) as env:
            if env.native is None:
                raise ValueError("Launch with CROSSARM_SIM_PYTHON directly for live collision geometry")
            yield env
        return

    from unittest.mock import patch
    arm_class = _arm_class(body)
    module = sys.modules[arm_class.__module__]
    headless = partial(module.ControlEnv, use_camera_obs=False, has_offscreen_renderer=False)

    def public(raw, _joint_count):
        return module.normalize_standard_observation(raw, ())

    # Narrow observation-only substitutions around the original reset and step.
    # All model construction arguments, state transfer, OSC and rewards are inherited.
    with patch.object(module, "OffScreenRenderEnv", headless), patch.object(module, "_public", public):
        with arm_class(arm=body, suite="libero_goal", task_id=5) as arm:
            yield CameraFree(arm)


def native_trial(args, body, pose, seed, repeat, tag, *, candidate=None, commands=None,
                 tracker_ref=None, donor=None, raw_probe=False, source_receipt=None):
    import numpy as np
    from demos import save_tape
    from expert_sweep import OuterSweep, contact_names, final_arm_geometry
    from oracle_fallback import BDDL_SHA, TrajectoryExpert

    cam = not raw_probe
    root = args.output
    trace_dir = root / "traces"
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"{body}-{pose}-r{repeat}-{tag}.jsonl"
    frames, actions, motions, rewards = [], [], [], []
    dest = Path(body) / (f"{pose}.npz" if repeat == 0 else f"{pose}-r{repeat}-s{seed}.npz")
    with session(body, cam) as env, trace_path.open("x") as trace:
        try:
            obs = env.reset(pose, seed)
        except ValueError as exc:
            if not str(exc).startswith(("Released grip-site pose unreachable",
                                        "Settling moved the paired task scene", "Task already successful")):
                raise
            row = unattempted(body, pose, seed, repeat, tag, str(exc))
            row.update(trace=str(trace_path.relative_to(root)), reset_rejected=True, camera=cam)
            trace.write(json.dumps(dict(result=row)) + "\n")
            record(root, row)
            return row, []
        reset = env.reset_info
        lineage = reset["environment_reproducibility"]
        prompt = obs["prompt"]
        if (lineage["bddl_sha256"] != BDDL_SHA or lineage["init_state_id"] != pose
                or lineage["seed"] != seed or reset["model_identity"]["arm"] != body):
            raise ValueError("Live task, seed, layout or arm disagrees with request")
        if donor and donor["instruction"] != prompt:
            raise ValueError("Official archive and live instruction disagree")
        scene = env.expert_scene()
        geometry = env.native.geometry if cam else env.geometry
        controller = OuterSweep(geometry, scene, candidate) if candidate is not None and commands is None else None
        tracker = TrajectoryExpert(tracker_ref, scene) if tracker_ref is not None else None
        row = dict(body=body, pose=pose, seed=seed, repeat=repeat, split="train" if pose < 28 else "offline_dev",
                   route=tag, reset=reset, camera=cam, accepted=False, trace=str(trace_path.relative_to(root)))
        if candidate:
            row["candidate"] = asdict(candidate)
        if donor:
            row.update(official_demo=donor, reference_demo=donor["demo"], reference_pose=donor["pose"])
        if commands is not None:
            row["available_controls"] = len(commands)
        if source_receipt is not None:
            row["search_trace"] = source_receipt["trace"]
            row["search_root"] = str(args.search_bank or root)
            row["same_initial_state"] = lineage["initial_state_fingerprint"] == (
                source_receipt["reset"]["environment_reproducibility"]["initial_state_fingerprint"])

        def emit(obj):
            trace.write(json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n")

        emit(dict(reset=reset, initial_scene=scene, active_contacts=contact_names(geometry),
                  waypoints=[v.tolist() for v in controller.waypoints] if controller else []))
        start_plate = np.array(scene["plate"][:2])
        goal = np.array(scene["goal"][:2]) + .5 * np.array(scene["goal_half"][:2])
        initial_dist = np.linalg.norm(goal-start_plate)
        best_gain = 0.
        best_history = [best_gain]
        history = [np.array(scene["tcp"])]
        plate_history = [start_plate]
        angular_travel = [0.]
        phase_ticks = 0
        previous_phase = None
        first_stall = None
        first_contact = None
        contact_steps = 0
        phase_counts = {}
        success = done = False
        stop = None
        limit = min(args.max_steps, reset["max_steps"])
        if commands is not None:
            limit = min(limit, len(commands))
        for t in range(limit):
            if commands is not None:
                a = np.asarray(commands[t], np.float32)
                diag = dict(phase="issued_tape")
            elif tracker:
                a = tracker.action(obs, scene)
                diag = dict(phase="official_tracker", **tracker.trace[-1])
            else:
                a = controller.action(scene)
                diag = controller.last
            if cam:
                frames.append(obs)
            after, motion, reward, done, success = env.step(a)
            if after["prompt"] != prompt:
                raise ValueError("Live instruction changed")
            new = env.expert_scene()
            tcp, plate = np.array(new["tcp"]), np.array(new["plate"][:2])
            history.append(tcp)
            plate_history.append(plate)
            angular_travel.append(angular_travel[-1] + float(np.linalg.norm(motion[3:6])))
            gain = float(initial_dist - np.linalg.norm(goal-plate))
            best_gain = max(best_gain, gain)
            best_history.append(best_gain)
            contacts = contact_names(geometry)
            phase_counts[diag["phase"]] = phase_counts.get(diag["phase"], 0) + 1
            phase_ticks = phase_ticks + 1 if previous_phase == diag["phase"] else 1
            previous_phase = diag["phase"]
            w = min(12, phase_ticks)
            held_tcp = np.stack(history[-w-1:])
            tcp_span = float(np.linalg.norm(np.ptp(held_tcp, axis=0)))
            recent_plate = float(np.linalg.norm(plate-plate_history[-w-1]))
            recent_rotation = angular_travel[-1] - angular_travel[-w-1]
            step = dict(step=t+1, pre_action_state=obs["state"].tolist(), issued=a.tolist(),
                        measured_motion=motion.tolist(), control=diag, post_scene=new, active_contacts=contacts,
                        goal_gain_m=gain, best_goal_gain_m=best_gain, tcp_span_m=tcp_span,
                        window_controls=w, plate_window_travel_m=recent_plate, rotation_window_rad=recent_rotation,
                        native_reward=reward,
                        done=done, native_success=success)
            if new["contact"]:
                contact_steps += 1
                if first_contact is None:
                    first_contact = dict(step=t+1, goal_gain_m=gain, plate=plate.tolist())
            if (controller and w == 12 and tcp_span < .0025 and recent_rotation < .06 and
                    recent_plate < .0025 and np.linalg.norm(diag["tcp_error"]) > .010):
                first_stall = dict(step=t+1, window_start=t+1-w, phase=diag["phase"],
                                   plate_before_window=plate_history[-w-1].tolist(), plate_now=plate.tolist(),
                                   tcp_before_window=history[-w-1].tolist(), tcp_now=tcp.tolist(),
                                   goal_gain_before_window_m=float(initial_dist - np.linalg.norm(goal-plate_history[-w-1])),
                                   best_goal_gain_before_window_m=best_history[-w-1], active_contacts=contacts)
                step["tcp_stall"] = first_stall
                stop = "TCP and plate stationary across 12 controls with >10 mm commanded error"
            emit(step)
            if t % 8 == 0:
                trace.flush()
            actions.append(a.copy())
            motions.append(motion)
            rewards.append(float(reward))
            obs, scene = after, new
            if done or success or stop:
                break
            if controller:
                stop = controller.early_stop()
                if stop:
                    break
        native = bool(done and success and rewards and rewards[-1] > 0 and not any(rewards[:-1]))
        if body == "Sawyer":
            wrapper = env.native.arm._env if cam else env.arm._env
            row["final_arm_geometry"] = final_arm_geometry(wrapper, geometry)
        if native and cam:
            if (root / dest).exists():
                raise FileExistsError(root / dest)
            save_tape(root / dest, frames, actions, motions, rewards, obs["state"], prompt, body, pose, seed)
            row.update(accepted=True, file=str(dest))
        row.update(steps=len(actions), terminal_reward=rewards[-1] if rewards else None,
                   done=bool(done), native_success=bool(success), search_success=bool(native and not cam),
                   best_goal_gain_m=best_gain, final_goal_gain_m=float(initial_dist-np.linalg.norm(
                       goal-np.array(scene["plate"][:2]))), first_plate_contact=first_contact,
                   plate_travel_xy_m=float(np.linalg.norm(plate_history[-1]-start_plate)),
                   plate_contact_steps=contact_steps, phase_counts=phase_counts,
                   first_tcp_stall=first_stall, outcome="accepted" if row["accepted"] else (
                       "search_success" if native else stop or ("native termination" if done else "action/step limit")))
        emit(dict(result={k: v for k,v in row.items() if k != "reset"}))
    record(root, row)
    return row, actions


def targets(args):
    for repeat in range(args.repeats):
        for pose in args.layouts:
            yield pose, args.seed + pose + 1000*repeat, repeat


def unattempted(body, pose, seed, repeat, route, reason):
    return dict(body=body, pose=pose, seed=seed, repeat=repeat, route=route, outcome=reason,
                steps=0, terminal_reward=None, done=False, native_success=False,
                accepted=False, search_success=False)


def external_search(path):
    if path is None:
        return {}
    index = json.loads((path / "index.json").read_text())
    if (index["suite"], index["task_id"], index["case"]) != ("libero_goal", 5, "BASE"):
        raise ValueError("Unexpected search bank task")
    winners = {}
    for row in index["attempts"]:
        if row["body"] != "Sawyer" or row["pose"] not in range(32):
            raise ValueError("Expected a source-only Sawyer search bank")
        if row.get("search_success"):
            winners.setdefault((row["pose"], row["seed"]), []).append(row)
    return winners


def trace_actions(root, row):
    import numpy as np
    trace = (root / row["trace"]).resolve(strict=True)
    if not trace.is_relative_to(root.resolve()):
        raise ValueError("Search trace outside bank")
    with trace.open() as f:
        actions = [item["issued"] for line in f if "issued" in (item := json.loads(line))]
    actions = np.asarray(actions, dtype=np.float32)
    if (actions.shape != (row["steps"], 7) or not np.isfinite(actions).all()
            or abs(actions).max() > 1 or not (row["done"] and row["native_success"] and row["terminal_reward"] > 0)):
        raise ValueError("Invalid successful search receipt")
    return actions


def sawyer(args):
    from expert_sweep import Candidate, candidate_list
    candidates = [] if args.search_bank else candidate_list(args.params, args.candidates)
    winners = external_search(args.search_bank)
    complete = 0
    for pose, seed, repeat in targets(args):
        if args.search_bank:
            matches = [r for r in winners.get((pose, seed), [])
                       if not args.candidates or r["candidate"]["name"] in args.candidates]
            if not matches:
                record(args.output, unattempted("Sawyer", pose, seed, repeat, "render",
                                                "no matching positive search receipt"))
            for previous in matches:
                c = Candidate(**previous["candidate"])
                row, _ = native_trial(args, "Sawyer", pose, seed, repeat, f"render-{c.name}",
                                      candidate=c, commands=trace_actions(args.search_bank, previous),
                                      source_receipt=previous)
                if row["accepted"]:
                    complete += 1
                    break
            continue
        ordered = candidates[repeat % len(candidates):] + candidates[:repeat % len(candidates)]
        has_winner = False
        for c in ordered:
            probe = args.mode == "search" or args.search_first
            tag = ("search-" if probe else "outer-") + c.name
            row, actions = native_trial(args, "Sawyer", pose, seed, repeat, tag, candidate=c, raw_probe=probe)
            if row.get("reset_rejected"):
                break
            if args.mode == "collect" and row["search_success"]:
                row, _ = native_trial(args, "Sawyer", pose, seed, repeat, f"render-{c.name}",
                                      candidate=c, commands=actions, source_receipt=row)
            if row["accepted"] or (args.mode == "search" and row["search_success"]):
                has_winner = True
                if not args.all_candidates:
                    break
        complete += int(has_winner)
    return complete


def panda(args):
    import h5py
    import numpy as np
    from oracle_fallback import panda_reference

    # Scan metadata to select released IDs. Open action/state arrays only after
    # the source split has been checked. Holdout donors cannot enter training.
    available = {}
    by_pose = {}
    ref_cache = {}
    with h5py.File(args.archive, "r") as archive:
        for key, group in archive["data"].items():
            info = json.loads(group.attrs["source_metadata"])
            p = int(info["init_state_id"])
            if p not in range(32) or (p >= 28 and p not in args.layouts):
                continue
            num = int(key.removeprefix("demo_"))
            if not (group.attrs["success"] and group["rewards"][-1] > 0 and group["dones"][-1]):
                continue
            available[num] = dict(demo=num, pose=p, seed=int(info["seed"]), instruction=info["instruction"])
            by_pose.setdefault(p, []).append(num)
        if not args.official_only:
            for n in args.panda_donors:
                if n not in available or available[n]["pose"] >= 28:
                    raise ValueError(f"Fallback donor {n} must be a verified training-layout release")
        complete = 0
        for pose, seed, repeat in targets(args):
            matches = sorted(by_pose.get(pose, []))
            if args.official_only and not matches:
                record(args.output, unattempted("Panda", pose, seed, repeat, "official",
                                                "no matching published successful demonstration"))
                continue
            ordered = matches[repeat % len(matches):] + matches[:repeat % len(matches)] if matches else []
            accepted = False
            for num in ordered:
                donor = available[num]
                # Use the recorded seed for the first matched release. Later
                # repeats use the requested independent seed and remain labelled.
                target_seed = donor["seed"] if repeat == 0 else seed
                commands = archive[f"data/demo_{num}/actions"][:].astype(np.float32)
                row, _ = native_trial(args, "Panda", pose, target_seed, repeat, f"official-{num}",
                                      commands=commands, donor=donor)
                if row["accepted"]:
                    accepted = True
                    break
            if not accepted and not args.official_only:
                donors = list(args.panda_donors)
                donors = donors[repeat % len(donors):] + donors[:repeat % len(donors)]
                for num in dict.fromkeys(ordered + donors):
                    if num not in ref_cache:
                        try:
                            ref_cache[num] = panda_reference(args.archive, num)
                        except RuntimeError as exc:
                            if not str(exc).startswith("Released Panda replay ended at"):
                                raise
                            ref_cache[num] = None
                            write_json(args.output / f"panda-reference-{num}.json",
                                       dict(donor=available[num], error=str(exc)))
                        else:
                            write_json(args.output / f"panda-reference-{num}.json", ref_cache[num]["evidence"])
                    if ref_cache[num] is None:
                        continue
                    row, _ = native_trial(args, "Panda", pose, seed, repeat, f"tracked-{num}",
                                          tracker_ref=ref_cache[num], donor=available[num])
                    if row["accepted"]:
                        accepted = True
                        break
            complete += int(accepted)
    return complete


def lane(args, body):
    return panda(args) if body == "Panda" else sawyer(args)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("search", "collect"),
                        help="Sawyer camera-free parameter search, or normal-camera demonstration collection")
    parser.add_argument("--output", "--out", type=Path, default=Path("bank"), help="Fresh bank directory")
    parser.add_argument("--bodies", nargs="+", choices=("Panda", "Sawyer"), help="Default both for collect, Sawyer for search")
    parser.add_argument("--layouts", "--poses", type=parse_layouts, default=[0],
                        help="Comma-separated source layouts/ranges, e.g. 0-27 or 28-31")
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--repeats", type=int, default=1, help="Independent target seeds requested per layout")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--candidates", type=lambda s: s.split(","), help="Ordered comma-separated Sawyer candidate names")
    parser.add_argument("--params", type=Path, help="JSON list of extra named candidate parameter objects")
    parser.add_argument("--all-candidates", action="store_true", help="Continue search after a positive native outcome")
    parser.add_argument("--search-first", action="store_true", help="Camera-free Sawyer search then full-camera command replay")
    parser.add_argument("--search-bank", type=Path, help="Only replay positive Sawyer command traces from this search bank")
    parser.add_argument("--archive", type=Path, default=os.environ.get("DUAL_ARCHIVE"),
                        help="Pinned official Goal 5 BASE episodes.hdf5, defaults to DUAL_ARCHIVE")
    parser.add_argument("--panda-donors", type=lambda s: [int(x) for x in s.split(",")], default=[22, 1, 0, 6],
                        help="Training donor IDs for feedback when the exact Panda release is missing or fails")
    parser.add_argument("--official-only", action="store_true", help="Use only matching archived Panda commands")
    args = parser.parse_args()
    args.bodies = args.bodies or (["Sawyer"] if args.mode == "search" else ["Panda", "Sawyer"])
    if (args.repeats < 1 or args.max_steps < 1 or args.seed < 0
            or args.seed + max(args.layouts) + 1000*(args.repeats-1) >= 2**32):
        parser.error("Expected positive repeat/step budgets and uint32 target seeds")
    if len(set(args.bodies)) != len(args.bodies) or (args.candidates and len(set(args.candidates)) != len(args.candidates)):
        parser.error("Duplicate body or candidate")
    if args.mode == "search" and (args.bodies != ["Sawyer"] or args.search_bank or args.search_first):
        parser.error("Search runs camera-free live Sawyer dynamics on each candidate")
    if (args.all_candidates and args.mode != "search") or (args.search_bank and args.search_first):
        parser.error("Use --all-candidates with search, and select one Sawyer replay mode")
    if "Panda" in args.bodies and args.archive is None:
        parser.error("Panda requires --archive or DUAL_ARCHIVE")
    if args.mode == "search" and any(p >= 28 for p in args.layouts):
        parser.error("Tune Sawyer candidates on training layouts 0..27 only")
    if (args.search_first or args.search_bank) and "Sawyer" not in args.bodies:
        parser.error("Sawyer replay mode needs the Sawyer body")
    args.output = args.output.resolve()
    return args


def main():
    args = arguments()
    source = [Path(__file__), HERE / "expert_sweep.py", *(PREVIOUS / name for name in (
        "sim.py", "demos.py", "oracle_fallback.py")), *(HERE.parent / "libero-crossarm-20261120-r1" / name
                                                       for name in ("arm_env.py", "sim_reset.py"))]
    archive_sha = None
    if "Panda" in args.bodies:
        from oracle_fallback import ARCHIVE_SHA
        archive_sha = sha(args.archive)
        if archive_sha != ARCHIVE_SHA:
            raise ValueError("Official Goal 5 BASE archive SHA-256 mismatch")
    params = []
    if "Sawyer" in args.bodies:
        if args.search_bank:
            search_index = json.loads((args.search_bank / "index.json").read_text())
            params = [c for c in search_index["candidates"] if not args.candidates or c["name"] in args.candidates]
        else:
            from expert_sweep import candidate_list
            params = [asdict(c) for c in candidate_list(args.params, args.candidates)]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "index.json", dict(
        suite="libero_goal", task_id=5, case="BASE", bodies=args.bodies, mode=args.mode,
        poses=args.layouts, repeats=args.repeats, seed_base=args.seed, candidates=params,
        archive_sha256=archive_sha, command=sys.argv, code_sha256={p.name: sha(p) for p in source},
        revision=os.environ.get("MOTOR_CODE_REVISION"), references={}, attempts=[]))
    if len(args.bodies) == 2:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
            runs = {body: pool.submit(lane, args, body) for body in args.bodies}
            counts = {body: job.result() for body, job in runs.items()}
    else:
        body, = args.bodies
        counts = {body: lane(args, body)}
    requested = args.repeats * len(args.layouts)
    index = json.loads((args.output / "index.json").read_text())
    by_body = {}
    for body, n in counts.items():
        rows = [r for r in index["attempts"] if r["body"] == body and (r["accepted"] or (
            args.mode == "search" and r["search_success"]))]
        by_body[body] = dict(requested=requested, completed=n, unique_layouts=sorted({r["pose"] for r in rows}),
                             unique_initial_states=len({r["reset"]["environment_reproducibility"][
                                 "initial_state_fingerprint"] for r in rows}))
    write_json(args.output / "quality.json", dict(mode=args.mode, by_body=by_body))
    return int(any(n != requested for n in counts.values()))


if __name__ == "__main__":
    raise SystemExit(main())
