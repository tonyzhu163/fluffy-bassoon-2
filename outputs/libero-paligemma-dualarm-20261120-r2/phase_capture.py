"""Capture dense phase labels from online Paddle camera rollouts on released training starts."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from expert import HERE, PREVIOUS, parse_layouts, session, sha, write_json
from generated_split import BANK_SHA, TASK

BODIES = ("Panda", "Sawyer")
FRAME = "world orientation of the hand link, XYZW in public state[3:7]"


def input_receipts(poses):
    from libero.libero import benchmark, get_libero_path
    from libero_remote.reproducibility import numerical_state_fingerprint

    suite = benchmark.get_benchmark_dict()[TASK["suite"]]()
    task = suite.get_task(TASK["task_id"])
    path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    digest = sha(path)
    if digest != BANK_SHA:
        raise ValueError("Released Goal 5 state bank hash mismatch")
    rows = suite.get_task_init_states(TASK["task_id"])
    return dict(path=str(path), sha256=digest,
                rows={p: numerical_state_fingerprint(rows[p]) for p in poses})


def provenance():
    arm = HERE.parent / "libero-crossarm-20261120-r1"
    paths = [Path(__file__).resolve(), *(HERE / n for n in (
        "paddle.py", "expert_sweep.py", "expert.py", "recover.py", "generated_split.py",
        "waypoint_labels.py")), PREVIOUS / "demos.py", PREVIOUS / "sim.py",
        arm / "arm_env.py", arm / "sim_reset.py"]
    revision = os.environ.get("MOTOR_CODE_REVISION")
    if not revision:
        revision = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                                           text=True).strip()
    return dict(code_revision=revision, code_sha256={p.name: sha(p) for p in paths},
                command=sys.argv)


def frame_receipt(state, geometry, identity):
    """Compare the observed quaternion to both live world frames, independent of sign."""
    from scipy.spatial.transform import Rotation

    public = Rotation.from_quat(state[3:7])
    hand_id = geometry.model.body_name2id(identity["hand_link"])
    hand = Rotation.from_matrix(geometry.data.body_xmat[hand_id].reshape(3, 3))
    site = Rotation.from_matrix(geometry.data.site_xmat[geometry.tcp_id].reshape(3, 3))
    hand_error = float((public.inv() * hand).magnitude())
    site_error = float((public.inv() * site).magnitude())
    if not hand_error < min(0.01, site_error):
        raise ValueError(f"Reset quaternion frame mismatch: hand={hand_error}, site={site_error}")
    return dict(public=FRAME, hand_link=identity["hand_link"], grip_site=identity["tcp_site"],
                public_to_hand_rad=hand_error, public_to_site_rad=site_error,
                hand_from_site_xyzw=(hand.inv() * site).as_quat().tolist())


def teacher_step(teacher, geometry, scene):
    """Read the teacher's pre-switch geometric error at its native 0.1 mm precision."""
    import numpy as np

    phase = teacher.phase
    action = teacher.action(geometry, scene)
    error = np.asarray(teacher.last["error"], dtype=np.float32)
    if error.shape != (3,) or not np.isfinite(error).all():
        raise ValueError("Invalid Paddle position error")
    return action, error, phase, teacher.phase != phase


def capture(out, body, pose, seed, repeat, route, limit, bank):
    import numpy as np
    from demos import save_tape
    from paddle import Paddle
    from recover import PROMPT, check_source, reset_identity
    from sim import _public
    from waypoint_labels import RESET_REJECTIONS, renderer

    check_source(body, pose)
    row = dict(body=body, pose=pose, seed=seed, repeat=repeat, recipe=0, params=asdict(route),
               route="whole_gripper_push", split="train", camera=True, accepted=False)
    with session(body, True) as env:
        try:
            obs = env.reset(pose, seed)
        except ValueError as exc:
            if not str(exc).startswith(RESET_REJECTIONS):
                raise
            return dict(row, outcome="reset_rejected", steps=0, reason=str(exc))
        reset = row["reset"] = env.reset_info
        physics = reset_identity(reset, body, pose, seed)
        if physics["released_row_fingerprint"] != bank["rows"][pose]:
            raise ValueError("Reset differs from requested released training row")
        renderer(reset)
        scene = env.expert_scene()
        geometry = env.native.geometry
        row["quaternion_frame"] = frame_receipt(obs["state"], geometry, reset["model_identity"])
        from scipy.spatial.transform import Rotation
        teacher = Paddle(route, scene)
        hand_from_site = Rotation.from_quat(row["quaternion_frame"]["hand_from_site_xyzw"])
        row["desired_hand_xyzw"] = (teacher.orientation * hand_from_site.inv()).as_quat().tolist()
        frames, plates, errors, phases, switches, actions, motions, rewards = ([] for _ in range(8))
        trace = []
        for tick in range(min(limit, reset["max_steps"])):
            if obs["prompt"] != PROMPT:
                raise ValueError("Unexpected Goal 5 instruction")
            frames.append(obs)
            plates.append(np.array(scene["plate"][:2], dtype=np.float32))
            action, error, phase, switched = teacher_step(teacher, geometry, scene)
            if tick % 20 == 0:
                trace.append(dict(t=tick, **teacher.last))
            # Same native transition and public packet conversion as sim._Native.step.
            packet = env.native.arm.step(action)
            if (packet["worker_step_index"] != tick + 1
                    or not np.array_equal(action, packet["issued_action"])):
                raise ValueError("Native transition and teacher action disagree")
            obs = _public(packet)
            reward = float(packet["reward"])
            success = bool(packet["success"])
            done = bool(packet["terminated"] or packet["truncated"])
            if (not np.isfinite(reward) or reward < 0 or (reward > 0) != success
                    or (success and not done) or obs["prompt"] != PROMPT):
                raise ValueError("Invalid native Goal 5 transition")
            phases.append(phase)
            switches.append(switched)
            errors.append(error.astype(np.float32))
            actions.append(packet["issued_action"])
            motions.append(np.asarray(packet["sensor_motion"], np.float32))
            rewards.append(reward)
            if done:
                break
            scene = env.expert_scene()

    positive = done and success and reward > 0
    row.update(done=done, native_success=success, terminal_reward=reward, native_return=sum(rewards),
               steps=len(actions), search_success=positive, last=teacher.last, trace=trace,
               outcome="accepted" if positive else ("native_termination" if done else "step_limit"))
    if positive:
        name = f"{pose}-{seed}-r{repeat}.npz"
        tape = out / body / name
        sidecar = out / "phase" / body / name
        save_tape(tape, frames, actions, motions, rewards, obs["state"], PROMPT, body, pose, seed)
        digest = sha(tape)
        pre_state = np.stack([frame["state"] for frame in frames])
        issued = np.stack(actions)
        with np.load(tape, allow_pickle=False) as saved:
            if not (np.array_equal(pre_state, saved["state"]) and np.array_equal(issued, saved["actions"])):
                raise ValueError("Phase labels and positive tape differ")
        with sidecar.open("xb") as stream:
            np.savez_compressed(stream, pre_state=pre_state, plate_xy=np.stack(plates),
                                error=np.stack(errors), phase=np.asarray(phases, dtype=np.int64),
                                switch=np.asarray(switches, dtype=bool), action=issued, tape_sha256=digest)
        row.update(accepted=True, file=str(tape.relative_to(out)), sha256=digest,
                   phase_file=str(sidecar.relative_to(out)), phase_sha256=sha(sidecar))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route", choices=("primary", "straight"), default="primary")
    p.add_argument("--poses", type=parse_layouts, default=list(range(28)), help="e.g. 0-5,8")
    p.add_argument("--bodies", choices=BODIES, nargs="+", default=list(BODIES))
    p.add_argument("--out", type=Path, required=True, help="Fresh bank directory")
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()
    if not all(pose in range(28) for pose in args.poses) or len(set(args.bodies)) != len(args.bodies):
        p.error("Use distinct source arms and released training layouts 0..27")
    if min(args.max_steps, args.repeats) < 1:
        p.error("Positive step and repeat counts required")
    last_seed = args.seed + max(args.poses) + 1000 * (args.repeats - 1)
    if args.seed + min(args.poses) < 0 or last_seed >= 2**32:
        p.error("Seeds must fit uint32")
    jobs = [(body, pose, args.seed + pose + 1000 * rep, rep) for rep in range(args.repeats)
            for pose in args.poses for body in args.bodies]
    if any(740000 <= seed < 750000 or 840000 <= seed < 850000 for _, _, seed, _ in jobs):
        p.error("Requested seed overlaps a reserved evaluation range")

    from paddle import Route
    from recover import RELEASE

    route = Route("rim", 28, 180, -35 if args.route == "primary" else 0)
    bank = input_receipts(args.poses)
    out = args.out.resolve()
    index = dict(**TASK, phase_schema=1, bodies=args.bodies, poses=args.poses, split="train",
                 seed_base=args.seed, repeats=args.repeats, max_steps=args.max_steps,
                 fixed_route=asdict(route), source_constructor="online_fixed_route_paddle",
                 input_receipts=dict(release=RELEASE, released_bank=bank), **provenance(),
                 public_quaternion=FRAME, error_units="world metres, before phase transition",
                 attempts=[], status="running", accepted=0)
    out.mkdir(parents=True, exist_ok=False)
    for body in args.bodies:
        (out / body).mkdir()
        (out / "phase" / body).mkdir(parents=True)
    write_json(out / "index.json", index)
    counts = dict.fromkeys(args.bodies, 0)
    for body, pose, seed, rep in jobs:
        row = capture(out, body, pose, seed, rep, route, args.max_steps, bank)
        index["attempts"].append(row)
        counts[body] += int(row["accepted"])
        index["accepted"] = sum(counts.values())
        write_json(out / "index.json", index)
        print(json.dumps({k: row[k] for k in ("body", "pose", "seed", "outcome", "steps",
                         "terminal_reward", "quaternion_frame") if k in row}), flush=True)
    index.update(status="complete", accepted_by_body=counts)
    write_json(out / "index.json", index)
    print(json.dumps(dict(index=str(out / "index.json"), accepted_by_body=counts)), flush=True)
    return 0 if all(counts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
