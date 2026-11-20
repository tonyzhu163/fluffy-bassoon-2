"""Search generated native Goal 5 layouts and replay positive controls on both source cameras."""

import argparse
from contextlib import closing
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from expert import session, sha, write_json
from demos import save_tape
from paddle import Paddle
from recover import BODIES, PROMPT, reset_identity
from sweep_bank import candidates


BANK_ROW = 0
RESET_REJECTIONS = (
    "Released grip-site pose unreachable",
    "Settling moved the paired task scene",
    "Task already successful",
)


def publish(out, row, frames, actions, motions, rewards, terminal):
    """Use the original R2 tape writer and add scalar, pickle-free origin metadata."""
    body, seed = row["body"], row["seed"]
    dest = Path(body) / f"generated-{seed}-route-{row['recipe']}.npz"
    with TemporaryDirectory(dir=out) as tmp:
        path = Path(tmp) / dest.name
        save_tape(path, frames, actions, motions, rewards, terminal, PROMPT, body, BANK_ROW, seed)
        origin_json = json.dumps(row["origin"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        with ZipFile(path, "a") as archive:
            for name, value in (("origin", origin_json), ("layout_seed", row["layout_seed"])):
                with archive.open(name + ".npy", "w") as entry:
                    np.lib.format.write_array(entry, np.asarray(value), allow_pickle=False)
        row["sha256"] = sha(path)
        path.rename(out / dest)
    row.update(accepted=True, file=str(dest))


def trial(out, body, seed, flat, row_fp, origin, number, route, max_steps, *, replay=None,
          expected_scene=None, search_reset=None):
    camera = replay is not None
    result = dict(body=body, pose=BANK_ROW, seed=seed, layout_seed=seed, origin=origin,
                  repeat=0, recipe=number, candidate_number=number, headless_candidates=number + 1,
                  params=asdict(route), route="rendered_commands" if camera else "whole_gripper_push",
                  camera=camera, steps=0, accepted=False, search_success=False, native_success=False,
                  terminal_reward=None, native_return=0., done=False)
    with session(body, camera) as env:
        arm = env.native.arm if camera else env.arm
        # ArmEnv.reset uses this instance's indexed Panda bank for both the native Panda
        # and Sawyer's named-scene transfer. No released row is passed to reset.
        arm._states = (flat,)
        try:
            obs = env.reset(BANK_ROW, seed)
        except ValueError as exc:
            if not str(exc).startswith(RESET_REJECTIONS):
                raise
            result.update(outcome="reset_rejected", reason=str(exc))
            return result, []
        result["reset"] = env.reset_info
        physics = reset_identity(env.reset_info, body, BANK_ROW, seed)
        if physics["released_row_fingerprint"] != row_fp or obs["prompt"] != PROMPT:
            raise ValueError("Native reset disagrees with generated Panda row or Goal 5 instruction")
        if expected_scene is not None and physics["scene_fingerprint"] != expected_scene:
            raise ValueError("Generated pre-settle task scene differs between native resets")
        if camera:
            same = physics["initial_state_fingerprint"] == search_reset["initial_state_fingerprint"]
            result.update(same_initial_state=same, search_steps=len(replay),
                          search_initial_state_fingerprint=search_reset["initial_state_fingerprint"])
            if not same:
                result.update(outcome="search_replay_reset_mismatch")
                return result, []
            vendor, renderer = env.reset_info["renderer"]
            if "NVIDIA" not in vendor.upper() or not renderer:
                raise ValueError("Native camera replay needs the allocated NVIDIA renderer")

        scene = env.expert_scene()
        geometry = env.native.geometry if camera else env.geometry
        controller = Paddle(route, scene)
        starting_distance = np.linalg.norm(controller.goal - controller.origin)
        best_gain, last_progress = 0., 0
        frames, actions, motions, rewards, samples = [], [], [], [], []
        limit = min(max_steps, env.reset_info["max_steps"])
        if camera:
            limit = min(limit, len(replay))
        outcome = "control_limit"
        for t in range(limit):
            # During replay, Paddle's diagnostics describe the camera rollout. Only
            # the exact action saved by the positive camera-free trial is issued.
            proposal = controller.action(geometry, scene)
            action = replay[t] if camera else proposal
            if camera:
                frames.append(obs)
            obs, motion, reward, done, success = env.step(action)
            if obs["prompt"] != PROMPT:
                raise ValueError("Native task instruction changed")
            if (reward > 0) != success or (success and not done) or reward < 0:
                raise ValueError("Native sparse Goal 5 reward and success disagree")
            actions.append(action.copy())
            motions.append(motion)
            rewards.append(float(reward))
            scene = env.expert_scene()
            gain = float(starting_distance - np.linalg.norm(controller.goal - np.array(scene["plate"][:2])))
            if gain > best_gain + .002:
                best_gain, last_progress = gain, t
            if t % 20 == 0:
                samples.append(dict(t=t, **controller.last))
            if done:
                outcome = "native_termination"
                break
            if not camera and controller.phase == 2 and controller.tick >= 65 and t - last_progress > 65:
                outcome = "no_plate_progress"
                break
        positive = bool(done and success and rewards[-1] > 0 and not any(rewards[:-1]))
        result.update(steps=len(actions), native_success=bool(success), done=bool(done),
                      terminal_reward=rewards[-1], native_return=sum(rewards),
                      goal_gain_m=gain, best_gain_m=best_gain, trace=samples,
                      last=controller.last, outcome=outcome, search_success=positive and not camera)
        if positive and camera:
            if len(actions) == len(replay):
                publish(out, result, frames, actions, motions, rewards, obs["state"])
                result["outcome"] = "accepted"
            else:
                result["outcome"] = "early_replay_termination"
        elif positive:
            result["outcome"] = "search_success"
        return result, actions if positive and not camera else []


def collect_layout(args, seed, flat, origin, row_fp, index):
    out = args.out
    state_file = Path("layouts") / f"{seed}.npy"
    with (out / state_file).open("xb") as stream:
        np.save(stream, flat, allow_pickle=False)
    layout = dict(layout_seed=seed, origin=origin, state_file=str(state_file),
                  state_sha256=sha(out / state_file), row_fingerprint=row_fp)
    index["layouts"].append(layout)
    write_json(out / "index.json", index)
    routes = candidates(seed, 0, args.candidates)
    scene_fp = None
    for body in args.bodies:
        for number, route in enumerate(routes):
            common = (out, body, seed, flat, row_fp, origin, number, route, args.max_steps)
            search, actions = trial(*common, expected_scene=scene_fp)
            index["attempts"].append(search)
            if "reset" in search:
                physics = search["reset"]["environment_reproducibility"]
                scene_fp = physics["scene_fingerprint"]
                layout["scene_fingerprint"] = scene_fp
            print(body, seed, number, "search", search["outcome"], search["terminal_reward"], flush=True)
            if search["outcome"] == "reset_rejected":
                break
            if search["search_success"]:
                draft = Path("drafts") / f"{body}-{seed}-route-{number}.npy"
                with (out / draft).open("xb") as stream:
                    np.save(stream, np.stack(actions), allow_pickle=False)
                search.update(search_file=str(draft), search_sha256=sha(out / draft))
                write_json(out / "index.json", index)
                row, _ = trial(*common, replay=actions, expected_scene=scene_fp, search_reset=physics)
                row.update(search_file=str(draft), search_sha256=search["search_sha256"])
                index["attempts"].append(row)
                print(body, seed, number, "replay", row["outcome"], row["terminal_reward"], flush=True)
                if row["accepted"]:
                    break
        write_json(out / "index.json", index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed-start", "--seed", type=int, default=20261120)
    p.add_argument("--layouts", "--num-layouts", type=int, default=1, help="Number of consecutive layout seeds")
    p.add_argument("--candidates", type=int, default=8, help="Maximum routes per arm and layout")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--bodies", nargs="+", choices=BODIES, default=list(BODIES))
    args = p.parse_args()
    if min(args.layouts, args.candidates, args.max_steps) < 1 or len(set(args.bodies)) != len(args.bodies):
        p.error("Use positive budgets and distinct source arms")
    if args.seed_start < 0 or args.seed_start + args.layouts > 2**32:
        p.error("The layout seeds must fit uint32")

    from generated_layouts import make_sampler, released_fingerprints, sample_layout
    from libero_remote.reproducibility import numerical_state_fingerprint

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in (*args.bodies, "layouts", "drafts"):
        (args.out / name).mkdir()
    index = dict(suite="libero_goal", task_id=5, case="BASE", bodies=args.bodies, poses=[BANK_ROW],
                 seed_base=args.seed_start, requested_layouts=args.layouts, layout_bank_row=BANK_ROW,
                 max_candidates=args.candidates, max_steps=args.max_steps, command=sys.argv,
                 revision=os.environ.get("MOTOR_CODE_REVISION"), source_constructor="native_generated_layout",
                 status="running", layouts=[], attempts=[])
    write_json(args.out / "index.json", index)
    released = released_fingerprints()
    seen = set()
    with closing(make_sampler()) as sampler:
        for seed in range(args.seed_start, args.seed_start + args.layouts):
            state, origin = sample_layout(sampler, seed, released)
            flat = np.array(state, copy=True)
            if (flat.ndim != 1 or not flat.size or flat.dtype.kind not in "fiu"
                    or not np.isfinite(flat).all() or not isinstance(origin, dict) or not origin):
                raise ValueError("Sampler must supply one finite flat Panda state and its origin dictionary")
            if origin.get("layout_seed", seed) != seed:
                raise ValueError("Sampler origin disagrees with the requested layout seed")
            row_fp = numerical_state_fingerprint(flat)
            if row_fp in released or row_fp in seen:
                raise ValueError("Generated Panda state duplicates a released or previously sampled row")
            seen.add(row_fp)
            collect_layout(args, seed, flat, origin, row_fp, index)
    counts = {body: sum(r["accepted"] for r in index["attempts"] if r["body"] == body) for body in args.bodies}
    index.update(status="complete", accepted=sum(counts.values()), accepted_by_body=counts)
    write_json(args.out / "index.json", index)
    print("accepted", counts, flush=True)
    return 0 if all(counts.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
