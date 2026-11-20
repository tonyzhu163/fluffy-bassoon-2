"""Search generated native Goal 5 layouts and replay positive controls on both source cameras."""

import argparse
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from expert import session, sha, write_json
from demos import save_tape
from generated_layouts import LayoutRejected, code_receipt, make_sampler, released_row_fingerprints, reset_generated, sample_layout
from generated_split import SOURCE_KIND, validate_generated_attempt, validate_generated_groups, validate_generated_tape
from paddle import Paddle
from recover import BODIES, PROMPT, RELEASE
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
        save_tape(path, frames, actions, motions, rewards, terminal, PROMPT, body, row["pose"], seed)
        origin_json = json.dumps(row["layout_origin"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        with ZipFile(path, "a") as archive:
            for name, value in (("layout_origin", origin_json), ("layout_seed", row["layout_seed"])):
                with archive.open(name + ".npy", "w") as entry:
                    np.lib.format.write_array(entry, np.asarray(value), allow_pickle=False)
        with np.load(path, allow_pickle=False) as tape:
            validate_generated_tape(tape, row)
        row["sha256"] = sha(path)
        path.rename(out / dest)
    row.update(accepted=True, file=str(dest))


def trial(out, body, seed, flat, origin, number, route, max_steps, index, *, replay=None, search_reset=None):
    camera = replay is not None
    result = dict(body=body, pose=origin["source_index"], split="train", seed=seed,
                  layout_seed=seed, source_kind=origin["source_kind"], layout_origin=origin,
                  repeat=0, recipe=number, candidate_number=number, headless_candidates=number + 1,
                  params=asdict(route), route="rendered_commands" if camera else "whole_gripper_push",
                  camera=camera, steps=0, accepted=False, search_success=False, native_success=False,
                  terminal_reward=None, native_return=0., done=False)
    with session(body, camera) as env:
        try:
            obs = reset_generated(env, body, camera, flat, origin)
        except ValueError as exc:
            if not isinstance(exc, LayoutRejected) and not str(exc).startswith(RESET_REJECTIONS):
                raise
            result.update(outcome="reset_rejected", reason=str(exc))
            return result, []
        result["reset"] = env.reset_info
        physics = env.reset_info["environment_reproducibility"]
        if obs["prompt"] != PROMPT:
            raise ValueError("Unexpected native Goal 5 instruction")
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
                result["accepted"] = True
                validate_generated_attempt(index, result, native_reset=RELEASE)
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
    layout = dict(**origin, state_file=str(state_file),
                  state_sha256=sha(out / state_file), row_fingerprint=row_fp)
    index["layouts"].append(layout)
    index["poses"].append(origin["source_index"])
    write_json(out / "index.json", index)
    routes = candidates(seed, 0, args.candidates)
    for body in args.bodies:
        for number, route in enumerate(routes):
            common = (out, body, seed, flat, origin, number, route, args.max_steps, index)
            search, actions = trial(*common)
            index["attempts"].append(search)
            print(body, seed, number, "search", search["outcome"], search["terminal_reward"], flush=True)
            if search["outcome"] == "reset_rejected":
                break
            if search["search_success"]:
                draft = Path("drafts") / f"{body}-{seed}-route-{number}.npy"
                with (out / draft).open("xb") as stream:
                    np.save(stream, np.stack(actions), allow_pickle=False)
                search.update(search_file=str(draft), search_sha256=sha(out / draft))
                write_json(out / "index.json", index)
                physics = search["reset"]["environment_reproducibility"]
                row, _ = trial(*common, replay=actions, search_reset=physics)
                row.update(search_file=str(draft), search_sha256=search["search_sha256"])
                index["attempts"].append(row)
                print(body, seed, number, "replay", row["outcome"], row["terminal_reward"], flush=True)
                if row["accepted"]:
                    break
        write_json(out / "index.json", index)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed-start", "--seed", type=int, default=283100)
    p.add_argument("--layouts", "--count", type=int, default=1, help="Number of distinct sampled layouts")
    p.add_argument("--max-seeds", type=int, help="Maximum consecutive seeds to try (default: --layouts)")
    p.add_argument("--candidates", type=int, default=8, help="Maximum routes per arm and layout")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--bodies", nargs="+", choices=BODIES, default=list(BODIES))
    args = p.parse_args()
    if min(args.layouts, args.candidates, args.max_steps) < 1 or len(set(args.bodies)) != len(args.bodies):
        p.error("Use positive budgets and distinct source arms")
    budget = args.layouts if args.max_seeds is None else args.max_seeds
    if budget < args.layouts or args.seed_start < 0 or args.seed_start + budget > 2**32:
        p.error("The seed budget must cover the requested layout count and fit uint32")

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    for name in (*args.bodies, "layouts", "drafts"):
        (args.out / name).mkdir()
    index = dict(suite="libero_goal", task_id=5, case="BASE", bodies=args.bodies, poses=[], split="train",
                 seed_base=args.seed_start, requested_layouts=args.layouts, layout_bank_row=BANK_ROW,
                 max_candidates=args.candidates, max_steps=args.max_steps, command=sys.argv,
                 **code_receipt(), source_constructor="native_generated_layout",
                 source_kind=SOURCE_KIND, max_seeds=budget, status="running", layouts=[], rejected=[], attempts=[],
                 public_row_fingerprints=released_row_fingerprints())
    write_json(args.out / "index.json", index)
    released = frozenset(index["public_row_fingerprints"])
    seen_states, seen_placements = set(), set()
    with closing(make_sampler()) as sampler:
        for seed in range(args.seed_start, args.seed_start + budget):
            try:
                state, origin = sample_layout(sampler, seed, released)
                if origin["movable_xy_hash"] in seen_placements or origin["initial_state_sha256"] in seen_states:
                    raise LayoutRejected("Duplicate generated row or movable XY placement")
            except LayoutRejected as exc:
                index["rejected"].append(dict(layout_seed=seed, reason=str(exc)))
                write_json(args.out / "index.json", index)
                continue
            row_fp = origin["initial_state_sha256"]
            seen_states.add(row_fp)
            seen_placements.add(origin["movable_xy_hash"])
            collect_layout(args, seed, state, origin, row_fp, index)
            if len(index["layouts"]) == args.layouts:
                break
    counts = {body: sum(r["accepted"] for r in index["attempts"] if r["body"] == body) for body in args.bodies}
    positive_groups = {b: {r["pose"] for r in index["attempts"] if r["accepted"] and r["body"] == b}
                       for b in ("Panda", "Sawyer")}
    records = (validate_generated_attempt(index, row, native_reset=RELEASE)
               for row in index["attempts"] if row["accepted"])
    validate_generated_groups(records, index["public_row_fingerprints"], native_reset=RELEASE)
    met_request = len(index["layouts"]) == args.layouts and all(counts.values())
    index.update(status="complete" if met_request else "shortfall", accepted=sum(counts.values()),
                 admitted_layouts=len(index["layouts"]), accepted_by_body=counts,
                 paired_positive_groups=sorted(positive_groups["Panda"] & positive_groups["Sawyer"]))
    write_json(args.out / "index.json", index)
    print("accepted", counts, flush=True)
    return 0 if met_request else 2


if __name__ == "__main__":
    raise SystemExit(main())
