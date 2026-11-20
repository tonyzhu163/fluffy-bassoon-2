"""Fresh source training pairs and three rendered, native Goal 5 development rollouts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path

import numpy as np

from native_sim import (ARMS, CHECKPOINT, DEV, FIT_SHA, FORK, HERE, N_TRAIN, PANEL_SHA, PIN,
                        REPORT_SHA, RenderedSession, dev_panel, prepare, train_rows)
from large_panel_protocol import bind, candidate, digest, publish, read, require, sha
from large_panel_rollout import BATCH, infer, load_motor, observation_hashes, public_observation


class Episode:
    def __init__(self, obs, reset, mode):
        self.obs = obs
        self.reset = reset
        self.mode = mode
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.truncated = []
        self.successes = []
        self.observations = []
        self.proposals = []
        self.features = []
        self.fork_x = None
        self.score = None
        self.enabled = False

    def command(self, public, raw, policy):
        import model
        step = len(self.actions)
        self.observations.append(observation_hashes(public))
        self.proposals.append(raw.copy())
        if step == FORK:
            self.fork_x = model.features(public, raw, step)
            if self.mode == "gated":
                self.score = float(policy.score(public, raw, step))
                require(np.isfinite(self.score), "Nonfinite gate score")
                self.enabled = self.score > model.GATE_THRESHOLD
            else:
                self.enabled = self.mode == "always"
        if self.mode == "teacher" and step >= FORK:
            self.features.append(self.fork_x if step == FORK else model.features(public, raw, step))
        if policy is None:
            return raw
        action = policy.apply(public, raw.copy(), step, self.enabled)
        require(action.shape == raw.shape and action.dtype == np.float32 and np.isfinite(action).all()
                and np.max(abs(action)) <= 1, "Invalid learned command")
        if not self.enabled:
            require(np.array_equal(raw, action), "Disabled correction changed the frozen proposal")
        return action

    def consume(self, observation, trace, command):
        self.obs = observation
        start = len(self.actions)
        require(1 <= len(trace) <= 5, "Invalid native step count")
        for j, (action, reward, terminated, truncated, success, index) in enumerate(trace):
            require(index == start + j + 1 and reward in (0, 1) and (reward == 1) == success and
                    (not success or terminated) and not (terminated and truncated) and
                    (not (terminated or truncated) or j == len(trace) - 1), "Native step or reward mismatch")
            action = np.asarray(action, dtype=np.float32)
            require(action.shape == (7,) and np.isfinite(action).all() and np.max(abs(action)) <= 1,
                    "Invalid issued control")
            if self.mode != "teacher" or start < FORK:
                require(np.array_equal(action, command[j]), "Executed action differs from policy")
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminated.append(terminated)
            self.truncated.append(truncated)
            self.successes.append(success)
        done = self.terminated[-1] or self.truncated[-1]
        require(len(self.actions) <= 300 and (self.successes[-1] or len(self.actions) == 300) == done
                and (done or len(trace) == 5), "Invalid native horizon or partial replan")
        return done

    def evidence(self):
        return dict(actions=np.asarray(self.actions, np.float32), base=np.asarray(self.proposals, np.float32),
                    observations=np.asarray(json.dumps(self.observations, sort_keys=True)),
                    native_steps=np.arange(1, len(self.actions) + 1, dtype=np.int32),
                    rewards=np.asarray(self.rewards, np.float32), success=np.asarray(self.successes, bool),
                    terminated=np.asarray(self.terminated, bool), truncated=np.asarray(self.truncated, bool))

    def result(self):
        return dict(success=bool(self.successes[-1]), native_return=float(sum(self.rewards)),
                    steps=len(self.actions), decision_reached=self.fork_x is not None,
                    gate_score=self.score, enabled=bool(self.enabled))


def rollout(pool, sessions, rows, mode, motor, policy, prompt):
    resets = [pool.submit(env.start, row["layout_seed"], mode) for env, row in zip(sessions, rows)]
    episodes = [Episode(*f.result(), mode) for f in resets]
    active = list(range(len(episodes)))
    while active:
        observations = [public_observation(episodes[i].obs, prompt) for i in active]
        raw = infer(motor, observations)
        jobs = []
        for i, obs, chunk in zip(active, observations, raw, strict=True):
            command = episodes[i].command(obs, chunk, policy)
            jobs.append((i, command, pool.submit(sessions[i].advance_native, command)))
        active = []
        for i, command, f in jobs:
            observation, trace = f.result()
            if not episodes[i].consume(observation, trace, command):
                active.append(i)
    return episodes


def compare(raw, other, *, complete=False):
    steps = len(raw.actions) if complete else min(FORK, len(raw.actions))
    require(len(other.actions) >= steps, "Branch terminated before its matched prefix")
    if complete or steps < FORK or raw.fork_x is None:
        require(len(other.actions) == len(raw.actions), "Different paired terminal steps")
    # Include the pre-action proposal/observation at the fork when it was reached.
    plans = len(raw.proposals) if complete else (steps + 4) // 5 + int(raw.fork_x is not None)
    require(raw.reset["admission"]["raw_state_sha256"] == other.reset["admission"]["raw_state_sha256"] and
            raw.observations[:plans] == other.observations[:plans] and
            np.array_equal(raw.proposals[:plans], other.proposals[:plans]) and
            ((raw.fork_x is None and other.fork_x is None) or np.array_equal(raw.fork_x, other.fork_x)),
            "Paired opening, public replans or fork proposal differ")
    for field in ("actions", "rewards", "successes", "terminated", "truncated"):
        require(np.array_equal(getattr(raw, field)[:steps], getattr(other, field)[:steps]),
                f"Different paired {field} across {steps} native controls")


def training_arrays(row, arm, raw, teacher):
    import model
    step = len(teacher.actions)
    controls = np.asarray(teacher.actions[FORK:], np.float32)
    n = len(teacher.features)
    require(n == (max(0, step - FORK) + 4) // 5, "Missing teacher features")
    targets = np.zeros((n, 5, 7), np.float32)
    if n:
        targets.reshape(-1, 7)[:len(controls)] = controls
    lengths = np.minimum(5, len(controls) - 5 * np.arange(n, dtype=np.int32))
    return dict(split="train", arm=arm, layout_id=row["layout_id"], layout_seed=row["layout_seed"],
                raw_success=bool(raw.successes[-1]), teacher_success=bool(teacher.successes[-1]),
                decision_reached=raw.fork_x is not None,
                fork_x=np.asarray(raw.fork_x if raw.fork_x is not None else [], np.float32),
                suffix_x=np.asarray(teacher.features, np.float32).reshape(n, model.DIM), teacher_actions=targets,
                base_chunks=np.asarray(teacher.proposals[FORK // 5:], np.float32).reshape(n, 5, 7),
                valid_lengths=lengths)


def archived_reference(args, row, arm, off, report):
    source = report[arm][row["layout_seed"]]
    require(all(source[k] == row[k] for k in ("layout_id", "layout_seed", "scene_hash", "movable_xy_hash"))
            and source["arm"] == arm, "Archived result has a different arm or layout identity")
    result = dict(layout_match=True, success=source["success"], native_return=source["native_return"],
                  steps=source["steps"], tape_sha256=source["tape_sha256"],
                  outcome_match=source["success"] == off.successes[-1] and
                  source["native_return"] == sum(off.rewards), steps_match=source["steps"] == len(off.actions))
    tape = args.panel / "dev/scores" / CHECKPOINT / "stride-5" / source["tape"]
    if tape.is_file():
        require(sha(tape) == source["tape_sha256"], "Archived native tape changed")
        with np.load(tape, allow_pickle=False) as data:
            result.update(issued_actions_match=np.array_equal(data["actions"], off.actions),
                          rewards_match=np.array_equal(data["rewards"], off.rewards),
                          opening_observation_match=json.loads(data["observations"].item())[0] == off.observations[0])
    else:
        result["tape_available"] = False
    return result


def save_pair(args, dest, row, arm, cases, contract, archive):
    off = cases["off"]
    for name, case in cases.items():
        if name != "off":
            compare(off, case)
    arrays = {}
    result = dict(arm=arm, layout_id=row["layout_id"], layout_seed=row["layout_seed"],
                  donor_sha256=row["initial_state_sha256"], scene_hash=row["scene_hash"],
                  contract_sha256=contract, prefix_controls=min(FORK, len(off.actions)),
                  branches={name: case.result() for name, case in cases.items()})
    if "teacher" in cases:
        arrays.update(training_arrays(row, arm, off, cases["teacher"]))
    else:
        selected = "always" if cases["gated"].enabled else "off"
        compare(cases[selected], cases["gated"], complete=True)
        result["gated_reproduces"] = selected
        if archive is not None:
            result["archived_raw"] = archived_reference(args, row, arm, off, archive)
    for name, case in cases.items():
        for field, values in case.evidence().items():
            arrays[f"trace_{name}_{field}"] = values
        arrays[f"trace_{name}_reset"] = json.dumps(case.reset, allow_nan=False, sort_keys=True)
    tape = dest.with_suffix(".npz")
    with publish(tape) as tmp, tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    result["tape_sha256"] = sha(tape)
    bind(dest.with_suffix(".json"), result)
    print(json.dumps(dict(arm=arm, seed=row["layout_seed"], **{
        k: int(v.successes[-1]) for k, v in cases.items()})), flush=True)


def run_stage(args, motor, prompt, rows, out, split, policy=None, archive=None, model_dir=None):
    collecting = policy is None
    branch_names = ("off", "teacher") if collecting else ("off", "always", "gated")
    base = args.out / "layouts" if split == "train" else args.panel / "dev"
    dest = out / ("train" if collecting else "eval")
    contract = dict(checkpoint_sha256=CHECKPOINT, fit_sha256=FIT_SHA, r2_revision=PIN, split=split,
                    motor_prompt=prompt, fork=FORK, cadence=5, horizon=300, inference_batch=BATCH,
                    manifest_sha256=sha(base / "manifest.json"), arms=list(ARMS), branches=list(branch_names),
                    seeds=[r["layout_seed"] for r in rows], model_sha256=None if collecting else sha(model_dir / "model.npz"),
                    files={f: sha(HERE / f) for f in ("native.py", "native_sim.py", "teacher.py", "model.py")})
    bind(dest / "contract.json", contract)
    identity = digest(contract)
    results = []
    for arm in ARMS:
        pending = []
        for row in rows:
            path = dest / arm / str(row["layout_seed"])
            if path.with_suffix(".json").exists():
                receipt = read(path.with_suffix(".json"))
                require(receipt["contract_sha256"] == identity and
                        receipt["tape_sha256"] == sha(path.with_suffix(".npz")), "Stale or changed native pair")
            else:
                pending.append(row)
        if pending:
            with ExitStack() as stack:
                count = min(args.workers, len(pending))
                sessions = [stack.enter_context(RenderedSession(base, arm, split)) for _ in range(count)]
                pool = stack.enter_context(ThreadPoolExecutor(max_workers=count))
                for i in range(0, len(pending), count):
                    group = pending[i:i + count]
                    branches = {mode: rollout(pool, sessions, group, mode, motor, policy, prompt)
                                for mode in branch_names}
                    for j, row in enumerate(group):
                        cases = {mode: branch[j] for mode, branch in branches.items()}
                        save_pair(args, dest / arm / str(row["layout_seed"]), row, arm, cases, identity, archive)
        results.extend(read(dest / arm / f"{r['layout_seed']}.json") for r in rows)
    return results


def fit_rows(out, rows, model_dir):
    import model
    root = out / "train"
    contract = read(root / "contract.json")
    require(contract["split"] == "train" and contract["branches"] == ["off", "teacher"] and
            contract["seeds"] == [r["layout_seed"] for r in rows] and contract["checkpoint_sha256"] == CHECKPOINT,
            "Fit inputs are not the registered collection")
    pairs = []
    for arm in ARMS:
        for row in rows:
            base = root / arm / str(row["layout_seed"])
            receipt = read(base.with_suffix(".json"))
            path = base.with_suffix(".npz")
            require(receipt["tape_sha256"] == sha(path) and receipt["contract_sha256"] == digest(contract),
                    "Training pair changed")
            with np.load(path, allow_pickle=False) as data:
                scalars = ("split", "arm", "layout_id", "layout_seed", "raw_success", "teacher_success", "decision_reached")
                pair = {k: data[k].item() for k in scalars}
                require((pair["split"], pair["arm"], pair["layout_id"], pair["layout_seed"]) ==
                        ("train", arm, row["layout_id"], row["layout_seed"]), "Wrong training pair identity")
                for k in ("suffix_x", "base_chunks", "teacher_actions", "valid_lengths"):
                    pair[k] = data[k]
                pair["fork_x"] = data["fork_x"] if pair["decision_reached"] else None
                pairs.append(pair)
    model.fit(pairs, model_dir)
    return model.load(model_dir)


def summary(results):
    arms = {}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        arms[arm] = data = dict(total=len(rows))
        for name in ("off", "always", "gated"):
            data[name] = sum(r["branches"][name]["success"] for r in rows)
        for a, b in (("always", "off"), ("gated", "off"), ("gated", "always")):
            change = [int(r["branches"][a]["success"]) - int(r["branches"][b]["success"]) for r in rows]
            data[f"{a}_vs_{b}"] = dict(gains=change.count(1), losses=change.count(-1), ties=change.count(0))
        data["gate_open"] = sum(r["branches"]["gated"]["enabled"] for r in rows)
        data["decision_reached"] = sum(r["branches"]["gated"]["decision_reached"] for r in rows)
        rescues = [r for r in rows if r["branches"]["always"]["success"] and not r["branches"]["off"]["success"]]
        losses = [r for r in rows if r["branches"]["off"]["success"] and not r["branches"]["always"]["success"]]
        data["choice"] = dict(available_rescues=len(rescues), rescues_taken=sum(
            r["branches"]["gated"]["success"] for r in rescues), available_losses=len(losses),
            losses_avoided=sum(r["branches"]["gated"]["success"] for r in losses),
            both_succeed=sum(r["branches"]["off"]["success"] and r["branches"]["always"]["success"] for r in rows),
            both_fail=sum(not r["branches"]["off"]["success"] and not r["branches"]["always"]["success"] for r in rows),
            hindsight_successes=data["off"] + len(rescues))
        if "archived_raw" in rows[0]:
            data["archived_raw_outcome_mismatch_seeds"] = [r["layout_seed"] for r in rows
                                                            if not r["archived_raw"]["outcome_match"]]
            data["archived_raw_steps_mismatch_seeds"] = [r["layout_seed"] for r in rows
                                                          if not r["archived_raw"]["steps_match"]]
            data["archived_raw_action_matches"] = sum(r["archived_raw"].get("issued_actions_match") is True
                                                       for r in rows)
    return dict(arms=arms, episodes=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "smoke", "collect", "fit", "evaluate"):
        p = commands.add_parser(command)
        p.add_argument("--out", required=True, type=Path)
        if command != "fit":
            p.add_argument("--fit", required=True, type=Path)
            p.add_argument("--panel", required=True, type=Path)
        if command in ("smoke", "collect", "evaluate"):
            p.add_argument("--weights", required=True, type=Path)
            p.add_argument("--workers", type=int, default=BATCH, choices=range(1, BATCH + 1))
        if command in ("fit", "evaluate"):
            p.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    rows = train_rows(args.out)
    if args.command == "fit":
        require(len(rows) == N_TRAIN, "Incomplete production fitting cohort")
        fit_rows(args.out, rows, args.model)
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import model
    data = dev_panel(args)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    args.step = 6400
    contract, checkpoint = candidate(args.fit, args.step)
    require(contract["checkpoint_sha256"] == CHECKPOINT and contract["fit_id"] == FIT_SHA and
            contract["architecture"] == "chunk_huber_v1", "Wrong frozen Huber motor")
    motor = load_motor(args, checkpoint, contract)
    prompt = contract["prompt"]
    if args.command == "collect":
        run_stage(args, motor, prompt, rows, args.out, "train")
    elif args.command == "smoke":
        pilot = args.out / "pilot"
        run_stage(args, motor, prompt, rows[:1], pilot, "train")
        fitted = pilot / "fitted"
        policy = fit_rows(pilot, rows[:1], fitted)
        cases = run_stage(args, motor, prompt, rows[1:2], pilot, "train", policy, model_dir=fitted)
        bind(pilot / "evaluation.json", summary(cases))
    else:
        policy = model.load(args.model)
        path = args.panel / "dev/scores" / CHECKPOINT / "stride-5/report.json"
        require(sha(path) == REPORT_SHA, "Wrong archived native score report")
        report = read(path)
        archive = {a: {r["layout_seed"]: r for r in report["arms"][a]["episodes"]} for a in ARMS}
        require(all(set(archive[a]) == set(DEV) for a in ARMS), "Incomplete native reference")
        cases = run_stage(args, motor, prompt, data["layouts"], args.out, "dev", policy, archive, args.model)
        result = summary(cases)
        result.update(checkpoint_sha256=CHECKPOINT, model_sha256=sha(args.model / "model.npz"),
                      panel_sha256=PANEL_SHA, archived_report_sha256=REPORT_SHA)
        bind(args.out / "evaluation.json", result)
        print(json.dumps(result["arms"]), flush=True)


if __name__ == "__main__":
    main()
