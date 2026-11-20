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
        self.live_observations = []
        self.policy_packets = []
        self.snapshots = [reset["control_zero_snapshot"]] if "control_zero_snapshot" in reset else []
        self.integration = []
        self.precommit = None
        self.proposals = []
        self.proposal_source = []
        self.features = []
        self.fork_x = None
        self.score = None
        self.enabled = False

    def command(self, public, raw, policy, hashes=None, source="live", dispatch=None):
        import model
        step = len(self.actions)
        self.observations.append(hashes if hashes is not None else observation_hashes(public))
        self.proposal_source.append(source)
        self.policy_packets.append(public)
        self.proposals.append(raw.copy())
        if step == FORK:
            self.fork_x = model.features(public, raw, step)
            if self.mode == "off" and policy is not None:
                self.precommit = float(policy.score(public, raw, step))
                require(np.isfinite(self.precommit), "Nonfinite public gate precommitment")
            if self.mode == "gated":
                self.score = float(dispatch) if dispatch is not None else float(policy.score(public, raw, step))
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
        for j, (action, reward, terminated, truncated, success, index, integration) in enumerate(trace):
            self.integration.append(integration)
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
                    live_observations=np.asarray(json.dumps(self.live_observations, sort_keys=True)),
                    physical_snapshots=np.asarray(json.dumps(self.snapshots, sort_keys=True)),
                    integration_sha256=np.asarray(self.integration), proposal_source=np.asarray(self.proposal_source),
                    native_steps=np.arange(1, len(self.actions) + 1, dtype=np.int32),
                    rewards=np.asarray(self.rewards, np.float32), success=np.asarray(self.successes, bool),
                    terminated=np.asarray(self.terminated, bool), truncated=np.asarray(self.truncated, bool))

    def result(self):
        return dict(success=bool(self.successes[-1]), native_return=float(sum(self.rewards)),
                    steps=len(self.actions), decision_reached=self.fork_x is not None,
                    gate_score=self.score, enabled=bool(self.enabled))


def reference_at(mode, episode, plan, previous, i):
    # Branch selection depends only on the public step and the precommitted
    # gate. The physics fingerprints below assert replay identity separately.
    if mode == "off":
        return None
    if plan * 5 <= FORK:
        return previous["off"][i]
    if mode == "gated":
        return previous["always" if episode.enabled else "off"][i]
    return None


def rollout(pool, base, arm, split, rows, mode, motor, policy, prompt, previous, dest):
    import model
    with ExitStack() as stack:
        sessions = [stack.enter_context(RenderedSession(base, arm, split)) for _ in rows]
        resets = [pool.submit(env.start, row["layout_seed"], mode) for env, row in zip(sessions, rows)]
        episodes = [Episode(*f.result(), mode) for f in resets]
        active = list(range(len(episodes)))
        while active:
            public = {i: public_observation(episodes[i].obs, prompt) for i in active}
            fresh = []
            proposals, sources, policy_hashes = {}, {}, {}
            for i in active:
                ep = episodes[i]
                plan = len(ep.proposals)
                live = observation_hashes(public[i])
                ep.live_observations.append(live)
                ref = reference_at(mode, ep, plan, previous, i)
                if ref is None:
                    fresh.append(i)
                    sources[i], policy_hashes[i] = "live", live
                    continue
                if plan >= len(ref.proposals):
                    raise ValueError(f"Physical replay exceeded reference at {arm}/{rows[i]['layout_seed']} control {5*plan}")
                # Only the already measured public packet and frozen proposal
                # from this branch's causal history reach the action function.
                public[i] = ref.policy_packets[plan]
                proposals[i], sources[i], policy_hashes[i] = ref.proposals[plan], ref.mode, ref.observations[plan]
            if fresh:
                chunks = infer(motor, [public[i] for i in fresh])
                proposals.update(zip(fresh, chunks, strict=True))
            jobs = []
            for i in active:
                ep = episodes[i]
                step = len(ep.actions)
                dispatch = previous["off"][i].precommit if mode == "gated" and step == FORK else None
                if mode == "gated" and step == FORK:
                    require(dispatch is not None, "Missing public decision made before the raw suffix")
                command = ep.command(public[i], proposals[i], policy, policy_hashes[i], sources[i], dispatch)
                if step == FORK and ep.precommit is not None:
                    # Persist before the first raw suffix action is sent for this layout.
                    bind(dest / "decisions" / arm / f"{rows[i]['layout_seed']}.json", dict(
                         elapsed_controls=FORK, score=ep.precommit, enabled=ep.precommit > model.GATE_THRESHOLD,
                         threshold=model.GATE_THRESHOLD, layout_seed=rows[i]["layout_seed"], arm=arm,
                         public=ep.observations[-1], raw_proposal=ep.proposals[-1].tolist()))
                jobs.append((i, command, pool.submit(sessions[i].advance_native, command)))
            active = []
            for i, command, f in jobs:
                observation, (trace, stamp) = f.result()
                ep = episodes[i]
                ep.snapshots.append(stamp)
                if not ep.consume(observation, trace, command):
                    active.append(i)
        return episodes


class PairMismatch(ValueError):
    def __init__(self, field, control, left, right, raw, other):
        self.details = dict(reference=raw.mode, branch=other.mode, field=field, elapsed_control=control,
                            expected=left, observed=right, lengths=[len(raw.actions), len(other.actions)])
        super().__init__(json.dumps(self.details))


def compare(raw, other, *, complete=False):
    def same(field, control, a, b):
        if a != b:
            raise PairMismatch(field, control, a, b, raw, other)

    def numeric(field, control, a, b):
        if not np.array_equal(a, b):
            actual, expected = np.asarray(b), np.asarray(a)
            index = tuple(np.argwhere(actual != expected)[0])
            axis = ",".join(str(int(j)) for j in index)
            same(f"{field}[{axis}]", control, expected[index].item(), actual[index].item())

    steps = len(raw.actions) if complete else min(FORK, len(raw.actions))
    if complete or raw.fork_x is None or len(other.actions) < steps:
        same("native length", steps, len(raw.actions), len(other.actions))
    plans = len(raw.proposals) if complete else (steps + 4) // 5 + int(raw.fork_x is not None)
    same("raw_state_sha256", 0, raw.reset["admission"]["raw_state_sha256"], other.reset["admission"]["raw_state_sha256"])
    same("number of compared replans", steps, plans, min(len(other.proposals), plans))
    for j in range(plans):
        for k in ("rgb", "wrist", "state", "prompt"):
            same(k, j * 5, raw.observations[j][k], other.observations[j][k])
        for k in ("state", "prompt"):
            same("live_" + k, j * 5, raw.live_observations[j][k], other.live_observations[j][k])
        numeric("base", j * 5, raw.proposals[j], other.proposals[j])
    for j in range((steps + 4) // 5 + 1):
        for k in ("integration", "controller", "clocks", "public", "full"):
            same("physical_" + k, min(j * 5, steps), raw.snapshots[j][k], other.snapshots[j][k])
    for j in range(steps):
        same("integration_sha256", j + 1, raw.integration[j], other.integration[j])
    same("decision reached", FORK, raw.fork_x is not None, other.fork_x is not None)
    if raw.fork_x is not None:
        numeric("fork_x", FORK, raw.fork_x, other.fork_x)
    for field in ("actions", "rewards", "successes", "terminated", "truncated"):
        left, right = getattr(raw, field), getattr(other, field)
        if not np.array_equal(left[:steps], right[:steps]):
            for i in range(steps):
                numeric(field, i, left[i], right[i])


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
        require(off.precommit == cases["gated"].score, "Gate readback differs from pre-suffix commitment")
        result["precommitted_gate_score"] = off.precommit
        if archive is not None:
            result["archived_raw"] = archived_reference(args, row, arm, off, archive)
    result["camera_replay_differences"] = {name: {
        camera: [j*5 for j, (a, b) in enumerate(zip(case.observations, case.live_observations, strict=True))
                 if a[camera] != b[camera]] for camera in ("rgb", "wrist")} for name, case in cases.items()}
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
                    proposal_reuse="canonical live public packets for the shared raw prefix and selected gated path",
                    fork_qualification="exact native/OSC/clock/public snapshot at every replan and integration each control",
                    gate_qualification="public score committed on the raw rollout before its suffix",
                    simulator_lifetime="fresh interpreter per layout and condition, PYTHONHASHSEED=0",
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
            count = min(args.workers, len(pending))
            with ThreadPoolExecutor(max_workers=count) as pool:
                for i in range(0, len(pending), count):
                    group = pending[i:i + count]
                    branches = {}
                    for mode in branch_names:
                        branches[mode] = rollout(pool, base, arm, split, group, mode, motor, policy, prompt, branches, dest)
                    for j, row in enumerate(group):
                        cases = {mode: branch[j] for mode, branch in branches.items()}
                        path = dest / arm / str(row["layout_seed"])
                        try:
                            save_pair(args, path, row, arm, cases, identity, archive)
                        except PairMismatch as exc:
                            failure = dest / "errors" / arm / str(row["layout_seed"])
                            arrays = {f"trace_{mode}_{k}": v for mode, case in cases.items()
                                      for k, v in case.evidence().items()}
                            for mode, case in cases.items():
                                arrays[f"trace_{mode}_reset"] = json.dumps(case.reset, sort_keys=True)
                                if case.fork_x is not None:
                                    arrays[f"trace_{mode}_fork_x"] = case.fork_x
                            with publish(failure.with_suffix(".npz")) as tmp, tmp.open("wb") as stream:
                                np.savez_compressed(stream, **arrays)
                            bind(failure.with_suffix(".json"), dict(arm=arm, layout_seed=row["layout_seed"],
                                 contract_sha256=identity, tape_sha256=sha(failure.with_suffix(".npz")), **exc.details))
                            raise
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
        run_stage(args, motor, prompt, rows[:BATCH], pilot, "train")
        fitted = pilot / "fitted"
        policy = fit_rows(pilot, rows[:BATCH], fitted)
        cases = run_stage(args, motor, prompt, rows[BATCH:2 * BATCH], pilot, "train", policy, model_dir=fitted)
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
