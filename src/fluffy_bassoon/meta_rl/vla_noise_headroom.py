"""Bounded frozen-pi0.5 constant-noise headroom and repeatability pilot.

One declared source task, two scenes, eight fixed coordinates repeated twice,
and two independent native-noise episodes per scene: 36 real episodes total.
"""

import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time

import numpy as np


PI05_RECEIPT = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"


def source_task(tasks, suite, task_id):
    rows = [row for row in tasks if row["suite"] == suite and row["task_id"] == task_id]
    if (len(rows) != 1 or rows[0]["split"] not in ("train", "validation")
            or not rows[0]["teacher_collection_allowed"]
            or not {0, 1}.issubset(rows[0]["source_state_indices"])):
        raise ValueError("Headroom task must be a declared train/validation source task; no test/excluded tasks")
    return rows[0]


def make_arms(seed):
    rng = np.random.default_rng(seed)
    return np.concatenate((np.zeros((1, 32), np.float32),
        rng.uniform(-1., 1., (7, 32)).astype(np.float32)), axis=0)


def array_digest(value):
    value = np.asarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def trajectory_digest(rows, observations):
    digest = hashlib.sha256()
    for observation in observations:
        for key in ("image", "wrist", "state"):
            digest.update(key.encode() + array_digest(observation[key]).encode())
    for row in rows:
        for key in ("native_initial_noise", "executed_actions", "reward", "raw_rewards", "done", "success", "primitive_steps"):
            digest.update(key.encode() + array_digest(row[key]).encode())
    return digest.hexdigest()


def wilson(successes, count):
    z, fraction = 1.959963984540054, successes / count
    denominator = 1. + z * z / count
    midpoint = (fraction + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / count + z * z / (4 * count * count)) / denominator
    return [max(0., midpoint - radius), min(1., midpoint + radius)]


def analyze(records):
    """Select on repeat zero, evaluate on separately executed repeat one."""
    fixed, native = {}, {}
    for row in records:
        identity = (row["initial_state"], row["repeat"])
        bank, key = (native, identity) if row["kind"] == "native" else (fixed, (*identity, row["arm"]))
        if key in bank:
            raise ValueError("Duplicate headroom episode identity")
        bank[key] = row
    if (set(fixed) != {(state, repeat, arm) for state in (0, 1) for repeat in (0, 1) for arm in range(8)}
            or set(native) != {(state, repeat) for state in (0, 1) for repeat in (0, 1)}):
        raise ValueError("Require all 36 actual completed headroom episodes")
    selected, pairs = [], []
    for state in (0, 1):
        arm = max(range(8), key=lambda a: (int(fixed[state, 0, a]["success"]), -a))
        evaluation = fixed[state, 1, arm]
        selected.append(dict(initial_state=state, selected_arm=arm,
            selection_success=bool(fixed[state, 0, arm]["success"]),
            repeat1_success=bool(evaluation["success"]), zero_repeat1_success=bool(fixed[state, 1, 0]["success"]),
            native_repeat1_success=bool(native[state, 1]["success"]),
            repeat1_primitive_steps=evaluation["primitive_steps"], selection_attempts=8,
            selection_primitive_steps=sum(fixed[state, 0, a]["primitive_steps"] for a in range(8))))
        for arm in range(8):
            a, b = fixed[state, 0, arm], fixed[state, 1, arm]
            pairs.append(dict(initial_state=state, arm=arm, success_matches=a["success"] == b["success"],
                initial_observation_matches=a["initial_observation_sha256"] == b["initial_observation_sha256"],
                trajectory_matches=a["trajectory_sha256"] == b["trajectory_sha256"]))
    global_arm = max(range(8), key=lambda a: (sum(int(fixed[state, 0, a]["success"]) for state in (0, 1)), -a))
    # These replay calculations reuse real complete episodes. No unobserved
    # third repeat or native attempt is fabricated for a larger budget.
    bandit = []
    for budget in range(1, 10):
        trials = []
        for state in (0, 1):
            explored = list(range(budget - 1))
            arm = max(explored, key=lambda a: (int(fixed[state, 0, a]["success"]), -a)) if explored else 0
            episodes = [fixed[state, 0, a] for a in explored] + [fixed[state, 1, arm]]
            trials.append(dict(initial_state=state, selected_arm=arm,
                success_last=bool(episodes[-1]["success"]), success_once=any(r["success"] for r in episodes),
                primitive_steps=sum(r["primitive_steps"] for r in episodes)))
        native_budget = None
        if budget <= 2:
            native_trials = [[native[state, repeat] for repeat in range(2 - budget, 2)] for state in (0, 1)]
            native_budget = dict(success_last=np.mean([r[-1]["success"] for r in native_trials]).item(),
                success_once=np.mean([any(e["success"] for e in r) for r in native_trials]).item(),
                mean_primitive_steps=np.mean([sum(e["primitive_steps"] for e in r) for r in native_trials]).item())
        bandit.append(dict(attempt_budget=budget, explored_arms=budget - 1, trials=trials,
            success_last=np.mean([r["success_last"] for r in trials]).item(),
            success_once=np.mean([r["success_once"] for r in trials]).item(),
            mean_primitive_steps=np.mean([r["primitive_steps"] for r in trials]).item(),
            native_same_attempt_budget=native_budget))
    fixed_search = []
    for budget in range(1, 9):
        trials = [[fixed[state, repeat, arm] for arm in range(budget)]
                  for state in (0, 1) for repeat in (0, 1)]
        fixed_search.append(dict(attempt_budget=budget, cases=4,
            success_last=np.mean([r[-1]["success"] for r in trials]).item(),
            success_once=np.mean([any(e["success"] for e in r) for r in trials]).item(),
            mean_primitive_steps=np.mean([sum(e["primitive_steps"] for e in r) for r in trials]).item()))
    native_successes = sum(int(row["success"]) for row in native.values())
    selected_successes = sum(int(row["repeat1_success"]) for row in selected)
    success_sets = {str(state): [a for a in range(8) if fixed[state, 1, a]["success"]] for state in (0, 1)}
    return dict(completed=True, native=dict(successes=native_successes, episodes=4,
        success_rate=native_successes / 4, descriptive_wilson95=wilson(native_successes, 4)),
        repeat0_selected_repeat1_evaluation=dict(successes=selected_successes, episodes=2,
            success_rate=selected_successes / 2, descriptive_wilson95=wilson(selected_successes, 2),
            gain_over_zero=np.mean([int(r["repeat1_success"]) - int(r["zero_repeat1_success"]) for r in selected]).item(),
            gain_over_native_repeat1=np.mean([int(r["repeat1_success"]) - int(r["native_repeat1_success"]) for r in selected]).item(),
            trials=selected, tie_break="first arm in predeclared zero,uniform1..7 order; no repeat1 labels used"),
        global_constant_arm=dict(selected_on_repeat0=global_arm,
            repeat1_success_rate=np.mean([fixed[state, 1, global_arm]["success"] for state in (0, 1)]).item()),
        repeatability=dict(pairs=pairs, success_matches=sum(r["success_matches"] for r in pairs),
            trajectory_matches=sum(r["trajectory_matches"] for r in pairs), pairs_count=16,
            initial_observation_matches=sum(r["initial_observation_matches"] for r in pairs)),
        heterogeneity=dict(repeat1_success_arm_sets=success_sets,
            common_successful_arms=sorted(set(success_sets["0"]) & set(success_sets["1"])),
            differing_arm_outcomes=sum(fixed[0, 1, a]["success"] != fixed[1, 1, a]["success"] for a in range(8))),
        simple_search_bandit=bandit, fixed_order_search=fixed_search,
        actual_cost=dict(episodes=len(records), primitive_steps=sum(r["primitive_steps"] for r in records),
                         settling_steps=sum(r["settling_steps"] for r in records)),
        limitations=["One task and two scenes; four native episodes and correlated repeated-arm outcomes are a small pilot.",
            "Repeat1 confirms replay on the same initial state and seed; it is not unseen-scene or task generalization.",
            "Wilson intervals are descriptive episode-level intervals, not uncertainty over tasks or independent scenes.",
            "Repeat0 arm selection consumes eight exploration episodes before its repeat1 evaluation; these costs are included.",
            "Search/bandit curves replay finite observed episodes; curve points reuse data and do not add collection interactions.",
            "No native comparison beyond two attempts was measured.",
            "A negative constant-arm result does not rule out chunk- or observation-conditioned DSRL steering.",
            "Ordinary static observation-conditioned predictor inadequacy remains untested; this pilot cannot establish it."])


class RecordingSimulator:
    def __init__(self, sim):
        self.sim, self.metadata, self.observations = sim, sim.metadata, []
    def reset(self, initial_state, scene_seed):
        result = self.sim.reset(initial_state, scene_seed)
        self.observations = [copy.deepcopy(result["observation"])]
        return result
    def step(self, actions):
        result = self.sim.step(actions)
        self.observations.append(copy.deepcopy(result["observation"]))
        return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--frozen-split", type=Path, required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--openpi-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arm-seed", type=int, default=0)
    p.add_argument("--feature-interface", choices=("none", "vla-prefix"), default="none")
    args = p.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run model computation and tests on allocated Linux")
    import os
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    from .vla_cnd_corpus import load_frozen_split
    from .vla_libero import FrozenOpenPI, SimulatorClient, append_json, run_episode, sha256, write_json
    root = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean reviewed source")
    split, tasks = load_frozen_split(args.frozen_split)
    task = source_task(tasks.values(), args.suite, args.task_id)
    if sha256(args.checkpoint / "READY.json") != PI05_RECEIPT:
        raise ValueError("Require the exact verified released pi0.5 checkpoint receipt")
    args.model_config = "pi05_libero"
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "traces").mkdir()
    started, stage, records = time.monotonic(), "initializing", []
    sim = SimulatorClient(args.port)
    try:
        if (sim.metadata["suite"] != args.suite or sim.metadata["task_id"] != args.task_id
                or sim.metadata["bddl_sha256"] != task["bddl_sha256"]
                or Path(sim.metadata["bddl_file"]).stem != task["semantic_task"]
                or sim.metadata["horizon"] != 400
                or sim.metadata["initial_states"] != task["initial_states"]["count"]):
            raise ValueError("Actual simulator differs from the frozen declared task")
        arms = make_arms(args.arm_seed)
        np.save(args.output / "arms.npy", arms, allow_pickle=False)
        plan = []
        for state in (0, 1):
            for repeat in (0, 1):
                noise_seed = int(np.random.SeedSequence([args.seed, state, repeat, 7593]).generate_state(1)[0])
                plan.append(dict(kind="native", arm=None, initial_state=state, repeat=repeat,
                    scene_seed=args.seed + state * 1000, native_noise_seed=noise_seed))
                plan.extend(dict(kind="fixed", arm=arm, initial_state=state, repeat=repeat,
                    scene_seed=args.seed + state * 1000, native_noise_seed=None) for arm in range(8))
        if len(plan) != 36 or len(plan) * 400 > 14400:
            raise ValueError("Headroom episode plan exceeds the predeclared budget")
        contract = dict(source_commit=args.source_commit, task=task, frozen_split_sha256=split["metadata_sha256"],
            checkpoint_receipt_sha256=PI05_RECEIPT, arm_seed=args.arm_seed, arms_sha256=sha256(args.output / "arms.npy"),
            arms=[dict(arm=index, label="zero" if index == 0 else f"uniform-{index}",
                vector=arm.tolist(), array_sha256=array_digest(arm)) for index, arm in enumerate(arms)],
            episodes=plan, max_episodes=36, max_primitive_steps=14400, horizon=400, chunk=10,
            noise="32D Uniform[-1,1] coordinate constant across all decisions in one fixed-arm episode; repeated over native H10",
            native="independent seeded Gaussian [10,32] per decision; identical declared scene state/seed across arms",
            selection="repeat0 chooses first successful arm; repeat1 evaluates it; report exploration cost",
            scope="one source task/two scenes; constant-noise headroom and repeatability; no learned policy or task OOD claim")
        write_json(args.output / "contract.json", contract)
        stage = "motor_qualification"
        motor = FrozenOpenPI(args)
        if motor.horizon != 10:
            raise ValueError("Require native pi0.5 horizon ten")
        obs = sim.reset(0, args.seed + 100000)["observation"]
        zeros = np.zeros((10, 32), np.float32)
        first, second = motor.infer(obs, sim.metadata["instruction"], zeros), motor.infer(obs, sim.metadata["instruction"], zeros.copy())
        delta = float(np.max(np.abs(first - second)))
        write_json(args.output / "motor-noise-parity.json", dict(maximum_delta=delta, passed=delta <= 1e-5, motor=motor.metadata))
        if delta > 1e-5:
            raise RuntimeError("Frozen motor does not reproduce an identical native noise input")
        if motor.feature_encoder is not None:
            write_json(args.output / "initial-feature-parity.json", motor.feature_encoder.qualify(obs, sim.metadata["instruction"]))
        recorder = RecordingSimulator(sim)
        stage = "headroom_episodes"
        for ordinal, identity in enumerate(plan):
            episode_started = time.monotonic()
            arm = np.zeros(32, np.float32) if identity["kind"] == "native" else arms[identity["arm"]]
            rows, summary = run_episode(recorder, motor, lambda _: arm.copy(),
                initial_state=identity["initial_state"], scene_seed=identity["scene_seed"], chunk=10,
                native=identity["kind"] == "native",
                rng=np.random.default_rng(identity["native_noise_seed"]) if identity["kind"] == "native" else None,
                bounded=identity["kind"] != "native", record_features=False)
            if (not rows or not rows[-1]["done"] or len(recorder.observations) != len(rows) + 1
                    or summary["primitive_steps"] > 400):
                raise ValueError("Require a complete real episode and all public observations")
            if identity["kind"] == "fixed" and any(not np.array_equal(row["native_initial_noise"], np.repeat(arm[None], 10, axis=0)) for row in rows):
                raise RuntimeError("The supposedly fixed arm changed during an episode")
            trace = args.output / "traces" / f"episode-{ordinal:03d}.pkl.gz"
            with gzip.open(trace, "xb", compresslevel=1) as stream:
                pickle.dump(dict(plan=identity, summary=summary, rows=rows,
                    public_observations=recorder.observations), stream, protocol=5)
            row = dict(identity, **summary, completed=True, ordinal=ordinal,
                trajectory_sha256=trajectory_digest(rows, recorder.observations), trace_sha256=sha256(trace),
                trace=str(trace.relative_to(args.output)), settling_steps=sim.metadata["settle_steps"],
                elapsed_seconds=time.monotonic() - episode_started)
            records.append(row)
            append_json(args.output / "episodes.jsonl", row)
            progress = dict(completed_episodes=len(records), expected_episodes=36,
                primitive_steps=sum(r["primitive_steps"] for r in records),
                settling_steps=sum(r["settling_steps"] for r in records), last_episode=row)
            if progress["primitive_steps"] > 14400:
                raise RuntimeError("Headroom collection exceeded the explicit primitive-step budget")
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        stage = "analysis"
        result = analyze(records)
        result.update(source_commit=args.source_commit, task=task, frozen_split_sha256=split["metadata_sha256"],
            elapsed_seconds=time.monotonic() - started, motor_calls=motor.calls,
            qualification_reset_settling_steps=sim.metadata["settle_steps"])
        result["actual_cost"].update(
            qualification_settling_steps=sim.metadata["settle_steps"],
            total_settling_steps=result["actual_cost"]["settling_steps"] + sim.metadata["settle_steps"],
            total_simulator_steps=result["actual_cost"]["primitive_steps"]
                + result["actual_cost"]["settling_steps"] + sim.metadata["settle_steps"])
        write_json(args.output / "analysis.json", result)
        write_json(args.output / "summary.json", dict(completed=True, **result["actual_cost"],
            elapsed_seconds=result["elapsed_seconds"], analysis="analysis.json"))
        return 0
    except Exception as error:
        write_json(args.output / "failure.json", dict(stage=stage, error=repr(error),
            completed_episodes=len(records), hostname=socket.gethostname(), elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        sim.close()


if __name__ == "__main__":
    raise SystemExit(main())
