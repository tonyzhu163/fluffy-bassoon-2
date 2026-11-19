"""Compare fixed and timing-conditioned motors on paired native LIBERO resets."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from fluffy_bassoon.libero import estimate_command_lag
from motor import Motor


CAMPAIGN = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1"
sys.path.insert(0, str(CAMPAIGN))
from reference import write
from simulator import Simulator


TIMING = {"BASE": 0, "S20": 1}
FORCED = {"adapter_base": 0, "adapter_s20": 1}
ARMS = ("default", "paced", "solver_zero", *FORCED, "adapter_lag")
PREFIX_CONTROLS = 40
LAG_THRESHOLD = 8
LAG_MAX = 21
NOISE_SEED_OFFSET = 817117


def digest(array):
    array = np.asarray(array)
    digest_prefix = repr((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(digest_prefix + array.tobytes()).hexdigest()


def public_digest(raw):
    return {key: digest(value) for key, value in raw.items()}


def episode(simulator, motor, case, arm, prompt, fallback):
    spec, initial = simulator.call("create", case)
    raw = initial["observation"]
    released_prompt = initial["language_instruction"]
    prompt = released_prompt if prompt is None else prompt
    reset = dict(physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
                 public=public_digest(raw), horizon=spec["max_steps"], prompt=prompt)
    bounds = initial["action_schema"]
    low, high = np.asarray(bounds["low"], np.float32), np.asarray(bounds["high"], np.float32)
    rng = np.random.default_rng(case["seed"] + NOISE_SEED_OFFSET)
    issued, positions, gripper, rewards = [], [], [], []
    clipped = 0
    step = None
    start = time.monotonic()

    def advance(timing, count, waits=0):
        nonlocal raw, step, clipped
        noise = rng.standard_normal((10, 32)).astype(np.float32)
        proposal = np.asarray(motor.proposal(raw, prompt, noise, timing=timing), np.float32)
        if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
            raise ValueError(f"Invalid {arm} proposal on {case}")
        holds = np.zeros((waits, 7), np.float32)
        holds[:, -1] = proposal[count - 1, -1]
        commands = np.concatenate((proposal[:count], holds))
        bounded = np.clip(commands, low, high)
        step = simulator.call("chunk", bounded)
        n = step["steps"]
        if not (0 < n <= len(commands) and len(issued) + n <= reset["horizon"]):
            raise ValueError(f"Invalid simulator steps on {case}")
        if any(len(step[key]) != n for key in ("positions", "gripper_positions", "rewards")):
            raise ValueError(f"Missing control-rate observations on {case}")
        issued.extend(bounded[:n])
        positions.extend(step["positions"])
        gripper.extend(step["gripper_positions"])
        rewards.extend(step["rewards"])
        raw = step["observation"]
        clipped += int(np.count_nonzero(np.any(bounded[:n] != commands[:n], axis=1)))
        return step["terminated"] or step["truncated"]

    done = False
    opening = None
    if arm != "paced":
        while len(issued) < PREFIX_CONTROLS and not done:
            done = advance(None, 5)
        opening = dict(controls=len(issued), issued=digest(issued), positions=digest(positions),
                       gripper=digest(gripper), rewards=digest(rewards), public=public_digest(raw),
                       ended=done)

    timing = -1 if arm == "solver_zero" else FORCED.get(arm)
    lag = None
    if arm == "adapter_lag" and not done:
        lag = estimate_command_lag(issued, positions, positions_timing="after",
                                   max_lag=LAG_MAX, prefix=PREFIX_CONTROLS)
        timing = fallback if lag is None else int(lag >= LAG_THRESHOLD)

    while not done:
        done = advance(timing, 10, 14) if arm == "paced" else advance(timing, 5)

    return dict(arm=arm, reset=reset, opening=opening, released_prompt=released_prompt,
                timing_chosen=timing, lag=lag, controls=len(issued), clipped_controls=clipped,
                success=bool(step["success"]), terminated=bool(step["terminated"]),
                truncated=bool(step["truncated"]), seconds=round(time.monotonic() - start, 3))


def native_cases(tasks, start_ids, seed_base):
    for suite_index, (suite, task_ids) in enumerate(sorted(tasks.items())):
        for task in task_ids:
            for repeat, reset_id in enumerate(start_ids):
                seed = seed_base + 10_000 * suite_index + 100 * task + repeat
                for variant in TIMING:
                    yield dict(suite=suite, task_id=task, variant=variant,
                               init_state_id=reset_id, seed=seed)


def collect(simulator, motor, cases, output, arms, fallback):
    output.mkdir()
    prompts = {}
    rows = []
    for case in cases:
        task = case["suite"], case["task_id"]
        comparison = {arm: episode(simulator, motor, case, arm, prompts.get(task), fallback)
                      for arm in arms}
        first = comparison[arms[0]]
        prompts.setdefault(task, first["reset"]["prompt"])
        if any(row["reset"] != first["reset"] for row in comparison.values()):
            raise ValueError(f"Paired reset changed on {case}")
        if any(comparison[arm]["opening"] != first["opening"] for arm in arms if arm != "paced"):
            raise ValueError(f"Shared opening changed on {case}")
        row = dict(case=case, arms=comparison)
        rows.append(row)
        write(output / "progress.json", rows)
        print(json.dumps(dict(completed=len(rows), case=case, success={
            arm: result["success"] for arm, result in comparison.items()})), flush=True)
    return rows


def select_fixed(rows):
    source_success = {arm: sum(row["arms"][arm]["success"] for row in rows) for arm in FORCED}
    fixed = max(FORCED, key=source_success.__getitem__)
    return fixed, source_success


def summarize(rows, fixed, source_success):
    readout = {}
    for panel in (*TIMING, "overall"):
        group = [row for row in rows if panel == "overall" or row["case"]["variant"] == panel]
        scores = {arm: [row["arms"][arm]["success"] for row in group] for arm in ARMS}
        scores["adapter_true"] = [row["arms"]["adapter_base" if row["case"]["variant"] == "BASE"
                                            else "adapter_s20"]["success"] for row in group]
        scores["adapter_fixed"] = scores[fixed]
        pairs = {}
        for arm, arms in scores.items():
            if arm == "adapter_true":
                continue
            pairs[arm] = dict(true_wins=sum(a and not b for a, b in zip(scores["adapter_true"], arms)),
                              true_losses=sum(b and not a for a, b in zip(scores["adapter_true"], arms)))
        lag_rows = [row for row in group if not row["arms"]["adapter_lag"]["opening"]["ended"]]
        readout[panel] = dict(trials=len(group), successes={arm: sum(xs) for arm, xs in scores.items()},
                              true_vs=pairs, lag_decisions=len(lag_rows),
                              lag_abstentions=sum(row["arms"]["adapter_lag"]["lag"] is None for row in lag_rows),
                              lag_correct=sum(row["arms"]["adapter_lag"]["timing_chosen"] ==
                                              TIMING[row["case"]["variant"]] for row in lag_rows))
    return dict(source_fixed=fixed, source_success=source_success, panels=readout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    args = parser.parse_args()

    import jax

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected one allocated GPU for native evaluation")
    splits = json.loads((CAMPAIGN / "SPLITS.json").read_text())
    source = {suite: task_ids[:1] for suite, task_ids in splits["train"].items()}
    tasks = splits["development"]
    start_ids = (13, 19)
    if args.panel == "smoke":
        source = tasks = {"libero_goal": [4]}
        start_ids = (13,)
    source_cases = list(native_cases(source, (11,), 19601000))
    query_cases = list(native_cases(tasks, start_ids, 19701000))
    args.output.mkdir()
    write(args.output / "design.json", dict(panel=args.panel, checkpoint=str(args.checkpoint),
          source_cases=source_cases, query_cases=query_cases, arms=ARMS, prefix_controls=PREFIX_CONTROLS,
          paced_opening="from reset", noise_seed_offset=NOISE_SEED_OFFSET,
          lag_threshold=LAG_THRESHOLD, lag_max=LAG_MAX,
          prompt="BASE wording in both timing arms", native_relative=True))
    motor = Motor(args.checkpoint)
    with Simulator() as simulator:
        source_rows = collect(simulator, motor, source_cases, args.output / "fitting", tuple(FORCED), None)
        fixed, source_success = select_fixed(source_rows)
        write(args.output / "selected-fixed.json", dict(arm=fixed, source_success=source_success))
        query = collect(simulator, motor, query_cases, args.output / args.panel, ARMS, FORCED[fixed])
    summary = summarize(query, fixed, source_success)
    write(args.output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
