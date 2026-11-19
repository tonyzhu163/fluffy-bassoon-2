"""Measure native retry outcomes for a frozen two-mode timing decision."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "libero-meta-learning-20261117-r1"
DEMONSTRATIONS = HERE.parent / "libero-pro2-demonstration-pairs-20261118-r1"
sys.path[:0] = [str(CAMPAIGN), str(DEMONSTRATIONS)]
from read_bowls import lag_of_prefix
from reference import CHECKPOINT, observation, write
from simulator import Simulator

MODES = {"default": (5, 0), "paced": (10, 14)}


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def rollout(motor, case, mode, output, prompt=None, expected=None, first_chunk=None):
    output.mkdir()
    simulator = Simulator()
    started = time.monotonic()
    try:
        spec, first = simulator.call("create", case)
        raw = first["observation"]
        released_prompt = first["language_instruction"]
        prompt = released_prompt if prompt is None else prompt
        identity = {
            "physics": first["environment_reproducibility"]["initial_state_fingerprint"],
            "public": {name: digest(value) for name, value in raw.items()},
            "horizon": spec["max_steps"],
            "prompt": prompt,
        }
        if expected is not None and expected != identity:
            raise ValueError(f"Query reset mismatch for {case}")
        rng = np.random.default_rng(case["seed"] + 817117)
        issued, positions, rewards, offsets, waits = [], [], [], [], []
        opening = None
        done = False
        move, hold = MODES[mode]
        while not done:
            offsets.append(len(issued))
            noise = rng.standard_normal((10, 32)).astype(np.float32)
            if opening is None and first_chunk is not None:
                chunk = first_chunk
            else:
                chunk = motor.infer(observation(raw, prompt), noise=noise)["actions"]
            if chunk.shape != (10, 7) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid motor action for {case}")
            if opening is None:
                opening = chunk.copy()
            pause = np.zeros((hold, 7), dtype=chunk.dtype)
            pause[:, -1] = chunk[move - 1, -1]
            for index, action in enumerate(np.concatenate((chunk[:move], pause))):
                transition = simulator.call("step", action)
                raw = transition["observation"]
                issued.append(action.copy())
                positions.append(raw["robot0_eef_pos"].copy())
                rewards.append(float(transition["reward"]))
                waits.append(index >= move)
                done = transition["terminated"] or transition["truncated"]
                if done:
                    break
            if len(issued) >= identity["horizon"] and not done:
                raise ValueError(f"Simulator did not terminate at the horizon for {case}")
        np.savez_compressed(output / "public.npz", issued=issued, positions=positions,
                            rewards=rewards, decision_offsets=offsets, waits=waits, opening=opening)
        record = {
            "case": case, "mode": mode, "success": bool(transition["success"]),
            "controls": len(issued), "waits": sum(waits), "identity": identity,
            "opening_sha256": digest(opening), "released_prompt": released_prompt,
            "seconds": time.monotonic() - started,
        }
        write(output / "result.json", record)
        print(json.dumps({**case, "role": output.name, "success": record["success"],
                          "controls": record["controls"]}), flush=True)
        return record, opening
    finally:
        simulator.close()


def panel(motor, output, splits, anchors):
    rows = []
    prompts = {}
    for suite_index, (suite, tasks) in enumerate(sorted(splits["development"].items())):
        for task_id in tasks:
            key = suite, task_id
            for repeat in range(2):
                base_seed = 19110000 + suite_index * 10000 + task_id * 100 + 3 * repeat
                for variant in ("BASE", "S20"):
                    shared = dict(suite=suite, task_id=task_id, variant=variant)
                    support = dict(shared, init_state_id=3 * repeat, seed=base_seed)
                    query = dict(shared, init_state_id=3 * repeat + 1, seed=base_seed + 1)
                    scenario = output / f"{suite}-{task_id}-r{repeat}-{variant}"
                    scenario.mkdir()
                    earlier, _ = rollout(motor, support, "default", scenario / "support",
                                         prompt=prompts.get(key))
                    prompts.setdefault(key, earlier["identity"]["prompt"])
                    with np.load(scenario / "support" / "public.npz") as tape:
                        lag = lag_of_prefix(tape["issued"], tape["positions"])
                    prefer_default = lag is not None and abs(lag - anchors["BASE"]) < abs(lag - anchors["S20"])
                    selected = "default" if prefer_default else "paced"
                    write(scenario / "decision.json", dict(lag=lag, selected=selected))
                    chosen, opening = rollout(motor, query, selected, scenario / selected, prompt=prompts[key])
                    alternative = "paced" if selected == "default" else "default"
                    other, _ = rollout(motor, query, alternative, scenario / alternative,
                                       prompt=prompts[key], expected=chosen["identity"], first_chunk=opening)
                    rows.append(dict(scenario=scenario.name, lag=lag, selected=selected, support=earlier,
                                     query={selected: chosen, alternative: other}))
                    write(output / "progress.json", rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    splits = json.loads((CAMPAIGN / "SPLITS.json").read_text())
    anchors = json.loads((DEMONSTRATIONS / "transfer.json").read_text())["oracle_medians"]
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected one allocated GPU for inference")
    args.output.mkdir()
    write(args.output / "design.json", dict(splits=splits["development"], lag_anchors=anchors,
                                           checkpoint=str(CHECKPOINT), modes=MODES))
    motor = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(CHECKPOINT))
    rows = panel(motor, args.output, splits, anchors)
    write(args.output / "complete.json", dict(trials=len(rows), episodes=3 * len(rows)))


if __name__ == "__main__":
    main()
