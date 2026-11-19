"""Collect paired native LIBERO outcomes for categorical decision pretraining."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as openpi
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent / "libero-meta-learning-20261117-r1"
sys.path.insert(0, str(CAMPAIGN))
from reference import CHECKPOINT, observation, write
from simulator import Simulator

MODES = ("default", "paced", "short_wait", "gripper_wait")
TIMINGS = ("BASE", "S20")


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


class FrozenMotor:
    def __init__(self):
        self.policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(CHECKPOINT))
        self.embed_prefix = nnx_utils.module_jit(self.policy._model.embed_prefix)

    def query_features(self, raw, prompt):
        item = observation(raw, prompt)
        inputs = self.policy._input_transform(dict(item))
        batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        obs = openpi.preprocess_observation(None, openpi.Observation.from_dict(batch), train=False)
        embeds, mask, _ = self.embed_prefix(obs)
        pooled = jnp.where(mask[..., None], embeds.astype(jnp.float32), 0).sum(axis=1)
        pooled = pooled / mask.sum(axis=1, keepdims=True)
        features = np.concatenate((item["observation/state"], np.asarray(pooled[0]))).astype(np.float32)
        if features.ndim != 1 or not np.isfinite(features).all():
            raise ValueError("Invalid public query features")
        return features

    def proposal(self, raw, prompt, noise):
        actions = self.policy.infer(observation(raw, prompt), noise=noise)["actions"]
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid action proposal")
        return actions


def issued_chunk(proposal, mode, previous_grip):
    count = 5 if mode == "default" else 10
    moves = proposal[:count]
    gripper = [int(np.sign(x)) for x in moves[:, -1] if x != 0]
    if mode == "default":
        hold = 0
    elif mode == "paced":
        hold = 14
    elif mode == "short_wait":
        hold = 7
    elif mode == "gripper_wait":
        changes = [previous_grip, *gripper]
        hold = 14 if previous_grip is None or any(a != b for a, b in zip(changes, changes[1:])) else 7
    else:
        raise ValueError(f"Unknown mode {mode}")
    if gripper:
        previous_grip = gripper[-1]
    waits = np.zeros((hold, 7), dtype=moves.dtype)
    waits[:, -1] = moves[-1, -1]
    return np.concatenate((moves, waits)), count, previous_grip


def reset(simulator, case, prompt=None):
    spec, initial = simulator.call("create", case)
    raw = initial["observation"]
    prompt = initial["language_instruction"] if prompt is None else prompt
    identity = dict(physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
                    public={name: digest(value) for name, value in raw.items()},
                    horizon=spec["max_steps"], prompt=prompt)
    return raw, identity


def rollout(simulator, motor, raw, case, identity, mode, output, opening=None):
    output.mkdir()
    initial_position = raw["robot0_eef_pos"].copy()
    rng = np.random.default_rng(case["seed"] + 817117)
    actions, positions, gripper, rewards, waits = [], [], [], [], []
    previous_grip = None
    done = False
    shared_opening = opening
    while not done:
        noise = rng.standard_normal((10, 32)).astype(np.float32)
        if not actions and shared_opening is not None:
            proposal = shared_opening
        else:
            proposal = motor.proposal(raw, identity["prompt"], noise)
        if not actions:
            opening = proposal.copy()
        commands, moves, previous_grip = issued_chunk(proposal, mode, previous_grip)
        step = simulator.call("chunk", commands)
        n = step["steps"]
        if not (0 < n <= len(commands)) or len(step["gripper_positions"]) != n:
            raise ValueError("Lost a control-rate observation")
        actions.extend(commands[:n])
        positions.extend(step["positions"])
        gripper.extend(step["gripper_positions"])
        rewards.extend(step["rewards"])
        waits.extend(np.arange(n) >= moves)
        raw = step["observation"]
        done = step["terminated"] or step["truncated"]
        if len(actions) >= identity["horizon"] and not done:
            raise ValueError("Missing simulator termination")
    if len(actions) > identity["horizon"]:
        raise ValueError("Exceeded native horizon")
    np.savez_compressed(output / "public.npz", issued=actions, positions=positions,
                        initial_position=initial_position, gripper=gripper, rewards=rewards,
                        waits=waits, opening=opening)
    record = dict(case=case, identity=identity, controls=len(actions), waits=int(sum(waits)),
                  success=bool(step["success"]), opening_sha256=digest(opening))
    write(output / "result.json", record)
    return record


def collect_panel(motor, output, tasks, seed_offset):
    output.mkdir()
    rows = []
    prompt_by_task = {}
    with Simulator() as simulator:
        for suite_index, (suite, task_ids) in enumerate(sorted(tasks.items())):
            for task in task_ids:
                for repeat in range(2):
                    seed = seed_offset + 10000 * suite_index + 100 * task + 3 * repeat
                    for timing in TIMINGS:
                        key = (suite, task)
                        stem = f"{suite}-{task}-r{repeat}-{timing}"
                        folder = output / stem
                        folder.mkdir()
                        support = dict(suite=suite, task_id=task, variant=timing,
                                       init_state_id=3 * repeat, seed=seed)
                        query = dict(support, init_state_id=3 * repeat + 1, seed=seed + 1)
                        raw, support_identity = reset(simulator, support, prompt_by_task.get(key))
                        prompt = prompt_by_task.setdefault(key, support_identity["prompt"])
                        older = rollout(simulator, motor, raw, support, support_identity,
                                        "default", folder / "support")
                        identity = None
                        branches = {}
                        for mode in MODES:
                            raw, current = reset(simulator, query, prompt)
                            if identity is None:
                                identity = current
                                np.save(folder / "query.npy", motor.query_features(raw, prompt), allow_pickle=False)
                                rng = np.random.default_rng(query["seed"] + 817117)
                                opening = motor.proposal(raw, prompt, rng.standard_normal((10, 32)).astype(np.float32))
                            elif identity != current:
                                raise ValueError(f"Reset or public observation changed between branches in {stem}")
                            branches[mode] = rollout(simulator, motor, raw, query, current,
                                                     mode, folder / mode, opening)
                        if len({entry["opening_sha256"] for entry in branches.values()}) != 1:
                            raise ValueError(f"Invalid opening match in {stem}")
                        rows.append(dict(trial=stem, support=older, queries=branches))
                        write(output / "progress.json", rows)
                        print(json.dumps(dict(trial=stem, count=len(rows),
                                              outcomes={k: v["success"] for k, v in branches.items()})), flush=True)
    write(output / "complete.json", dict(trials=len(rows), episodes=len(rows) * (1 + len(MODES))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    splits = json.loads((CAMPAIGN / "SPLITS.json").read_text())
    args.output.mkdir()
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected one allocated GPU for OpenPI")
    write(args.output / "design.json", dict(modes=MODES, timings=TIMINGS, motor=str(CHECKPOINT)))
    motor = FrozenMotor()
    collect_panel(motor, args.output / "development", splits["development"], 19110000)
    collect_panel(motor, args.output / "fitting", splits["development_scaling"]["task_subsets"]["8"], 19120000)
    write(args.output / "complete.json", dict(development=32, fitting=32, episodes=320))


if __name__ == "__main__":
    main()
