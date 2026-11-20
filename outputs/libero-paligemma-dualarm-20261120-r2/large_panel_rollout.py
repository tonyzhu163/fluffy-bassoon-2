"""Frozen source inference over independent native simulator workers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np

from large_panel_protocol import (ARMS, PUBLIC, bind, check_opening, check_tape, digest, evidence,
                                 inputs, lock, publish, require, sha, training)
from large_panel_sim import Session

HERE = Path(__file__).resolve().parent
sys.path.insert(1, str(HERE.parent / "libero-paligemma-crossarm-20261120-r1"))
BATCH = 8


def load_motor(args, file, contract):
    import torch
    import motor
    import servo_motor
    from waypoint_motor import ARCH, WaypointMotor

    saved = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
    expected, _, _ = training(args.fit)
    require(saved["schema"] == motor.SCHEMA and saved["contract"] == expected and
            saved["fit_id"] == contract["fit_id"] and saved["step"] == saved["best"]["step"] == args.step and
            sha(file) == contract["checkpoint_sha256"], "Checkpoint envelope differs from registered source fit")
    if contract["architecture"] == ARCH:
        model = WaypointMotor(args.weights)
        require(model.identity == expected["weights"], "General PaliGemma snapshot differs from source fit")
        model.restore_delta(saved["best"]["model"])
        model.requires_grad_(False).eval()
    else:
        model = servo_motor.load_selected(saved, args.weights)
    require(not model.training and not any(p.requires_grad for p in model.parameters()), "Motor is not frozen")
    return model


def public_observation(obs, prompt):
    require(set(obs) == PUBLIC and obs["prompt"] == prompt and
            all(obs[k].dtype == np.uint8 and obs[k].shape == (256, 256, 3) for k in ("rgb", "wrist")) and
            obs["state"].shape == (8,) and np.isfinite(obs["state"]).all(), "Invalid public observations")
    return {k: obs[k] for k in ("rgb", "wrist", "state", "prompt")}


def observation_hashes(obs):
    result = {"prompt": obs["prompt"]}
    for k in ("rgb", "wrist", "state"):
        a = np.ascontiguousarray(obs[k])
        h = hashlib.sha256(f"{a.dtype.str}/{a.shape}/".encode())
        h.update(memoryview(a).cast("B"))
        result[k] = h.hexdigest()
    return result


def infer(model, observations):
    import torch
    # Pad the final group to the same kernel batch shape on single-arm shards and retries.
    with torch.inference_mode():
        batch = observations + [observations[-1]] * (BATCH - len(observations))
        output = model(batch).float()
        require(output.shape == (BATCH, 5, 7) and torch.isfinite(output).all().item(), "Invalid action proposal")
        return output[:len(observations)].clamp(-1, 1).cpu().numpy()


def record(stage, arm, row, contract, reset, history):
    path, _ = evidence(stage, arm, row)
    header = dict(contract_sha256=digest(contract), layout_seed=row["layout_seed"],
                  layout_id=row["layout_id"], arm=arm, reset=reset)
    with publish(path) as tmp, tmp.open("wb") as stream:
        np.savez_compressed(stream, header=json.dumps(header, allow_nan=False, sort_keys=True),
                            observations=json.dumps(history["observations"], allow_nan=False),
                            **{k: np.asarray(v) for k, v in history.items() if k != "observations"})
    return check_tape(stage, arm, row, contract, recover=True)


def run_group(pool, sessions, rows, model, stage, arm, contract):
    # Claim layout IDs in ascending order so overlapping shards can safely resume together.
    with ExitStack() as stack:
        active = {}
        for env, row in zip(sessions, rows):
            path, receipt = evidence(stage, arm, row)
            stack.enter_context(lock(path.with_suffix(".lock")))
            if path.exists() or receipt.exists():
                check_tape(stage, arm, row, contract, recover=True)
                continue
            history = {k: [] for k in ("observations", "chunks", "actions", "motion", "done", "success", "rewards", "native_steps")}
            active[env] = (row, history)
        # A receipt precedes each native reset. A process death leaves an identifiable
        # interrupted attempt; a retry still owes this exact arm and layout.
        attempts = {}
        for env, (row, _) in active.items():
            path = stage / "attempts" / arm / str(row["layout_seed"]) / uuid4().hex
            bind(path.with_suffix(".start.json"), dict(arm=arm, layout_id=row["layout_id"],
                 layout_seed=row["layout_seed"], contract_sha256=digest(contract),
                 revision=os.environ.get("MOTOR_CODE_REVISION"), scheduler_host=os.uname().nodename))
            attempts[env] = path
        opening = {env: pool.submit(env.reset, row["layout_id"]) for env, (row, _) in active.items()}
        observations = {env: f.result() for env, f in opening.items()}
        while active:
            envs = list(active)
            obs = [public_observation(observations[env], contract["prompt"]) for env in envs]
            for env, public in zip(envs, obs):
                row, history = active[env]
                hashes = observation_hashes(public)
                if not history["observations"]:
                    check_opening(stage, arm, row, contract, hashes, env.reset_info)
                history["observations"].append(hashes)
            proposals = infer(model, obs)
            pending = []
            for env, chunk in zip(envs, proposals):
                _, history = active[env]
                history["chunks"].append(chunk.copy())
                left = 300 - len(history["actions"])
                require(left > 0, "Native session exceeded horizon")
                prefix = chunk[:min(left, contract["stride"])]
                pending.append((env, prefix, pool.submit(env.advance_native, prefix)))
            for env, prefix, future in pending:
                observations[env], trace = future.result()
                require(0 < len(trace) <= len(prefix), "Missing native steps")
                row, history = active[env]
                for proposed, (motion, reward, done, success, issued, index) in zip(prefix, trace):
                    action = np.asarray(issued, dtype=np.float32)
                    require(index == len(history["actions"]) + 1 and np.array_equal(action, proposed),
                            "Simulator issued a different action or skipped a native step")
                    history["native_steps"].append(index)
                    require(reward in (0, 1) and success == (reward == 1) and (not success or done),
                            "Native reward, terminal and success flags differ")
                    history["actions"].append(action.copy())
                    history["motion"].append(np.asarray(motion, dtype=np.float32))
                    history["done"].append(done)
                    history["success"].append(success)
                    history["rewards"].append(reward)
                if trace[-1][2]:
                    rec = record(stage, arm, row, contract, env.reset_info, history)
                    bind(attempts[env].with_suffix(".complete.json"), dict(tape_sha256=rec["tape_sha256"]))
                    print(json.dumps({k: rec[k] for k in ("arm", "layout_id", "steps", "success")}), flush=True)
                    del active[env]
                else:
                    require(len(trace) == len(prefix), "Truncated nonterminal prefix")


def source(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    require(args.workers <= BATCH, "Choose one to eight independent simulator workers")
    panel, checkpoint, contract, stage = inputs(args)
    bind(stage / "contract.json", contract)
    rows = panel["layouts"][args.shard_index::args.shard_count]
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model = None
    for arm in (ARMS if args.arm == "both" else (args.arm,)):
        pending = []
        for row in rows:
            path, receipt = evidence(stage, arm, row)
            if path.exists() or receipt.exists():
                with lock(path.with_suffix(".lock")):
                    check_tape(stage, arm, row, contract, recover=True)
            else:
                pending.append(row)
        if not pending:
            continue
        if model is None:
            model = load_motor(args, checkpoint, contract)
        with ExitStack() as stack:
            workers = min(args.workers, len(pending))
            sessions = [stack.enter_context(Session(args.root, args.phase, arm)) for _ in range(workers)]
            pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers))
            for i in range(0, len(pending), workers):
                run_group(pool, sessions, pending[i:i+workers], model, stage, arm, contract)
