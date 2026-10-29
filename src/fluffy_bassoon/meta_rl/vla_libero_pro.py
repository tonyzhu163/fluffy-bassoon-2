"""Pinned LIBERO-Pro development variants over the allocation-local RPC API.

Only the two declared cream-cheese/basket variants and initial state zero are
admitted. No benchmark enumeration, perturbation generation or test selection
occurs in this worker. Model computation stays in the separate motor process.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import random
import re
import subprocess
import time

import numpy as np

from .vla_libero import AUTHKEY, append_json, attest_process_gpu, public_observation, sha256, write_json


PRO_COMMIT = "eafdb809426b13153aa1e4c42d6601844217dfec"
DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
DATA_RECEIPT = "98a5cae87d4939303f451c955b6dfc478f720ee0fd83fa611b5a771a1478fefe"
ASSET_RECEIPT = "90a08e2df84608921b0f072bce6d902e1657cafbc9b847c9947a2a841f7f3acb"
TASK_STEM = "pick_up_the_cream_cheese_and_place_it_in_the_basket"
CASES = {
    "swap": dict(case_id="pro_object_swap_init0", family="libero_object_swap", dimension="position",
        bddl_sha256="59984ec9fb4b381a614f50f271cacecca73b08d3969ddfff71202063de5aacfd",
        init_sha256="f9bb455a4fd1dac83a72d33673509a2f175f7fb63026b9fef78d66515771e022",
        condition="Official position swap of cream cheese and alphabet soup"),
    "appearance": dict(case_id="pro_object_appearance_init0", family="libero_object_object", dimension="object",
        bddl_sha256="2bdd42a7e444d5d02723901b7e11c2ee1a2964b5a4ba972b495d79fce7a0c4a5",
        init_sha256="0c127f132eb2d4877b01daad923f00df8917c77f0a5a58cb6434cc5e62a553bd",
        condition="Official red cream cheese and red basket appearance variant"),
}
ORIGIN_BASE = dict(suite="libero_object", task_id=1, train_task_key="libero_90:47", split="train",
    semantic_task=TASK_STEM, goal_binding_sha256="0e9f8a4e7474f0b0de50e8ce9331758513c2bcde3d6ebabe71db6c8eb600441d",
    variant_base_family="full-goal-family-b2a798f45b2c57e3",
    lineage="Original base has identical full goal and typed bindings to TRAIN 90:47; appearance variant inherits this base lineage")
SETTLE_STEPS = 5


def declared_files(data_root, case):
    """Validate frozen public data before any deserialization or simulation."""
    if case not in CASES:
        raise ValueError("Only predeclared development Pro cases are allowed")
    data_root = Path(data_root)
    receipt_path = data_root / "READY.json"
    if sha256(receipt_path) != DATA_RECEIPT:
        raise ValueError("Pro dataset receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt["status"] != "verified" or receipt["revision"] != DATA_REVISION
            or receipt["source_commit"] != PRO_COMMIT):
        raise ValueError("Wrong pinned official Pro source/data")
    spec = CASES[case]
    bddl = data_root / "bddl_files" / spec["family"] / f"{TASK_STEM}.bddl"
    states = data_root / "init_files" / spec["family"] / f"{TASK_STEM}.pruned_init"
    if sha256(bddl) != spec["bddl_sha256"] or sha256(states) != spec["init_sha256"]:
        raise ValueError("Declared Pro BDDL or initialization bytes changed")
    return bddl, states, spec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pro-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--horizon", type=int, default=280)
    args = p.parse_args(argv)
    if args.horizon != 280:
        raise ValueError("This Pro development declaration uses official object horizon 280")
    args.output.mkdir(parents=True, exist_ok=False)
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    started = time.monotonic()
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.pro_root, text=True).strip()
    if actual != PRO_COMMIT or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=args.pro_root, text=True).strip():
        raise ValueError("Require the exact clean official Pro source")
    if sha256(args.pro_root.parent / "source-assets.json") != ASSET_RECEIPT:
        raise ValueError("Pro source asset receipt changed")
    bddl, states_path, spec = declared_files(args.data_root, args.case)
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero.envs as envs_module
    import torch
    if not Path(envs_module.__file__).resolve().is_relative_to(args.pro_root.resolve()):
        raise ValueError("Simulator imported a different LIBERO implementation")
    # These exact official files contain NumPy states in a torch archive. They
    # were hash-checked above; no benchmark code is executed by deserialization.
    states = torch.load(str(states_path), map_location="cpu", weights_only=False)
    if len(states) != 50 or np.asarray(states[0]).shape != (110,):
        raise ValueError("The declared Pro initialization bank shape changed")
    language = re.search(r"\(:language\s+([^)]*)\)", bddl.read_text())
    if language is None:
        raise ValueError("Declared BDDL omitted its language instruction")
    metadata = dict(benchmark_kind="pro", development=True, suite=spec["family"], task_id=1,
        case_id=spec["case_id"], dimension=spec["dimension"], condition=spec["condition"],
        pro_case=dict(spec, initial_state=0),
        origin_base=ORIGIN_BASE, instruction=language.group(1).strip(),
        semantic_task=TASK_STEM, libero_commit=PRO_COMMIT, pro_commit=PRO_COMMIT,
        dataset_repo="zhouxueyang/LIBERO-Pro", dataset_revision=DATA_REVISION,
        dataset_receipt_sha256=DATA_RECEIPT, assets_receipt_sha256=ASSET_RECEIPT,
        bddl_file=str(bddl), bddl_sha256=spec["bddl_sha256"],
        init_file=str(states_path), init_sha256=spec["init_sha256"],
        initial_states=len(states), development_initial_state_indices=[0],
        problem_folder=spec["family"], horizon=args.horizon, settle_steps=SETTLE_STEPS,
        settling_action=[0.] * 7,
        settling_basis=f"{PRO_COMMIT}:benchmark_scripts/render_single_task.py:22,66; official rendering utility, development POC",
        horizon_basis=f"{PRO_COMMIT}:README.md TASK_MAX_STEPS LIBERO_OBJECT_* = 280",
        reset="Fresh environment per episode; RNG seeded before construction and reset; official state installed before settling",
        observation="two RGB cameras and robot proprioception only", pid=os.getpid())
    listener = Listener(("127.0.0.1", args.port), authkey=AUTHKEY)
    write_json(args.output / "simulator-ready.json", metadata)
    print(json.dumps(dict(simulator_startup="pro_listener_ready", case=args.case,
        elapsed_seconds=time.monotonic() - started)), flush=True)
    faulthandler.cancel_dump_traceback_later()
    connection, env, raw = listener.accept(), None, None
    count, success, renderer_verified = 0, False, False
    try:
        while True:
            request = connection.recv()
            command = request["command"]
            if command == "metadata":
                connection.send(metadata)
            elif command == "reset":
                index, seed = int(request["initial_state"]), int(request["seed"])
                if index != 0:
                    raise ValueError("Only declared development initialization zero is permitted")
                if env is not None:
                    env.close()
                np.random.seed(seed)
                random.seed(seed)
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
                    camera_widths=256, render_gpu_device_id=0, horizon=1000)
                env.seed(seed)
                env.reset()
                raw = env.set_init_state(states[index])
                for _ in range(SETTLE_STEPS):
                    raw, _, _, _ = env.step([0.] * 7)
                if not renderer_verified:
                    attest_process_gpu(args.output / "renderer-binding.json", graphics=True)
                    renderer_verified = True
                count, success = 0, False
                obs = public_observation(raw)
                digest = hashlib.sha256(b"".join(obs[k].tobytes() for k in ("image", "wrist", "state"))).hexdigest()
                append_json(args.output / "resets.jsonl", dict(initial_state=index, seed=seed,
                    initial_observation_sha256=digest, simulator_state_sha256=hashlib.sha256(
                        np.asarray(env.get_sim_state()).tobytes()).hexdigest()))
                connection.send(dict(observation=obs, initial_state=index, initial_observation_sha256=digest))
            elif command == "step":
                actions = np.asarray(request["actions"], np.float32)
                if raw is None or count >= args.horizon or success:
                    raise RuntimeError("Reset before stepping a completed episode")
                if (actions.ndim != 2 or actions.shape[1] != 7 or not len(actions)
                        or not np.isfinite(actions).all()):
                    raise ValueError("Require a finite decoded LIBERO action chunk")
                rewards, executed = [], []
                for action in actions[:args.horizon - count]:
                    raw, reward, _, _ = env.step(action.tolist())
                    count += 1
                    rewards.append(float(reward))
                    executed.append(action.copy())
                    success |= bool(env.check_success())
                    if success:
                        break
                connection.send(dict(observation=public_observation(raw), rewards=rewards,
                    executed_actions=np.asarray(executed), primitive_steps=len(executed),
                    episode_steps=count, success=success, done=success or count >= args.horizon))
            elif command == "close":
                connection.send(dict(closed=True))
                break
            else:
                raise ValueError("Unknown loopback simulator request")
    finally:
        if env is not None:
            env.close()
        connection.close()
        listener.close()


if __name__ == "__main__":
    main()
