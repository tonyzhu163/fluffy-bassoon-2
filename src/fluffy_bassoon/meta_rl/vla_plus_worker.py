"""Disjoint predeclared LIBERO-Plus development and source cases; local RPC.

Uses official virtual case names and wrapper perturbations. Does not import the
expanded benchmark task registry, construct new perturbations, or load a motor.
"""
import argparse
import ast
import hashlib
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time

import numpy as np

from .vla_libero import AUTHKEY, attest_process_gpu, public_observation, sha256, write_json


PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
TASK_MAP_SHA256 = "41640d2f542169b400be7e94249e3b1f862e7500745dc318bab49b19576b31ac"
BASE_NAME = "pick_up_the_cream_cheese_and_place_it_in_the_basket"
BASE_BDDL_SHA256 = "7019f37ee158d67a21338a6df0c441dd8f84979b946b5e20bb49463ef0508ea2"
BASE_INIT_SHA256 = "5ea5483bca8abd80a8f1935e2c60570d60eaaf6a985c283d62805b46fb76aa94"
CASES = {
    667: dict(classification_id=668, category="Camera Viewpoints", difficulty_level=1,
              name=BASE_NAME + "_view_0_0_115_0_0_initstate_0"),
    290: dict(classification_id=291, category="Robot Initial States", difficulty_level=2,
              name=BASE_NAME + "_view_0_0_100_0_0_initstate_2"),
}
SOURCE_TRAINING_CASES = {
    2387: dict(classification_id=2388, category="Light Conditions", difficulty_level=5,
               name=BASE_NAME + "_light_11"),
    1438: dict(classification_id=1439, category="Sensor Noise", difficulty_level=2,
               name=BASE_NAME + "_view_0_0_100_0_0_initstate_0_noise_1"),
}
LIGHT_BDDL_SHA256 = "59caa81c79b99659143b2880107ae8ed315d836ecf4efe3fb49602c8e9ad06b2"


def case_contract(root, case_index, *, role="development"):
    root = Path(root)
    allowed = CASES if role == "development" else SOURCE_TRAINING_CASES if role == "source_training" else {}
    if case_index not in allowed:
        raise ValueError("Only the cases predeclared for this explicit development/source-training role are allowed")
    folder = root / "libero/libero/benchmark"
    classification, task_map = folder / "task_classification.json", folder / "libero_suite_task_map.py"
    if sha256(classification) != CLASSIFICATION_SHA256 or sha256(task_map) != TASK_MAP_SHA256:
        raise ValueError("Plus classification/task-map bytes changed")
    rows = json.loads(classification.read_text())["libero_object"]
    mapping = ast.literal_eval(ast.parse(task_map.read_text()).body[0].value)["libero_object"]
    selected = [row for row in rows if row["id"] == case_index + 1]
    expected = allowed[case_index]
    if (len(selected) != 1 or selected[0] != dict(id=expected["classification_id"],
            name=expected["name"], category=expected["category"], difficulty_level=expected["difficulty_level"])
            or mapping[case_index] != expected["name"]):
        raise ValueError("Official one-based classification ID and zero-based case name differ")
    base = root / "libero/libero"
    bddl = base / "bddl_files/libero_object" / (BASE_NAME + ".bddl")
    init = base / "init_files/libero_object" / (BASE_NAME + ".pruned_init")
    if sha256(bddl) != BASE_BDDL_SHA256 or sha256(init) != BASE_INIT_SHA256:
        raise ValueError("Pinned Plus base task or initial states changed")
    resolved_name = expected["name"] if case_index == 2387 else BASE_NAME
    resolved_sha = LIGHT_BDDL_SHA256 if case_index == 2387 else BASE_BDDL_SHA256
    if sha256(base / "bddl_files/libero_object" / (resolved_name + ".bddl")) != resolved_sha:
        raise ValueError("Pinned Plus perturbation BDDL changed")
    return dict(benchmark="LIBERO-Plus", split=role, role=role, runtime_index=case_index,
        suite="libero_object", case_id=f"plus:libero_object:{case_index}",
        task_order_index=0, **expected, source_commit=PLUS_COMMIT,
        resolved_bddl_name=resolved_name, resolved_bddl_sha256=resolved_sha,
        classification_sha256=CLASSIFICATION_SHA256, task_map_sha256=TASK_MAP_SHA256,
        original_base=dict(suite="libero_object", task_id=1, semantic_task=BASE_NAME,
            bddl_sha256=BASE_BDDL_SHA256, init_sha256=BASE_INIT_SHA256),
        attachment_train_lineage=dict(task_key="libero_90:47",
            family="full-goal-family-b2a798f45b2c57e3",
            goal_binding_sha256="0e9f8a4e7474f0b0de50e8ce9331758513c2bcde3d6ebabe71db6c8eb600441d"),
        scope=("Development perturbations of a training-goal family; not final external-test results"
               if role == "development" else "Approved Plus source-training perturbations; excluded from validation/final tests"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--case-index", type=int, choices=tuple(CASES) + tuple(SOURCE_TRAINING_CASES), required=True)
    parser.add_argument("--source-training", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--horizon", type=int, default=280)
    parser.add_argument("--settle-steps", type=int, default=10)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or args.horizon < 1 or args.settle_steps < 0:
        raise ValueError("Simulator work requires allocated Linux and a positive episode horizon")
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get("FLUFFY_GPU_EXECUTION", "baige"), os.environ)
    source = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()):
        raise ValueError("Require exact clean simulator source")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.libero_root, text=True).strip()
    if actual != PLUS_COMMIT:
        raise ValueError("Require the pinned staged LIBERO-Plus source")
    case = case_contract(args.libero_root, args.case_index,
                         role="source_training" if args.source_training else "development")
    args.output.mkdir(parents=True, exist_ok=False)
    config = args.output / "libero-config"
    config.mkdir()
    base = args.libero_root / "libero/libero"
    (config / "config.yaml").write_text(json.dumps(dict(benchmark_root=str(base),
        bddl_files=str(base / "bddl_files"), init_states=str(base / "init_files"),
        datasets=str(args.libero_root / "datasets"), assets=str(base / "assets"))))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    started = time.monotonic()
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    states = torch.load(base / "init_files/libero_object" / (BASE_NAME + ".pruned_init"), weights_only=False)
    if len(states) != 50:
        raise ValueError("Pinned Plus base initial-state count differs from fifty")
    # The official wrapper resolves _view_/_initstate_ virtual names itself.
    virtual = base / "bddl_files/libero_object" / (case["name"] + ".bddl")
    instruction = " ".join(re.search(r"\(:language\s+([^)]*)\)",
        (base / "bddl_files/libero_object" / (case["resolved_bddl_name"] + ".bddl")).read_text()).group(1).split())
    metadata = dict(benchmark_kind="plus", development=not args.source_training,
        source_training=args.source_training, role=case["role"],
        origin_base=dict(case["original_base"], train_task_key="libero_90:47"),
        suite="libero_object", task_id=args.case_index, instruction=instruction,
        libero_commit=PLUS_COMMIT, initial_states=len(states), bddl_sha256=case["resolved_bddl_sha256"],
        bddl_file=virtual.name, problem_folder="libero_object", horizon=args.horizon,
        settle_steps=args.settle_steps, pid=os.getpid(), hostname=socket.gethostname(), plus_case=case,
        observation="two RGB cameras and eight robot proprioception values only",
        protocol="Declared object horizon280/settle10 from pinned RLinf standalone OpenPI libero_eval.py; not an asserted full Plus leaderboard protocol",
        fresh_environment_on_reset=True,
        reset_protocol="fresh environment per reset; seed before constructor and reset; actual initial observations hashed")
    listener = Listener(("127.0.0.1", args.port), authkey=AUTHKEY)
    write_json(args.output / "simulator-ready.json", metadata)
    print(json.dumps(dict(ready=True, case=case, elapsed_seconds=time.monotonic() - started)), flush=True)
    connection = listener.accept()
    env, raw, count, success, renderer_verified = None, None, 0, False, False
    try:
        while True:
            request = connection.recv()
            if request["command"] == "metadata":
                connection.send(metadata)
            elif request["command"] == "reset":
                index, seed = int(request["initial_state"]), int(request["seed"])
                if not 0 <= index < len(states):
                    raise ValueError("Invalid pinned Plus initial-state index")
                np.random.seed(seed)
                random.seed(seed)
                if env is not None:
                    env.close()
                env = OffScreenRenderEnv(bddl_file_name=str(virtual), camera_heights=256,
                    camera_widths=256, render_gpu_device_id=0)
                env.seed(seed)
                env.reset()
                raw = env.set_init_state(states[index])
                for _ in range(args.settle_steps):
                    raw, _, _, _ = env.step([0.] * 6 + [-1.])
                if not renderer_verified:
                    attest_process_gpu(args.output / "renderer-binding.json", graphics=True)
                    renderer_verified = True
                count, success = 0, False
                obs = public_observation(raw)
                receipt = dict(initial_state=index, seed=seed,
                    initial_state_vector_sha256=hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest(),
                    initial_observation_sha256=hashlib.sha256(b"".join(obs[k].tobytes()
                        for k in ("image", "wrist", "state"))).hexdigest())
                write_json(args.output / "last-reset.json", receipt)
                connection.send(dict(observation=obs, **receipt))
            elif request["command"] == "step":
                actions = np.asarray(request["actions"], np.float32)
                if raw is None or count >= args.horizon or success:
                    raise RuntimeError("Reset before stepping a completed Plus episode")
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError("Invalid decoded action chunk")
                rewards, executed = [], []
                for action in actions[:args.horizon - count]:
                    raw, reward, done, _ = env.step(action.tolist())
                    count += 1
                    rewards.append(float(reward))
                    executed.append(action)
                    success |= bool(done)
                    if done:
                        break
                connection.send(dict(observation=public_observation(raw), rewards=rewards,
                    executed_actions=np.asarray(executed), primitive_steps=len(executed),
                    episode_steps=count, success=success, done=success or count >= args.horizon))
            elif request["command"] == "close":
                connection.send(dict(closed=True))
                break
            else:
                raise ValueError("Unknown simulator command")
    finally:
        if env is not None:
            env.close()
        connection.close()
        listener.close()


if __name__ == "__main__":
    main()
