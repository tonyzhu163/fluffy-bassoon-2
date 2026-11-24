"""Native CPU intervention on identical prefixes of archived frozen pi0.5 programs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "libero-xemb-friction-dev-20261119-r1" / "cpu01"
RELEASE_PIN = "e5500c690c1498eca1b7050350654964190cdf70"
STEPS = 80
CONDITIONS = ("BASE", "BASE_fifo5", "S20")
GAINS = (1.0, 0.6)


def digest(array):
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def replay_tape(args):
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero_remote.environment.libero import LiberoEnvironmentAdapter
    from libero_remote.environment.task_manifest import TaskManifest

    release, name, source = args
    case = source["record"]["case"]
    suite, task, seed = case["suite"], case["task_id"], case["seed"]
    init = benchmark.get_benchmark_dict()[suite]().get_task_init_states(task)
    manifest = TaskManifest.load(str(release / "assets/task_manifest.json"))
    with np.load(SOURCE / f"{name}-ordinary.npz") as trace:
        issued = trace["issued"]
        assert digest(issued) == source["actions_sha256"]
        archive = trace["robot0_eef_pos"][:STEPS + 1]
    assert len(archive) == STEPS + 1 and np.max(np.abs(issued[:STEPS, :3])) <= 1
    states, receipts = {}, []
    for condition in CONDITIONS:
        variant = "S20" if condition == "S20" else "BASE"
        bddl = manifest.resolve_bddl(suite, task, variant)
        bddl_sha = hashlib.sha256(bddl.read_bytes()).hexdigest()
        if variant == "BASE":
            assert bddl_sha == source["bddl_sha256"]
        for gain in GAINS:
            np.random.seed(seed)
            random.seed(seed)
            env = LiberoEnvironmentAdapter()
            env._init_states = init
            env._max_steps = source["record"]["identity"]["horizon"]
            env._env = ControlEnv(str(bddl), use_camera_obs=False, camera_names=[],
                                  has_offscreen_renderer=False)
            try:
                first = env.reset(case["init_state_id"], seed)
                reset = first["environment_reproducibility"]["initial_state_fingerprint"]
                positions = [first["observation"]["robot0_eef_pos"].copy()]
                for t, raw in enumerate(issued[:STEPS]):
                    action = raw.copy()
                    if condition == "BASE_fifo5":
                        action[:3] = issued[t - 5, :3] if t >= 5 else 0
                    action[:3] *= gain
                    step = env.step(action)
                    assert step["worker_step_index"] == t + 1
                    positions.append(step["observation"]["robot0_eef_pos"].copy())
                    if step["terminated"] or step["truncated"]:
                        break
                diag = env._env.env.perturbation_runtime.diagnostics()
                delays = {e["event"]: e["delay_steps"] for e in diag["events"] if "delay_steps" in e}
                assert diag["case_id"] == variant
                if variant == "S20":
                    assert delays["observation_delay"] == delays["fixed_action_delay"] == 7
                else:
                    assert not delays
            finally:
                env.close()
            key = f"{name}_{condition}_{gain:g}"
            positions = np.stack(positions)
            states[key] = positions
            record = dict(key=key, task=f"{suite}/{task}", mode=source["record"]["mode"],
                          condition=condition, gain=gain, controls=len(positions)-1,
                          reset=reset, layout=case["init_state_id"], seed=seed,
                          bddl_sha256=bddl_sha, native_delay_events=delays,
                          source_commands_sha256=source["actions_sha256"])
            if condition == "BASE" and gain == 1:
                record["archive_max_position_difference_m"] = float(np.linalg.norm(
                    positions - archive[:len(positions)], axis=1).max())
                record["archived_physics_match"] = reset == source["record"]["identity"]["physics"]
            receipts.append(record)
    assert len({r["reset"] for r in receipts}) == 1
    print(name, "verified matched resets", flush=True)
    return states, receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    release = args.release.resolve()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert subprocess.check_output(["git", "-C", str(release), "rev-parse", "HEAD"],
                                   text=True).strip() == RELEASE_PIN
    sources = json.loads((SOURCE / "sources.json").read_text())["sources"]
    assert len(sources) == 16
    import sys
    sys.path[:0] = [str(release), str(release / "simulator-source")]
    lib = release / "simulator-source" / "libero" / "libero"
    with tempfile.TemporaryDirectory() as config:
        os.environ["LIBERO_CONFIG_PATH"] = config
        paths = {k:str(lib / v) for k, v in dict(benchmark_root="", bddl_files="bddl_files",
                                                  init_states="init_files", datasets="", assets="assets").items()}
        (Path(config) / "config.yaml").write_text(json.dumps(paths))
        start = datetime.now(timezone.utc)
        arrays, rows = {}, []
        with ProcessPoolExecutor(args.workers) as pool:
            for states, records in pool.map(replay_tape, [(release, k, v) for k, v in sorted(sources.items())]):
                arrays.update(states)
                rows.extend(records)
    receipt = dict(utc_start=start.isoformat(), utc_end=datetime.now(timezone.utc).isoformat(),
                   release_pin=RELEASE_PIN, workers=args.workers, gpu_hours=0, pid=os.getpid(),
                   packages={k:version(k) for k in ("mujoco", "robosuite", "numpy", "torch")},
                   cases=rows)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "wrist.npz", **arrays)
    (args.out / "replay-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
