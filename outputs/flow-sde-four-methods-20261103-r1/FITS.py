"""Four independent convergence fits sharing one exclusive allocated GPU.

Each learner is a separate process with its own actor/critic/Adam/sampler/RNG.
The invocation boundary saves full state; only validation plateaus mean done.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assembly", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--attempt", required=True)
    a = p.parse_args()
    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Use the exact allocated Linux container")
    import torch
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    code = Path(__file__).resolve().parents[2]
    checked_checkout(code, a.source_commit)
    if torch.cuda.device_count() != 1:
        raise ValueError("All four independent learners share exactly one exclusive GPU")
    binding = json.loads(a.assembly.read_text())
    if (binding.get("completed") is not True or not binding.get("student_fit_ready")
            or binding.get("source_TRAIN_controls") != 921600 or binding.get("action_dim") != 600
            or binding.get("source_physical_horizons") != [400]):
        raise ValueError("Require the complete qualified Flow direct-action corpus")
    a.output.mkdir(parents=True, exist_ok=True)
    probe = a.output / ("allocation-" + a.attempt + ".json")
    if probe.exists():
        raise ValueError("Never repeat an allocation attempt identity")
    subprocess.run([sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
        "--expected-count", "1", "--output", str(probe), "--baige-container", "--purpose", "training",
        "--baige-exclusive-training"], check=True, timeout=100)
    processes, completed = {}, {}
    try:
        for objective in ("ad", "dpt", "dit", "ic_iql"):
            directory = a.output / objective
            status = directory / "training-status.json"
            if status.exists() and json.loads(status.read_text()).get("converged"):
                completed[objective] = json.loads(status.read_text())
                continue
            train, validation = [binding["datasets"][objective + "_" + split]
                                 for split in ("train", "validation")]
            for item in (train, validation):
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError("Bound source dataset changed")
            trainer = "history_value_train" if objective in ("dit", "ic_iql") else "vla_history_train"
            cmd = [sys.executable, "-B", "-m", "fluffy_bassoon.meta_rl." + trainer,
                "--train-data", train["path"], "--train-sha256", train["sha256"],
                "--validation-data", validation["path"], "--validation-sha256", validation["sha256"],
                "--output-dir", str(directory), "--mode", "full", "--width", "128", "--layers", "2",
                "--heads", "4", "--context-length", "128", "--backbone", "decoder",
                "--max-episodes-per-window", "12", "--batch-size", "8", "--learning-rate", "0.0003",
                "--eval-every", "100", "--validation-batches", "16", "--seed", "0", "--threads", "4",
                "--device", "cuda:0", "--checkpoint-selection", "validation-nll", "--until-converged",
                "--minimum-updates", "1000", "--validation-patience", "10",
                "--validation-relative-delta", "0.001", "--max-elapsed-seconds", "3000",
                "--source-commit", a.source_commit, "--expected-hostname", a.expected_hostname]
            if objective in ("dit", "ic_iql"):
                cmd += ["--objective", objective]
            if (directory / "latest.pt").exists():
                cmd += ["--resume", str(directory / "latest.pt")]
            log = (a.output / (objective + "-" + a.attempt + ".log")).open("x")
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            processes[objective] = (process, log)
        while any(process.poll() is None for process, _ in processes.values()):
            progress = {}
            for objective in ("ad", "dpt", "dit", "ic_iql"):
                status = a.output / objective / "training-status.json"
                progress[objective] = json.loads(status.read_text()) if status.exists() else dict(status="starting")
            temporary = a.output / "PROGRESS.json.tmp"
            temporary.write_text(json.dumps(dict(checked_unix=time.time(), methods=progress), indent=2) + "\n")
            temporary.replace(a.output / "PROGRESS.json")
            failures = {name: process.returncode for name, (process, _) in processes.items()
                        if process.poll() is not None and process.returncode != 0}
            if failures:
                raise RuntimeError("Learner failure; preserve siblings and complete recovery: " + repr(failures))
            time.sleep(5)
        for objective, (process, _) in processes.items():
            if process.returncode != 0:
                raise RuntimeError("Learner failed: " + objective)
            completed[objective] = json.loads((a.output / objective / "training-status.json").read_text())
        result = dict(attempt=a.attempt, source_commit=a.source_commit,
            source_assembly=dict(path=str(a.assembly), sha256=file_sha256(a.assembly)),
            methods=completed, all_converged=all(row.get("converged") for row in completed.values()),
            candidate_checkpoints={method: dict(path=str(a.output / method / "candidate.pt"),
                sha256=file_sha256(a.output / method / "candidate.pt")) for method in completed})
        (a.output / ("RESULT-" + a.attempt + ".json")).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        # Learners independently checkpoint at <=3000sec. On a sibling failure
        # wait for those complete boundaries, rather than killing valid fits.
        for process, log in processes.values():
            process.wait(); log.close()


if __name__ == "__main__":
    main()
