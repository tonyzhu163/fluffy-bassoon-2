#!/usr/bin/env python3
"""Four independent meta-learner fits (AD, DPT, DIT, IC-IQL) on a 130-update 10M Flow-SDE corpus.

Each learner runs as an independent process with its own actor/critic/Adam/sampler/RNG,
consuming the 12-chunk-per-episode H600 trajectories with context_length=192
(up to 12 complete episodes per window).
"""
import argparse
import fcntl
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
    p.add_argument("--assembly-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--max-elapsed-seconds", type=int, default=6600)
    a = p.parse_args()

    if sys.platform != "linux" or socket.gethostname() != a.expected_hostname:
        raise ValueError("Require exact allocated Linux container hostname")

    import torch
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    from fluffy_bassoon.meta_rl.vla_history import file_sha256

    code = Path(__file__).resolve().parents[2]
    checked_checkout(code, a.source_commit)
    if torch.cuda.device_count() != 1:
        raise ValueError("All four independent learners share one exclusive allocated GPU")
    if file_sha256(a.assembly) != a.assembly_sha256:
        raise ValueError("Audited source assembly SHA256 mismatch")

    binding = json.loads(a.assembly.read_text())
    if (binding.get("completed") is not True or not binding.get("student_fit_ready")
            or binding.get("source_TRAIN_controls") != 9984000
            or binding.get("action_dim") != 600
            or binding.get("source_physical_horizons") != [600]):
        raise ValueError("Require the complete 130-update H600 Flow corpus assembly")

    a.output.mkdir(parents=True, exist_ok=True)
    master_lock = (a.output / ".fit-master.lock").open("a")
    fcntl.flock(master_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    probe = a.output / ("allocation-" + a.attempt + ".json")
    if probe.exists():
        raise ValueError("Never reuse an allocation attempt identity")
    subprocess.run([
        sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
        "--expected-count", "1", "--output", str(probe),
        "--baige-container", "--purpose", "training", "--baige-exclusive-training",
    ], check=True, timeout=100)

    processes, completed = {}, {}
    try:
        for objective in ("ad", "dpt", "dit", "ic_iql"):
            directory = a.output / objective
            status = directory / "training-status.json"
            if status.exists() and json.loads(status.read_text()).get("converged"):
                completed[objective] = json.loads(status.read_text())
                continue
            train, validation = [
                binding["datasets"][objective + "_" + split]
                for split in ("train", "validation")
            ]
            for item in (train, validation):
                if file_sha256(item["path"]) != item["sha256"]:
                    raise ValueError("Bound source dataset SHA256 changed")
            trainer = "history_value_train" if objective in ("dit", "ic_iql") else "vla_history_train"
            cmd = [
                sys.executable, "-B", "-m", "fluffy_bassoon.meta_rl." + trainer,
                "--train-data", train["path"], "--train-sha256", train["sha256"],
                "--validation-data", validation["path"], "--validation-sha256", validation["sha256"],
                "--output-dir", str(directory),
                "--mode", "full", "--width", "128", "--layers", "2", "--heads", "4",
                "--context-length", "192", "--backbone", "decoder",
                "--max-episodes-per-window", "12", "--batch-size", "8",
                "--learning-rate", "0.0003", "--eval-every", "100", "--validation-batches", "16",
                "--seed", "0", "--threads", "4", "--device", "cuda:0",
                "--checkpoint-selection", "validation-nll", "--until-converged",
                "--minimum-updates", "1000", "--validation-patience", "10",
                "--validation-relative-delta", "0.001",
                "--max-elapsed-seconds", str(a.max_elapsed_seconds),
                "--source-commit", a.source_commit, "--expected-hostname", a.expected_hostname,
            ]
            if objective in ("dit", "ic_iql"):
                cmd += ["--objective", objective]
            if (directory / "latest.pt").exists():
                cmd += ["--resume", str(directory / "latest.pt")]
            log = (a.output / f"{objective}-{a.attempt}.log").open("x")
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            processes[objective] = (process, log)

        while any(process.poll() is None for process, _ in processes.values()):
            progress = {
                obj: json.loads((a.output / obj / "training-status.json").read_text())
                if (a.output / obj / "training-status.json").exists() else dict(status="starting")
                for obj in ("ad", "dpt", "dit", "ic_iql")
            }
            tmp = a.output / "PROGRESS.json.tmp"
            tmp.write_text(json.dumps(dict(checked_unix=time.time(), methods=progress), indent=2) + "\n")
            tmp.replace(a.output / "PROGRESS.json")
            failures = {
                name: proc.returncode for name, (proc, _) in processes.items()
                if proc.poll() is not None and proc.returncode != 0
            }
            if failures:
                raise RuntimeError("Learner failure: " + repr(failures))
            time.sleep(5)

        for objective, (process, _) in processes.items():
            if process.returncode != 0:
                raise RuntimeError("Learner failed: " + objective)
            completed[objective] = json.loads((a.output / objective / "training-status.json").read_text())

        result = dict(
            attempt=a.attempt, source_commit=a.source_commit,
            source_assembly=dict(path=str(a.assembly), sha256=file_sha256(a.assembly)),
            methods=completed,
            all_converged=all(row.get("converged") for row in completed.values()),
            candidate_checkpoints={
                method: dict(
                    path=str(a.output / method / "candidate.pt"),
                    sha256=file_sha256(a.output / method / "candidate.pt"),
                )
                for method in completed if (a.output / method / "candidate.pt").exists()
            },
        )
        (a.output / f"RESULT-{a.attempt}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for process, log in processes.values():
            process.wait()
            log.close()
        (a.output / f"RELEASED-{a.attempt}.json").write_text(json.dumps(dict(
            hostname=a.expected_hostname, source_commit=a.source_commit, ended_unix=time.time(),
            all_children_released=all(process.poll() is not None for process, _ in processes.values()),
            exit_codes={name: process.returncode for name, (process, _) in processes.items()},
        ), indent=2) + "\n")
        master_lock.close()


if __name__ == "__main__":
    main()
