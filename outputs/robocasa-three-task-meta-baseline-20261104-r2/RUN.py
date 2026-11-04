#!/usr/bin/env python3
"""Four-GPU corpus preparation or pooled training with checkpointed resumes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=("OpenDrawer", "CloseToasterOvenDoor", "CoffeeSetupMug", "joint"), required=True)
    p.add_argument("--stage", choices=("prepare", "fit"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--files-per-batch", type=int, default=4)
    p.add_argument("--wall-seconds", type=int, default=28800)
    p.add_argument("--prepare-seconds", type=int, default=12600)
    a = p.parse_args()
    if sys.platform != "linux" or a.files_per_batch < 1 or not 300 < a.prepare_seconds < a.wall_seconds - 600:
        raise ValueError("Require allocated Linux and preparation/fit/checkpoint time reserves")
    if (a.task == "joint") != (a.stage == "fit"):
        raise ValueError("Per-task jobs only prepare data; fitting requires the pooled joint task")
    scripts = Path(__file__).resolve().parent
    code = scripts.parents[1]
    # Resource approval and the human launch hold are recorded separately.
    budget = json.loads((scripts / "BUDGET.json").read_text())
    if (budget.get("status") != "approved" or budget.get("approved_ceiling_GPU_hours", 0) < 96
            or budget.get("launch_hold") is True or a.wall_seconds not in (14400, 28800)):
        raise ValueError("Require the approved ceiling and release of the human launch hold")
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout
    source = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    checked_checkout(code, source)
    a.output.mkdir(parents=True, exist_ok=True)
    lock = (a.output / ".task.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = a.output / a.attempt
    attempt.mkdir(exist_ok=False)
    started = time.time()
    deadline = started + a.wall_seconds - 300
    children, logs = [], []

    def interrupted(signum, frame):
        raise TimeoutError("Owned task deadline or scheduler stop; completed evidence retained")

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, interrupted)
    signal.setitimer(signal.ITIMER_REAL, a.wall_seconds - 120)

    def run_one(command, name, env=None):
        log = (attempt / f"{name}.log").open("x")
        logs.append(log)
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(child)
        while child.poll() is None:
            if time.time() >= deadline:
                raise TimeoutError("Allocation deadline")
            time.sleep(1)
        if child.returncode:
            raise RuntimeError(f"{name} failed with exit {child.returncode}; inspect its log")

    status = "failed"
    try:
        probe = attempt / "GPU-PROBE.json"
        run_one([sys.executable, "-B", str(code / "scripts/[train]gpu_probe.py"),
                 "--expected-count", "4", "--output", str(probe), "--baige-container",
                 "--purpose", "training", "--baige-exclusive-training"], "probe")
        devices = json.loads(probe.read_text())["devices"]
        prep = a.output / "preparation"
        if a.stage == "prepare":
            workers = []
            for index, device in enumerate(devices):
                directory = prep / f"shard-{index:02d}"
                # New attempts do not silently overwrite incomplete or completed shards.
                if directory.exists():
                    raise ValueError("Existing preparation requires a diagnosed explicit recovery")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device["visibility_token"],
                           LYCO_BAIGE_GPU_MAP=str(probe), LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
                           OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                cmd = [sys.executable, "-B", str(scripts / "PREPARE-STREAM.py"),
                       "--task", a.task, "--code", str(code), "--source-commit", source,
                       "--output", str(directory), "--expected-hostname", socket.gethostname(),
                       "--max-seconds", str(a.prepare_seconds), "--files-per-batch", str(a.files_per_batch),
                       "--shard-index", str(index), "--allocation-probe", str(probe)]
                log = (attempt / f"prepare-{index:02d}.log").open("x")
                logs.append(log)
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
                workers.append(child)
            while any(child.poll() is None for child in workers):
                if any(child.poll() not in (None, 0) for child in workers):
                    raise RuntimeError("Preparation worker failed; no automatic retry")
                if time.time() >= deadline:
                    raise TimeoutError("Preparation deadline")
                time.sleep(2)
            if any(child.returncode for child in workers):
                raise RuntimeError("Preparation worker failed")
            run_one([sys.executable, "-B", str(scripts / "MERGE.py"), "--output", str(prep),
                     "--task", a.task, "--source-commit", source,
                     "--expected-hostname", socket.gethostname()], "merge")
            status = "prepared"
            return
        if not (prep / "ASSEMBLY.json").exists():
            run_one([sys.executable, "-B", str(scripts / "POOL.py"),
                     "--sources", str(a.output.parent), "--output", str(prep),
                     "--source-commit", source, "--expected-hostname", socket.gethostname()], "pool")
        assembly = prep / "ASSEMBLY.json"
        from fluffy_bassoon.meta_rl.vla_history import file_sha256
        # A fixed per-invocation bound keeps the learners' resume contract exact.
        # Each invocation preserves optimizer, sampler, RNG and plateau counters.
        # Scheduler expiry is an operational checkpoint, never convergence.
        cycle = 0
        status = "operational_checkpoint"
        while deadline - time.time() > 6600 + 300:
            fit_attempt = f"{a.attempt}-cycle{cycle:02d}"
            run_one([sys.executable, "-B", str(scripts / "FITS.py"), "--assembly", str(assembly),
                     "--assembly-sha256", file_sha256(assembly), "--output", str(a.output / "fits"),
                     "--source-commit", source, "--expected-hostname", socket.gethostname(),
                     "--attempt", fit_attempt, "--max-elapsed-seconds", "6600",
                     "--allocation-probe", str(probe)], f"fits-{cycle:02d}")
            result = json.loads((a.output / "fits" / f"RESULT-{fit_attempt}.json").read_text())
            if result["all_converged"]:
                status = "converged"
                break
            cycle += 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        # Children are our own process trees. FITS handles its own learner children.
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        ended = time.time()
        (attempt / "RELEASED.json").write_text(json.dumps(dict(
            status=status, source_commit=source, hostname=socket.gethostname(), task=a.task,
            started_unix=started, ended_unix=ended, worker_GPU_hours=4 * (ended-started) / 3600,
            scheduler_charge_pending=True, exit_codes=[child.returncode for child in children],
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
