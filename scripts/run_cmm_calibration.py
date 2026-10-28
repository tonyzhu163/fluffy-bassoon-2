"""Qualify one allocated GPU, then run the first full CMM calibration seed.

Invoke with the chosen environment's Python from a clean, commit-pinned
worktree. On Slurm this command belongs inside an ``srun --ntasks=1`` step;
on Baige it belongs in a fresh exclusive one-GPU scheduler container. The
output root and the training directory must be new, including on resume.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", choices=("slurm", "baige"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-elapsed-seconds", type=float, default=23 * 3600)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run qualification and calibration on allocated Linux")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        parser.error("Require a complete source commit")
    if not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0:
        parser.error("Require a positive finite time cap")
    if args.torch_threads < 1:
        parser.error("Require a positive Torch thread count")
    root = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if actual != args.source_commit or dirty:
        parser.error("Require the exact clean pinned source")
    if args.execution == "slurm" and (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_STEP_ID") in (None, "batch", "extern")
        or os.environ.get("SLURM_STEP_NUM_TASKS", "1") != "1"
    ):
        parser.error("Use a one-task allocated srun step")
    if args.resume and not args.resume.resolve().is_file():
        parser.error("Resume checkpoint does not exist")
    environment = dict(os.environ)
    if args.execution == "baige" and not environment.get("CUDA_VISIBLE_DEVICES"):
        # The probe below verifies that this is the scheduler's sole GPU.
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment.update(FLUFFY_GPU_EXECUTION=args.execution, FLUFFY_META_RL_CUDA_QUALIFICATION="1",
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    probe = [sys.executable, str(root / "scripts" / "[train]gpu_probe.py"), "--expected-count", "1",
             "--purpose", "training", "--output", str(output / "gpu-probe.json")]
    if args.execution == "baige":
        probe.append("--baige-container")
    tests = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "tests/test_continuous_memory_machine.py", "tests/test_meta_rl_cmm.py",
             "tests/test_meta_rl_fewshot.py", "tests/test_meta_rl_copy.py",
             "tests/test_meta_rl_ppo.py", "tests/test_meta_rl_train.py", "tests/test_meta_rl_cuda.py"]
    training = [sys.executable, "-m", "fluffy_bassoon.meta_rl.fewshot_regression",
                "--source-commit", args.source_commit, "--output-dir", str(output / "training"),
                "--variant", "cmm", "--shots", "20", "--seed", "0", "--epochs", "1000",
                "--batch-size", "64", "--eval-every-epochs", "25", "--checkpoint-every-epochs", "5",
                "--dev-episodes", "1000", "--device", "cuda",
                "--torch-threads", str(args.torch_threads),
                "--max-elapsed-seconds", str(args.max_elapsed_seconds)]
    if args.resume:
        training.extend(("--resume", str(args.resume.resolve())))
    receipt = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), source_commit=actual,
                   hostname=socket.gethostname(), execution=args.execution, python=sys.executable,
                   probe=probe, tests=tests, training=training,
                   allocation={key: environment.get(key) for key in (
                       "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_CPUS_PER_TASK",
                       "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")})
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for name, command in (("gpu-probe", probe), ("qualification", tests)):
        with (output / f"{name}.log").open("x", buffering=1) as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(phase=name, returncode=result.returncode)), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    # Replace the launcher so scheduler USR1/TERM reaches the checkpoint handler.
    os.chdir(root)
    os.execvpe(training[0], training, environment)


if __name__ == "__main__":
    main()
