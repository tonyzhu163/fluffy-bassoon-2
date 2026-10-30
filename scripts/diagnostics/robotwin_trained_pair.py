#!/usr/bin/env python3
"""One bounded, checkpoint-initialized N6 round for each original PPO stream.

This is an execution artifact. It never allocates resources, stops an old
writer, creates a physical ledger, retries a failed group, or initializes a
replacement learner. The scientific driver remains pinned to RUNTIME.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

RUNTIME = "57016ba08eed79043fd67bf8b085bd6d42e6d766"
LINEAGE = "db0184089fa88a5f96c153f9b80029bb91b77def"
PARENT_SHA = "1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0"
ROOT = Path("/mowenjie/fluffy-bassoon-2")
CAMPAIGN = "robotwin-native-wholetrial-20261030-r2"
METHODS = ("ppo", "ppo_reset")
CAPS = {"ppo": 1226400, "ppo_reset": 1168800}
MODEL = ROOT / "cache/robotwin/pi05-motus-effc6e1a-r1"
NATIVE = ROOT / "sources/robotwin-baseline-smoke-20261028-r1"
PYTHON = ROOT / "envs/robotwin-vlrla-20261029-r1/bin/python"
RESOURCES = dict(gpus=1, cpus=32, memory_gib=192, shared_memory_gib=16,
                 scheduler_cap_seconds=2700)
HEAD_KEYS = ("journal_events", "journal_sha256", "charged_actions", "decision_count")


def require(condition, message):
    if not condition:
        raise ValueError("Trained paired launcher: " + message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atom(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def check_ref(ref):
    require(isinstance(ref, dict) and set(ref) == {"path", "sha256"}, "exact artifact reference required")
    require(Path(ref["path"]).is_absolute() and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "absolute hash-bound artifact required")
    return Path(ref["path"])


def read_ref(ref):
    path = check_ref(ref)
    require(sha(path) == ref["sha256"], f"artifact changed: {path}")
    return json.loads(path.read_text())


def ledger_root(method):
    return ROOT / "campaigns" / CAMPAIGN / "worker-ledgers" / method


def build_commands(plan):
    """Pure command construction; compilation performs no Torch/CUDA work."""
    commands = {}
    for method in METHODS:
        arm = plan["methods"][method]
        command = [str(PYTHON), str(Path(plan["worktree"]) / "scripts/diagnostics/robotwin_noise_compare.py"),
            "--source-commit", RUNTIME, "--lineage-source-commit", LINEAGE,
            "--lineage-parent-manifest", plan["parent_manifest"]["path"],
            "--lineage-parent-manifest-sha256", PARENT_SHA,
            "--model-path", str(MODEL), "--norm-stats",
            str(MODEL / "assets/pi0.5_clean_randomize_joint_training/norm_stats.json"),
            "--robotwin-root", str(NATIVE / "RoboTwin"), "--rlinf-root", str(NATIVE / "RLinf"),
            "--output", arm["output"], "--seed", "20261030", "--source-steps", "1100000",
            "--source-iterations", "100", "--source-eval-iterations", "10", "25", "50", "100",
            "--benchmark-episode-limits", "--ppo-trials-per-update", "6", "--ppo-epochs", "5",
            "--ppo-minibatch-trials", "16", "--learning-rate", "1e-6", "--ppo-max-update-kl", ".1",
            "--train-episodes", "4", "--eval-episodes", "8", "--eval-trials", "1",
            "--methods", method, "--native-horizon", "32", "--episode-limit", "600",
            "--eval-tasks", "adjust_bottle", "move_stapler_pad", "--source-validation-episodes", "4",
            "--native-episodes", "1" if method == "ppo" else "0", "--task-config-name", "demo_clean",
            "--action-limit", "1.5", "--source-tasks", "handover_mic", "pick_diverse_bottles",
            "place_a2b_left", "place_object_stand", "place_phone_stand", "scan_object",
            "--sac-warmup", "500", "--sac-batch-size", "256", "--sac-utd", "20",
            "--campaign-root", str(ledger_root(method)), "--campaign-id", CAMPAIGN + "-" + method,
            "--campaign-max-actions", str(CAPS[method]), "--record-corpus", "--final-observation-only",
            "--capture-qualification", str(ROOT / "outputs/robotwin-capture-qual-20261029-r1/qualification/qualification.json"),
            "--wall-seconds", str(plan["child_wall_seconds"]), "--boundary-reserve-seconds", "180",
            "--resume", arm["checkpoint"]["path"], "--resume-trial-manifest", arm["manifest"]["path"],
            "--resume-trial-manifest-sha256", arm["manifest"]["sha256"],
            "--collector-lanes", "6", "--collector-group-seconds", "600",
            "--max-source-trials-per-process", "6", "--recycle-at-round-boundary"]
        commands[method] = command
    return commands


def validate_plan(plan, *, commands_required=True, read_evidence=True):
    require(plan["schema"] == "robotwin_trained_pair_plan_v1", "unknown plan schema")
    require(plan["runtime_commit"] == RUNTIME and plan["lineage_commit"] == LINEAGE,
            "runtime/scientific lineage pin differs")
    require(re.fullmatch(r"robotwin-trained-pair-[A-Za-z0-9-]+", plan["run_id"]), "fresh named run required")
    output = Path(plan["output"])
    require(output == ROOT / "outputs" / plan["run_id"], "output identity must be canonical and fresh")
    require(Path(plan["worktree"]).is_absolute(), "absolute runtime worktree required")
    require(plan["parent_manifest"]["sha256"] == PARENT_SHA, "original parent SHA differs")
    check_ref(plan["parent_manifest"])
    require(set(plan["methods"]) == set(METHODS), "two independent trained PPO streams required")
    require(plan["resources"] == RESOURCES, "reviewed resource envelope differs")
    require(plan["maximum_seconds"] == 2520 and plan["child_wall_seconds"] == 2160,
            "reviewed parent/child operational bounds differ")
    require(plan["maximum_new_source_actions"] == 21600
            and plan["maximum_input_due_evaluation_actions"] == 48800,
            "original phase bounds differ")
    for method in METHODS:
        arm = plan["methods"][method]
        require(Path(arm["output"]) == output / method, "method output identity differs")
        for name in ("manifest", "checkpoint", "frozen_motor"):
            check_ref(arm[name])
    if commands_required:
        require(plan["child_commands"] == build_commands(plan), "commands changed from reviewed production recipe")
    for name in ("owner_release", "compute_binding"):
        check_ref(plan[name])
    if not read_evidence:
        return
    parent = read_ref(plan["parent_manifest"])
    require(parent["source_commit"] == LINEAGE and parent["campaign_id"] == CAMPAIGN,
            "another source campaign cannot supply this parent")
    for method in METHODS:
        worker = parent["workers"][method]
        require(worker["ledger_root"] == str(ledger_root(method)) and worker["maximum_actions"] == CAPS[method]
                and worker["source_safety_actions"] == 1100000, "original canonical worker changed")
    release = read_ref(plan["owner_release"])
    require(release["schema"] == "robotwin_trained_pair_owner_release_v1"
            and release["runtime_commit"] == RUNTIME and release["parent_manifest_sha256"] == PARENT_SHA,
            "owner release is not bound to this migration")
    require(set(release["methods"]) == set(METHODS), "both predecessor writers need terminal releases")
    for method in METHODS:
        row = release["methods"][method]
        require(row["released"] is True and re.fullmatch(r"job-[a-z0-9]+", row["allocation_id"])
                and row["terminal_state"] in ("Stopped", "Succeeded", "Failed", "Abnormal", "Completed"),
                "old source allocation is not attested terminal")
        require(row["manifest"] == plan["methods"][method]["manifest"]
                and row["checkpoint"] == plan["methods"][method]["checkpoint"], "released checkpoint differs")
        require(set(row["canonical_head"]) == set(HEAD_KEYS), "released canonical journal head is incomplete")
        # A screenshot/text receipt remains provenance; the source owner verifies
        # terminal scheduler state and exact PID ownership before issuing it.
        check_ref(row["terminal_receipt"])
        require(sha(row["terminal_receipt"]["path"]) == row["terminal_receipt"]["sha256"],
                "terminal scheduler evidence changed")
    compute = read_ref(plan["compute_binding"])
    require(compute["schema"] == "robotwin_trained_pair_compute_v1" and compute["campaign_id"] == CAMPAIGN,
            "original source compute account required")
    require(compute["original_total_compute_ceiling_seconds"] is None
            and compute["source_goal_rounds"] == 100
            and compute["qualification_capacity_reused"] is False
            and compute["maximum_scheduler_gpu_seconds"] == 2700
            and compute["admitted_scheduler_gpu_seconds"] == 2700,
            "goal-completion authority or this slice's operational ceiling differs")
    require(compute["resources"] == RESOURCES and compute["owner_release"] == plan["owner_release"],
            "resource residual is not bound to this exclusive handoff")
    for name in ("authority", "prior_charge_and_physical_snapshot"):
        path = check_ref(compute[name])
        require(sha(path) == compute[name]["sha256"], "original authority/accounting evidence changed")


def pinned_worktree(path, commit):
    path = Path(path)
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "full commit required")
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=10).strip()
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True, timeout=10).strip()
    require(actual == commit and not dirty, "clean immutable pinned worktree required")


def runtime_environment(worktree):
    env = dict(os.environ)
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        FLUFFY_GPU_EXECUTION="baige", PYTHONPATH=f"{worktree}/src:{NATIVE}/RLinf:{NATIVE}/RoboTwin",
        HF_HOME="/mowenjie/.cache/huggingface", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_XET="1", JAX_PLATFORMS="cpu", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MPLBACKEND="Agg",
        OPENPI_DATA_HOME="/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1",
        ROBOT_PLATFORM="ALOHA", ASSETS_PATH=str(NATIVE / "RoboTwin"), ROBOTWIN_PATH=str(NATIVE / "RoboTwin"),
        EMBODIED_PATH=str(NATIVE / "RLinf/examples/embodiment"), REPO_PATH=str(NATIVE / "RLinf"))
    # Preserve the scheduler's device namespace; the real probe attests it.
    return env


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--launcher-commit", required=True)
    parser.add_argument("--compile-plan", type=Path)
    args = parser.parse_args(argv)
    require(sha(args.plan) == args.plan_sha256, "input plan SHA differs")
    plan = json.loads(args.plan.read_text())
    if args.compile_plan is not None:
        require(not args.compile_plan.exists(), "compiled plan must be fresh")
        validate_plan(plan, commands_required=False, read_evidence=False)
        plan["child_commands"] = build_commands(plan)
        atom(args.compile_plan, plan)
        print(json.dumps(dict(plan=str(args.compile_plan.resolve()), sha256=sha(args.compile_plan))))
        return 0
    require(sys.platform == "linux" and socket.gethostname().startswith("job-"),
            "run only in the exclusive allocated Baige container")
    launcher = Path(__file__).resolve().parents[2]
    pinned_worktree(launcher, args.launcher_commit)
    pinned_worktree(plan["worktree"], RUNTIME)
    # All potentially slow/shared-storage evidence reads occur in the bounded
    # CPU subprocess, before even the one idle CUDA probe.
    validate_plan(plan, read_evidence=False)
    output = Path(plan["output"])
    require(not output.exists(), "output already exists; no automatic retry")
    output.mkdir(parents=True)
    atom(output / "plan.json", plan)
    from robotwin_trained_pair_runtime import TrainedPairRuntime
    env = runtime_environment(plan["worktree"])
    status = "failed_requires_owner_audit"
    result = None
    failure = None
    try:
        with TrainedPairRuntime(Path(plan["worktree"]), output, maximum_seconds=2520,
                                cleanup_reserve_seconds=180) as runtime:
            preflight = [str(PYTHON), str(launcher / "scripts/diagnostics/robotwin_trained_pair_preflight.py"),
                "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
                "--receipt", str(output / "preflight.json"), "--mode", "before"]
            runtime.run_bounded(preflight, cwd=launcher, env=env, log_path=output / "preflight.log",
                                timeout_seconds=90, purpose="cpu_preflight")
            require((output / "preflight.json").is_file(), "missing successful CPU preflight receipt")
            admitted = json.loads((output / "preflight.json").read_text())
            require(admitted.get("status") == "passed" and admitted.get("cuda_initialized") is False
                    and admitted.get("plan", {}).get("sha256") == args.plan_sha256,
                    "CPU preflight receipt did not admit this exact plan without CUDA")
            probe = output / "gpu-probe.json"
            runtime.run_bounded([str(PYTHON), str(Path(plan["worktree"]) / "scripts/[train]gpu_probe.py"),
                "--expected-count", "1", "--purpose", "training", "--baige-container", "--output", str(probe)],
                cwd=plan["worktree"], env=env, log_path=output / "gpu-probe.log",
                timeout_seconds=90, purpose="gpu_probe")
            gpu = json.loads(probe.read_text())["devices"][0]
            require(90 * (1 << 30) <= gpu["cuda_total_bytes"] <= 100 * (1 << 30),
                    "this paired launch requires the reviewed96GB GPU class")
            env["LYCO_BAIGE_GPU_MAP"] = str(probe)
            result = runtime.run_pair(plan["child_commands"], probe_path=probe, env=env)
            atom(output / "pair-result.json", result)
            # Validation runs in a separate CPU process after all owned simulator
            # sessions have gone. A crash never becomes an automatic next attempt.
            after = [str(PYTHON), str(launcher / "scripts/diagnostics/robotwin_trained_pair_audit.py"),
                "--plan", str(args.plan.resolve()), "--plan-sha256", args.plan_sha256,
                "--receipt", str(output / "audit.json")]
            runtime.run_bounded(after, cwd=launcher, env=env, log_path=output / "audit.log",
                                timeout_seconds=90, purpose="cpu_preflight")
            audit = json.loads((output / "audit.json").read_text())
            require(audit.get("status") == "passed_one_trained_round_each", "one-round audit did not pass")
            status = "passed_one_trained_round_each"
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        atom(output / "launcher-result.json", dict(status=status, runtime_commit=RUNTIME,
            launcher_commit=args.launcher_commit, plan_sha256=args.plan_sha256,
            no_automatic_retry=True, scheduler_gpu_seconds_requires_terminal_owner_reconciliation=True,
            parent_elapsed_seconds=time.monotonic() - started, failure=failure, result=result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
