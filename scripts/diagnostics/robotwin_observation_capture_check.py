#!/usr/bin/env python3
"""Allocated-Linux parity/timing check for optional final-only camera capture.

This is a runtime check, not learner training. All executed actions and motor
queries are reported as qualification cost and receive no source-data credit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import faulthandler
import gc
import hashlib
import json
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np


CASES = (("place_phone_stand", 615661050), ("place_phone_stand", 638361497))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def public_hashes(env, features):
    # These are the same public RGB/state/instruction inputs supplied to VLA.
    raw = env.backend._raw_obs
    return {
        "main_images": array_digest(raw["main_images"]),
        "wrist_images": array_digest(raw["wrist_images"]),
        "states": array_digest(raw["states"]),
        "task_descriptions": digest(list(raw["task_descriptions"])),
        "features": array_digest(np.asarray(features, dtype=np.float32)),
    }


def runtime_counters(env):
    backend = env.backend
    return {name: getattr(backend, name) for name in
            ("prefix_calls", "motor_queries", "prefix_seconds", "motor_seconds")}


def episode(env, task, seed, noise_bank, label, output, receipts):
    started = time.monotonic()
    before = runtime_counters(env)
    trace = dict(task=task, requested_seed=seed, steps=[])
    receipt = dict(label=label, trace=trace, status="failed", counts_as_learning=False)
    receipts[label] = receipt  # Preserve partial traces/cost even after errors.
    try:
        features = env.reset(task, seed)
        trace.update(actual_seed=int(env.last_reset_info["actual_seed"]),
                     initial=public_hashes(env, features))
        for index, noise in enumerate(noise_bank):
            features, reward, end, trunc, info = env.step(noise)
            trace["steps"].append(dict(chunk=index + 1,
                noise_sha256=array_digest(noise),
                decoded_joint_actions_sha256=info["actions_sha256"],
                issued_primitive_steps=int(info["issued_primitive_steps"]),
                executed_primitive_steps=int(info["primitive_steps"]),
                reward=float(reward), terminated=bool(end), truncated=bool(trunc),
                success=bool(info["success"]), observation_after=public_hashes(env, features)))
            if end or trunc:
                break
        else:
            raise RuntimeError("Episode did not finish within the fixed 13-chunk/400-action check")
        receipt["status"] = "complete"
    finally:
        after = runtime_counters(env)
        receipt.update(trace_sha256=digest(trace), elapsed_seconds=time.monotonic() - started,
            primitive_steps=sum(r["executed_primitive_steps"] for r in trace["steps"]),
            runtime_counters={key: after[key] - before[key] for key in before})
        write_json(output / (label + ".json"), receipt)
    print(json.dumps(dict(stage="episode", label=label,
                          primitive_steps=receipt["primitive_steps"],
                          chunks=len(trace["steps"]),
                          elapsed_seconds=receipt["elapsed_seconds"])), flush=True)
    return receipt


def run_mode(args, final_only, noise_bank, report):
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv

    name = "final_only" if final_only else "full_reference"
    mode = dict(final_observation_only=final_only, episodes={})
    report["modes"][name] = mode
    env = None
    started = time.monotonic()
    try:
        env = RobotwinNoiseEnv(
            model_path=args.model_path, norm_stats=args.norm_stats,
            robotwin_root=args.robotwin_root, rlinf_root=args.rlinf_root,
            episode_limit=400, action_horizon=32, num_steps=5,
            task_config_name="demo_clean", output_dir=args.output / name,
            final_observation_only=final_only)
        mode["model_setup_seconds"] = time.monotonic() - started
        mode["runtime_contract"] = deepcopy(env.contract)
        mode["frozen_before"] = env.frozen_fingerprint()
        env.reset(*CASES[0])
        mode["native_noise_parity"] = env.check_noise_parity(noise_bank[0])
        # Parity invokes one adapter decode and one native sampler invocation.
        mode["parity_sampler_invocations"] = 2
        for case, (task, seed) in enumerate(CASES, start=1):
            repetitions = ("final_only",) if final_only else ("full_reference_A", "full_reference_B")
            for repetition in repetitions:
                label = f"case{case}-{repetition}"
                episode(env, task, seed, noise_bank, label, args.output, mode["episodes"])
        mode["frozen_after"] = env.frozen_fingerprint()
        mode["frozen_unchanged"] = mode["frozen_before"] == mode["frozen_after"]
        mode["runtime_totals_including_parity"] = runtime_counters(env)
        return mode
    finally:
        mode["total_seconds"] = time.monotonic() - started
        try:
            if env is not None:
                env.close()  # Same calling thread as every scene operation.
        finally:
            write_json(args.output / (name + "-runtime.json"), mode)


def candidate_comparison(reference, candidate):
    a, b = reference["trace"], candidate["trace"]
    mismatches = []
    terminal_observations = []
    for field in ("task", "requested_seed", "actual_seed", "initial"):
        if a[field] != b[field]:
            mismatches.append(dict(field=field, reference=a[field], candidate=b[field]))
    if len(a["steps"]) != len(b["steps"]):
        mismatches.append(dict(field="chunk_count", reference=len(a["steps"]), candidate=len(b["steps"])))
    outcome_fields = ("noise_sha256", "decoded_joint_actions_sha256", "issued_primitive_steps",
                      "executed_primitive_steps", "reward", "terminated", "truncated", "success")
    for index, (left, right) in enumerate(zip(a["steps"], b["steps"]), start=1):
        for field in outcome_fields:
            if left[field] != right[field]:
                mismatches.append(dict(chunk=index, field=field, reference=left[field], candidate=right[field]))
        observed_match = left["observation_after"] == right["observation_after"]
        if left["terminated"] or left["truncated"]:
            terminal_observations.append(dict(chunk=index, passed=observed_match,
                terminated=left["terminated"], truncated=left["truncated"],
                reference=left["observation_after"], candidate=right["observation_after"]))
        elif not observed_match:
            mismatches.append(dict(chunk=index, field="nonterminal_decision_observation",
                                   reference=left["observation_after"], candidate=right["observation_after"]))
    terminal_match = bool(terminal_observations) and all(r["passed"] for r in terminal_observations)
    return dict(decision_trace_passed=not mismatches, mismatches=mismatches,
                terminal_observations=terminal_observations,
                terminal_observations_passed=terminal_match,
                passed=not mismatches and terminal_match)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--norm-stats", type=Path, required=True)
    p.add_argument("--robotwin-root", type=Path, required=True)
    p.add_argument("--rlinf-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--noise-seed", type=int, default=19029)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh qualification output identity")
    args.output.mkdir(parents=True)
    faulthandler.enable()
    def interrupted(signum, frame):
        raise TimeoutError(f"Qualification interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    started = time.monotonic()
    report = dict(status="failed", passed=False, timing_comparison_eligible=False,
                  counts_as_learning=False, source_commit=args.source_commit,
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  cases=[dict(task=task, seed=seed) for task, seed in CASES],
                  modes={}, comparisons={},
                  terminal_image_policy="Report separately; mismatches also block automatic qualification")
    write_json(args.output / "qualification.json", report)
    try:
        if sys.platform != "linux":
            raise RuntimeError("Execute only inside the allocated Linux GPU job")
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            raise ValueError("Require a full source commit")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], text=True).strip()
        if actual != args.source_commit:
            raise RuntimeError(f"Source commit mismatch: {actual}")
        noise_bank = np.random.default_rng(args.noise_seed).normal(size=(13, 32, 32)).astype(np.float32)
        np.save(args.output / "shared-noise-bank.npy", noise_bank, allow_pickle=False)
        report["noise"] = dict(distribution="fixed standard Gaussian", seed=args.noise_seed,
                               shape=list(noise_bank.shape), sha256=array_digest(noise_bank),
                               shared_by="all six episodes and both motor parity checks")
        full = run_mode(args, False, noise_bank, report)
        for case in range(1, len(CASES) + 1):
            a = full["episodes"][f"case{case}-full_reference_A"]
            b = full["episodes"][f"case{case}-full_reference_B"]
            report["comparisons"][f"case{case}-reference_repeat"] = dict(
                passed=a["trace"] == b["trace"], reference_a_sha256=a["trace_sha256"],
                reference_b_sha256=b["trace_sha256"])
        if not all(row["passed"] for row in report["comparisons"].values()):
            raise RuntimeError("Full-reference repeat is not deterministic; candidate admission stopped")
        gc.collect()
        final = run_mode(args, True, noise_bank, report)
        report["real_early_success_observed"] = any(
            step["success"] and step["executed_primitive_steps"] < step["issued_primitive_steps"]
            for mode in report["modes"].values() for ep in mode["episodes"].values()
            for step in ep["trace"]["steps"])
        for case in range(1, len(CASES) + 1):
            reference = full["episodes"][f"case{case}-full_reference_B"]
            candidate = final["episodes"][f"case{case}-final_only"]
            report["comparisons"][f"case{case}-final_only"] = candidate_comparison(reference, candidate)
        frozen_match = all(mode["frozen_unchanged"] for mode in report["modes"].values())
        frozen_match &= full["frozen_before"] == final["frozen_before"]
        report["same_unchanged_frozen_motor"] = bool(frozen_match)
        report["passed"] = bool(frozen_match) and all(r["passed"] for r in report["comparisons"].values())
        if not report["passed"]:
            raise RuntimeError("Candidate parity failed; no timing comparison is admitted")
        report["timing_comparison_eligible"] = True
        reference_seconds = sum(full["episodes"][f"case{i}-full_reference_B"]["elapsed_seconds"]
                                for i in range(1, len(CASES) + 1))
        final_seconds = sum(row["elapsed_seconds"] for row in final["episodes"].values())
        report["paired_episode_timing"] = dict(reference_seconds=reference_seconds,
            final_only_seconds=final_seconds, measured_speedup=reference_seconds / final_seconds)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error))
    finally:
        report["total_seconds"] = time.monotonic() - started
        report["total_qualification_primitive_steps"] = sum(
            row["primitive_steps"] for mode in report["modes"].values()
            for row in mode["episodes"].values())
        write_json(args.output / "qualification.json", report)
    print(json.dumps(dict(status=report["status"], passed=report["passed"],
                          qualification=str(args.output / "qualification.json"),
                          total_seconds=report["total_seconds"])), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
