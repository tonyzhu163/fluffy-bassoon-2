#!/usr/bin/env python3
"""One explicitly diagnostic PPO update on immutable, fresh r4 trajectories.

No simulator or VLA is instantiated. The three-trial batch is a separate fork;
it never advances the batch-eight matched comparison or its source checkpoint.
Run only in an actual allocated Linux GPU job.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import runpy
import signal
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-batch-trials", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, default=45)
    args = parser.parse_args()
    if sys.platform != "linux" or args.diagnostic_batch_trials != 3 or not 20 <= args.wall_seconds <= 45:
        raise ValueError("Allocated Linux only; explicitly require the three-trial diagnostic and <=45 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="initializing", source_commit=args.source_commit,
        diagnostic="one_saved_fresh_PPO_update_batch3", main_comparison_batch_trials=8,
        main_comparison_advanced=False, executed_new_robot_actions=0,
        input_checkpoint=str(args.input), input_sha256=args.input_sha256,
        complete_preupdate_recovery=str(args.input), retained_ppo_updates=0,
        scientific_scope="optimizer execution on three existing complete K4 source trials; no robot outcome or SAC comparison")
    write_json(args.output/"result.json", report)

    def expired(signum, frame):
        raise TimeoutError("Diagnostic wall budget reached; original full recovery is unchanged")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.wall_seconds)
    update_started = False
    try:
        actual_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
        if actual_sha != args.input_sha256:
            raise ValueError("Immutable source checkpoint SHA mismatch")
        # Invoke the normal allocation probe in this process, avoiding a second
        # Torch import and CUDA context inside the remaining one-minute slice.
        saved_argv = sys.argv
        sys.argv = ["gpu_probe.py", "--expected-count", "1", "--output",
                    str(args.output/"gpu-probe.json"), "--purpose", "training", "--baige-container"]
        try:
            runpy.run_path(str(Path(__file__).parents[1]/"[train]gpu_probe.py"), run_name="__main__")
        finally:
            sys.argv = saved_argv
        import torch
        from fluffy_bassoon.meta_rl.vla_noise_ppo import PPOConfig, RecurrentNoisePPO
        payload = torch.load(args.input, map_location="cpu", weights_only=False)
        pending = payload["pending_rollouts"]
        if (payload["method"] != "ppo" or len(pending) != args.diagnostic_batch_trials
                or payload["learner"]["updates"] != 0 or payload["learner"]["optimizer_steps"] != 0):
            raise ValueError("Require the unchanged initial persistent PPO policy and exactly three fresh trials")
        for rollout, result in pending:
            ends = rollout["trial_ends"].reshape(-1).tolist()
            if (result["requested_episodes"] != 4 or len(result["episodes"]) != 4
                    or result["budget_truncated"] or not ends[-1] or any(ends[:-1])
                    or result["task"] not in payload["contract"]["source_tasks"]):
                raise ValueError("Every diagnostic trajectory must be a complete eligible K4 source trial")
        config = PPOConfig(**payload["learner"]["config"])
        if config.epochs != 5 or config.reset_each_episode or config.minibatch_trials < len(pending):
            raise ValueError("Preserve the original five-epoch persistent PPO objective")
        learner = RecurrentNoisePPO(payload["learner"]["observation_dim"],
            payload["learner"]["noise_shape"], config=config, device="cuda")
        learner.load_state_dict(payload["learner"])
        errors = [learner.replay_error(rollout) for rollout, _ in pending]
        if any(not all(math.isfinite(value) for value in row.values())
               or row["log_prob_max_error"] > 2e-4 or row["value_max_error"] > 1e-5 for row in errors):
            raise RuntimeError("Saved behavior likelihood/value replay does not match the unchanged policy")
        def weight_hash(weights):
            digest = hashlib.sha256()
            for key, tensor in sorted(weights.items()):
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        before = weight_hash(learner.policy.state_dict())
        report.update(status="updating", behavior_replay=errors,
            source_actions=sum(result["primitive_steps"] for _, result in pending),
            source_episodes=sum(len(result["episodes"]) for _, result in pending),
            chunks=sum(result["chunks"] for _, result in pending),
            tasks=[result["task"] for _, result in pending], policy_before_sha256=before,
            preparation_seconds=time.monotonic()-started)
        write_json(args.output/"result.json", report)
        if time.monotonic()-started > args.wall_seconds-15:
            raise TimeoutError("Less than15seconds remains; do not start an update without checkpoint reserve")
        torch.cuda.synchronize()
        update_started = True
        update_start = time.monotonic()
        metrics = learner.update([rollout for rollout, _ in pending])
        torch.cuda.synchronize()
        update_seconds = time.monotonic()-update_start
        after = weight_hash(learner.policy.state_dict())
        if (not all(math.isfinite(float(value)) for value in metrics.values())
                or metrics["gradient_norm"] <= 0 or before == after
                or learner.updates != 1 or learner.optimizer_steps != 5):
            raise RuntimeError("Require finite nonzero learning, changed weights, one PPO update and five optimizer steps")
        recovery = deepcopy(payload)
        recovery.update(contract=dict(diagnostic=report["diagnostic"],source_commit=args.source_commit,
            diagnostic_batch_trials=3, original_source_contract=payload["contract"],
            input_checkpoint_sha256=args.input_sha256, main_comparison_advanced=False),
            learner=learner.state_dict(), pending_rollouts=[], consumed_rollouts=pending,
            boundary="complete diagnostic PPO update; all original source trajectories retained")
        checkpoint = args.output/"diagnostic-recovery.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(recovery, temporary)
        temporary.replace(checkpoint)
        report.update(status="complete", retained_ppo_updates=1, optimizer_steps=learner.optimizer_steps,
            update_seconds=update_seconds, metrics=metrics, policy_after_sha256=after,
            recovery=str(checkpoint), recovery_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            optimizer_state_count=len(learner.optimizer.state_dict()["state"]),
            consumed_trials=len(pending), pending_trials=0)
    except Exception as exc:
        report.update(status="wall_budget_stop" if isinstance(exc, TimeoutError) else "failed",
            error_type=type(exc).__name__, reason=str(exc),
            partial_optimization_discarded=update_started,
            complete_preupdate_recovery_unchanged=True)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic()-started
        write_json(args.output/"result.json", report)
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
