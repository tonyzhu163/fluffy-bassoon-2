"""Paired, bounded continuation control for the measured actor/critic imbalance."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .checkpoint import restore_checkpoint, save_checkpoint
from .metaworld_env import benchmark_pools, task_digest
from .metaworld_policy import make_policy
from .metaworld_ppo_diagnose import gradient_components, replay_drift
from .metaworld_train import evaluate
from .metaworld_workers import make_worker_lanes
from .ppo import collect_trials, update_ppo
from .screen import allocation, clean_source


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            assert_exact(left, right)
    else:
        assert actual == expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--reference-diagnostic", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--candidate-value-coef", type=float, default=.005)
    p.add_argument("--max-elapsed-seconds", type=float, default=2400)
    args = p.parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    if args.updates < 1 or not 0 < args.candidate_value_coef < .5 or args.max_elapsed_seconds <= 0:
        raise ValueError("Invalid bounded value-loss control")
    parent_summary = json.loads((args.parent_dir/"summary.json").read_text())
    parent_config = json.loads((args.parent_dir/"config.json").read_text())
    parent = parent_config["contract"]
    if (not parent_summary["completed"] or parent_summary["iteration"] != 80
            or parent_summary["counters"]["environment_steps"] != 10_000_000
            or parent["benchmark"] != "ml10" or parent["core"] not in ("persistent_ctm", "cmm")
            or parent["num_envs"] != 50 or parent["attempts"] != 5 or parent["horizon"] != 500
            or parent["optimization"]["epochs"] != 10 or parent["optimization"]["value_coef"] != .5
            or parent["optimization"]["replay_chunk_length"] != 100):
        raise ValueError("Require the completed qualified ML10 parent")
    reference_summary = json.loads((args.reference_diagnostic/"summary.json").read_text())
    if not reference_summary["completed"] or reference_summary["parent_checkpoint_sha256"] != parent_summary["checkpoint_sha256"]:
        raise ValueError("Reference diagnostic has a different parent")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir/"parent-complete.pt"
    shutil.copyfile(args.parent_dir/"latest.pt", checkpoint)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != parent_summary["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum mismatch")
    reference_path = args.reference_diagnostic/"diagnostic-after-update.pt"
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != reference_summary["diagnostic_checkpoint_sha256"]:
        raise ValueError("Reference checkpoint checksum mismatch")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools("ml10", parent["benchmark_seed"])
    assert {k: task_digest(v[1]) for k,v in pools.items()} == parent["pool_hashes"]
    envs = make_worker_lanes(pools["train"], 50, benchmark="ml10", attempts=5, horizon=500,
                             workers=8, allocated_cpus=16)
    policy = make_policy(parent["core"], parent["core_config"], envs[0]).cuda().eval()
    optimizer = torch.optim.Adam(policy.parameters(), **{k:v for k,v in parent["optimizer"].items() if k != "name"})
    sampler = np.random.default_rng()
    started, stop, results = time.monotonic(), False, {}

    def request_stop(signum, frame):
        nonlocal stop
        stop = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    label, iteration = None, 0
    try:
        candidate_label = f"value-coef-{args.candidate_value_coef:g}".replace(".", "p")
        for label, coefficient in (("baseline", .5), (candidate_label, args.candidate_value_coef)):
            if stop or time.monotonic()-started >= args.max_elapsed_seconds:
                break
            output = args.output_dir/label
            output.mkdir()
            fork_started = time.monotonic()
            parent_iteration, counters = restore_checkpoint(checkpoint, policy=policy, optimizer=optimizer,
                                                           environment_rng=sampler, contract=parent)
            optimization = dict(parent["optimization"], value_coef=coefficient)
            contract = dict(deepcopy(parent), protocol="metaworld_value_loss_control_v1", source_commit=source,
                optimization=optimization, comparison_eligible=False,
                initialization=dict(kind="full_optimizer_sampler_normalizer_rng_fork",
                    checkpoint_sha256=parent_summary["checkpoint_sha256"], source=parent["source_commit"],
                    iteration=parent_iteration, prior_counters=parent_summary["counters"]),
                diagnostic_evaluation=dict(block_seed_starts=[1_000_000_000,1_000_000_010],
                    conditions=["normal","memory_reset"], lanes=10, attempts=25))
            write_json(output/"config.json", dict(contract=contract, allocation=evidence,
                target_additional_updates=args.updates, target_additional_steps=args.updates*125000,
                interpretation="Paired continuation from10M steps; separate from the original comparison."))
            iteration, first_diagnostics = 0, None
            with (output/"metrics.jsonl").open("x", buffering=1) as log:
                for iteration in range(1, args.updates+1):
                    if stop or time.monotonic()-started >= args.max_elapsed_seconds:
                        iteration -= 1
                        break
                    before = time.monotonic()
                    rollout = collect_trials(policy, envs, sampler.integers(0,1_000_000_000,50), replay_chunk_length=100)
                    torch.cuda.synchronize()
                    collection = time.monotonic()-before
                    if iteration == 1:
                        reference_rollout = torch.load(args.reference_diagnostic/"diagnostic-rollout.pt",
                                                       map_location="cuda", weights_only=False)
                        assert_exact(rollout, reference_rollout)
                        del reference_rollout
                        first_diagnostics = dict(before=gradient_components(policy, rollout, optimization))
                    update_started = time.monotonic()
                    metrics = update_ppo(policy, optimizer, rollout, **optimization)
                    torch.cuda.synchronize()
                    update_seconds = time.monotonic()-update_started
                    if iteration == 1:
                        first_diagnostics["after"] = gradient_components(policy, rollout, optimization)
                        first_diagnostics["cache_drift_after"] = replay_drift(policy, rollout, 100, optimization)
                        write_json(output/"first-update-diagnostics.json", first_diagnostics)
                    policy.core.observe_after_update(rollout["observations"])
                    if iteration == 1 and label == "baseline":
                        reference = torch.load(reference_path, map_location="cuda", weights_only=False)
                        assert_exact(policy.state_dict(), reference["policy"])
                        assert_exact(optimizer.state_dict(), reference["optimizer"])
                        assert sampler.bit_generator.state == reference["environment_rng"]
                        del reference
                        write_json(output/"baseline-reference-parity.json", dict(exact=True,
                            reference_checkpoint_sha256=reference_summary["diagnostic_checkpoint_sha256"]))
                    counters["environment_steps"] += 125000
                    counters["optimizer_steps"] += metrics["optimizer_steps"]
                    counters["training_seconds"] += time.monotonic()-before
                    counters["cumulative_elapsed_seconds"] = parent_summary["counters"]["cumulative_elapsed_seconds"]+time.monotonic()-fork_started
                    digest = save_checkpoint(output/"latest.pt", policy=policy, optimizer=optimizer,
                        environment_rng=sampler, iteration=parent_iteration+iteration, counters=counters, contract=contract)
                    row = dict(iteration=iteration, environment_steps=counters["environment_steps"],
                        additional_environment_steps=iteration*125000, optimizer_steps=counters["optimizer_steps"],
                        additional_gpu_hours=(time.monotonic()-fork_started)/3600,
                        training=metrics, rollout_success=float(np.mean([e.outcomes for e in envs])),
                        timing=dict(rollout_seconds=collection, update_seconds=update_seconds))
                    log.write(json.dumps(row,allow_nan=False)+"\n")
                    print(json.dumps(dict(branch=label,**row),allow_nan=False),flush=True)
            evaluations = {}
            before = time.monotonic()
            for block, seed_start in enumerate((1_000_000_000,1_000_000_010)):
                evaluations[str(block)] = evaluate(policy,pools,"ml10",attempts=25,horizon=500,lane_count=10,
                    conditions=("normal","memory_reset"),seed_start=seed_start,training_attempts=5,
                    env_workers=8,allocated_cpus=16)
            counters["evaluation_seconds"] += time.monotonic()-before
            counters["evaluation_steps"] += 1_000_000
            counters["cumulative_elapsed_seconds"] = parent_summary["counters"]["cumulative_elapsed_seconds"]+time.monotonic()-fork_started
            digest = save_checkpoint(output/"latest.pt",policy=policy,optimizer=optimizer,environment_rng=sampler,
                iteration=parent_iteration+iteration,counters=counters,contract=contract)
            result = dict(completed=iteration==args.updates,additional_updates=iteration,counters=counters,
                evaluations=evaluations,checkpoint_sha256=digest,fork_active_seconds=time.monotonic()-fork_started)
            write_json(output/"summary.json",result)
            results[label] = result
        write_json(args.output_dir/"summary.json",dict(completed=len(results)==2 and all(x["completed"] for x in results.values()),
            source_commit=source,parent_checkpoint_sha256=parent_summary["checkpoint_sha256"],branches=results,
            active_seconds=time.monotonic()-started,allocation=evidence))
    except BaseException as error:
        write_json(args.output_dir/"failure.json",dict(error=type(error).__name__,message=str(error),
            branch=label,last_iteration=iteration,parent_preserved=True))
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
