"""Plan and execute an immutable parameter-matched compact learner screen.

Planning constructs models and therefore requires an allocated Linux step or
Baige scheduler container. Each worker owns a disjoint shard; every execution
uses a fresh attempt identity. Interrupted training resumes into a new attempt
with --resume, preserving optimizer, environment sampler and all RNG state.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.gpu_execution import validate_execution


VARIANTS = (
    ("ctm-t1", "persistent_ctm", 1),
    ("ctm-t4", "persistent_ctm", 4),
    ("ctm-t8", "persistent_ctm", 8),
    ("cmm", "cmm", 1),
    ("transformer-xl", "transformer_xl", 1),
    ("gru", "gru", 1),
    ("looped-transformer", "looped_transformer", 4),
)


def allocation():
    if sys.platform != "linux":
        raise ValueError("Screen model construction and execution require allocated Linux resources")
    execution = os.environ.get("FLUFFY_GPU_EXECUTION", "slurm")
    if execution not in ("slurm", "baige"):
        raise ValueError("Screen requires allocated Slurm or Baige execution")
    validate_execution(execution, os.environ)
    return dict(execution=execution, job_id=os.environ.get("SLURM_JOB_ID", os.uname().nodename),
                hostname=os.uname().nodename, utc=datetime.now(timezone.utc).isoformat())


def clean_source(expected):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (source != expected
            or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()):
        raise ValueError("Screen requires its exact clean source commit")
    return source


def matching_width(core, env, settings, budget, minimum_fraction=.85):
    """Largest width divisible by four under a declared total policy parameter cap."""
    from .cores import build_policy, core_config_from_args, policy_size
    selected = None
    low, high = 1, 256
    while low <= high:
        middle = (low + high) // 2
        config = core_config_from_args(SimpleNamespace(core=core, width=4 * middle, **settings))
        size = policy_size(build_policy(core, config, env))
        if size["parameters"] <= budget:
            selected = dict(width=4 * middle, core_config=config, **size)
            low = middle + 1
        else:
            high = middle - 1
    if selected is None or selected["parameters"] < minimum_fraction * budget:
        raise ValueError(f"Cannot match {core} within [{minimum_fraction * budget}, {budget}] parameters")
    if selected["width"] == 1024:
        raise ValueError("Parameter budget exceeds the screen's width search; increase the explicit limit")
    return selected


def training_options(args):
    return dict(updates=args.updates, num_envs=args.num_envs, learning_rate=args.learning_rate,
                reward_offset=args.reward_offset, value_coef=args.value_coef,
                ppo_epochs=args.ppo_epochs, minibatch_trials=args.minibatch_trials,
                eval_every=args.eval_every, eval_trials=args.eval_trials,
                checkpoint_every=args.checkpoint_every, max_elapsed_seconds=args.max_elapsed_seconds,
                device=args.device)


def plan(args):
    evidence = allocation()
    clean_source(args.source_commit)
    from .train import make_env
    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    for values, label in ((args.attempts, "attempt lengths"), (args.seeds, "seeds")):
        if len(values) != len(set(values)):
            raise ValueError(f"Require distinct {label}")
    positive = (args.parameter_budget, args.updates, args.num_envs, args.ppo_epochs,
                args.minibatch_trials, args.eval_every, args.eval_trials, args.checkpoint_every,
                args.arms, *args.attempts)
    if (any(value < 1 for value in positive) or args.arms < 2
            or any(not 0 <= seed < 1_000_000_000 for seed in args.seeds)
            or not 0 < args.minimum_parameter_fraction <= 1
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.reward_offset) or not 0 <= args.reward_offset <= 1
            or not math.isfinite(args.value_coef) or args.value_coef < 0):
        raise ValueError("Invalid screen budget or optimization settings")
    members = []
    variants = []
    env = make_env("bernoulli", args.attempts[0], 1, args.arms)
    try:
        for label, core, ticks in VARIANTS:
            settings = dict(ticks=ticks, stm_length=32, ltm_tokens=32, synapse_depth=8,
                            bounded_synchronisation=core == "persistent_ctm",
                            decay_per_observation=core == "persistent_ctm",
                            transformer_memory_length=max(args.attempts), transformer_layers=2,
                            looped_memory_tokens=8)
            match = matching_width(core, env, settings, args.parameter_budget,
                                   args.minimum_parameter_fraction)
            variant = dict(label=label, core=core, settings=settings, **match)
            variants.append(variant)
    finally:
        env.close()
    for attempts in args.attempts:
        for variant in variants:
            for seed in args.seeds:
                members.append(dict(index=len(members), task="bernoulli", attempts=attempts,
                                    arms=args.arms, horizon=1, seed=seed, variant=variant["label"]))
    manifest = dict(version=1, source_commit=args.source_commit, allocation=evidence,
                    parameter_budget=args.parameter_budget,
                    matching="largest width divisible by four within total policy parameter cap",
                    minimum_parameter_fraction=args.minimum_parameter_fraction,
                    matching_axes=["trainable parameters", "environment data", "PPO optimizer schedule"],
                    reported_unmatched_axes=["retained state bytes", "FLOPs", "wall time"],
                    training=training_options(args), variants=variants, members=members,
                    diagnostics=dict(trials=args.eval_trials, device=args.device,
                                     environment_seed_start=1_000_000_000,
                                     policy_seeds=[2_000_000_000, 2_000_000_001, 2_000_000_002],
                                     conditions=["normal", "reset_between_attempts", "reward_blind"]))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", manifest)
    digest = sha256(args.output_dir / "manifest.json")
    (args.output_dir / "manifest.sha256").write_text(digest + "\n")
    print(json.dumps(dict(manifest=str(args.output_dir / "manifest.json"), sha256=digest,
                          members=len(members), variants=variants), allow_nan=False), flush=True)
    return manifest


def load_manifest(path):
    path = Path(path).resolve()
    digest = sha256(path)
    if path.with_suffix(".sha256").read_text().strip() != digest:
        raise ValueError("Screen manifest checksum changed")
    manifest = json.loads(path.read_text())
    if (manifest.get("version") != 1 or not manifest.get("members")
            or [m["index"] for m in manifest["members"]] != list(range(len(manifest["members"])))):
        raise ValueError("Invalid screen manifest")
    clean_source(manifest["source_commit"])
    return path, digest, manifest


def command_options(options):
    result = []
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                result.append("--" + key.replace("_", "-"))
        else:
            result.extend(("--" + key.replace("_", "-"), str(value)))
    return result


def cost_receipt(training_dir, manifest, member):
    records = [json.loads(line) for line in (training_dir / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in records if "training" in r]
    durations = [r["timing"]["rollout_seconds"] + r["timing"]["update_seconds"] for r in updates]
    evaluations = [r["timing"]["evaluation_seconds"] for r in updates
                   if "evaluation_seconds" in r.get("timing", {})]
    training = manifest["training"]
    estimate = None
    if durations:
        estimate = statistics.median(durations) * training["updates"]
        if evaluations:
            estimate += statistics.median(evaluations) * (math.ceil(training["updates"] / training["eval_every"]) + 1)
    return dict(measured_updates=len(updates),
                median_rollout_and_update_seconds=None if not durations else statistics.median(durations),
                median_evaluation_seconds=None if not evaluations else statistics.median(evaluations),
                estimated_training_seconds_at_declared_updates=estimate,
                note="Pilot extrapolation excludes diagnostic evaluation, startup and checkpoint overhead",
                policy_forward_samples_per_update=member["attempts"] * training["num_envs"]
                    * (1 + training["ppo_epochs"]))


def run_one(manifest_path, digest, manifest, index, attempt_id, resume=None):
    evidence = allocation()
    if not 0 <= index < len(manifest["members"]):
        raise ValueError("Screen member index is outside the manifest")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}", attempt_id):
        raise ValueError("Use a short path-safe fresh attempt identity")
    member = manifest["members"][index]
    variant = next(v for v in manifest["variants"] if v["label"] == member["variant"])
    root = manifest_path.parent / "members" / f"{index:03d}-{member['variant']}-k{member['attempts']}-seed{member['seed']}" / attempt_id
    root.mkdir(parents=True, exist_ok=False)
    training_dir = root / "training"
    options = dict(source_commit=manifest["source_commit"], output_dir=training_dir,
                   task=member["task"], arms=member["arms"], attempts=member["attempts"],
                   horizon=member["horizon"], seed=member["seed"], core=variant["core"],
                   width=variant["width"], **variant["settings"], **manifest["training"])
    if resume is not None:
        options["resume"] = Path(resume).resolve()
    # Core-specific switches are rejected for other cores by the training CLI.
    if variant["core"] != "persistent_ctm":
        options.pop("bounded_synchronisation")
        options.pop("decay_per_observation")
    command = [sys.executable, "-m", "fluffy_bassoon.meta_rl.train", *command_options(options)]
    receipt = dict(manifest=dict(path=str(manifest_path), sha256=digest), member=member,
                   attempt_id=attempt_id, allocation=evidence, command=command,
                   expected_parameters=variant["parameters"],
                   resumed_from=None if resume is None else dict(path=str(options["resume"]), sha256=sha256(options["resume"])))
    write_json(root / "launch.json", receipt)
    started = time.monotonic()
    with (root / "training.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(root / "failure.json", dict(stage="training", returncode=result.returncode,
                                                elapsed_seconds=time.monotonic() - started))
        raise RuntimeError(f"Screen member {index} training failed; inspect {root / 'training.log'}")
    config = json.loads((training_dir / "config.json").read_text())
    if (config["parameters"] != variant["parameters"]
            or config["retained_state_bytes_per_lane"] != variant["retained_state_bytes_per_lane"]
            or config["contract"]["core_config"] != variant["core_config"]):
        raise RuntimeError("Training policy differs from the matched manifest")
    summary = json.loads((training_dir / "summary.json").read_text())
    receipt.update(training=dict(config_sha256=sha256(training_dir / "config.json"),
                                  summary_sha256=sha256(training_dir / "summary.json"),
                                  checkpoint=summary["checkpoint"], completed=summary["completed"],
                                  iteration=summary["iteration"]),
                   cost=cost_receipt(training_dir, manifest, member))
    if summary["completed"]:
        diagnostics = manifest["diagnostics"]
        command = [sys.executable, "-m", "fluffy_bassoon.meta_rl.evaluate",
                   "--source-commit", manifest["source_commit"], "--checkpoint", str(training_dir / "latest.pt"),
                   "--output-dir", str(root / "diagnostics"), "--trials", str(diagnostics["trials"]),
                   "--device", diagnostics["device"], "--reward-blind", "--policy-seeds",
                   *map(str, diagnostics["policy_seeds"])]
        with (root / "diagnostics.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            write_json(root / "failure.json", dict(stage="diagnostics", returncode=result.returncode,
                                                    elapsed_seconds=time.monotonic() - started))
            raise RuntimeError(f"Screen member {index} diagnostics failed; inspect {root / 'diagnostics.log'}")
        receipt["diagnostics"] = dict(command=command,
                                       summary_sha256=sha256(root / "diagnostics/summary.json"))
    receipt.update(completed=summary["completed"], elapsed_seconds=time.monotonic() - started)
    write_json(root / "receipt.json", receipt)
    print(json.dumps(dict(index=index, attempt_id=attempt_id, completed=receipt["completed"],
                          receipt=str(root / "receipt.json"), cost=receipt["cost"])), flush=True)
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    q = commands.add_parser("plan", help="Create a fresh immutable screen manifest")
    q.add_argument("--source-commit", required=True)
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--parameter-budget", type=int, default=100_000)
    q.add_argument("--minimum-parameter-fraction", type=float, default=.85)
    q.add_argument("--attempts", type=int, nargs="+", default=[64, 128])
    q.add_argument("--arms", type=int, default=6)
    q.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    q.add_argument("--updates", type=int, default=300)
    q.add_argument("--num-envs", type=int, default=32)
    q.add_argument("--learning-rate", type=float, default=3e-4)
    q.add_argument("--reward-offset", type=float, default=.5)
    q.add_argument("--value-coef", type=float, default=.01)
    q.add_argument("--ppo-epochs", type=int, default=4)
    q.add_argument("--minibatch-trials", type=int, default=16)
    q.add_argument("--eval-every", type=int, default=20)
    q.add_argument("--eval-trials", type=int, default=128)
    q.add_argument("--checkpoint-every", type=int, default=10)
    q.add_argument("--max-elapsed-seconds", type=float, default=7 * 24 * 3600)
    q.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    q = commands.add_parser("run", help="Execute one member into a fresh attempt")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--index", type=int, required=True)
    q.add_argument("--attempt-id", required=True)
    q.add_argument("--resume", type=Path)
    q = commands.add_parser("work", help="Sequential worker for one disjoint shard")
    q.add_argument("--manifest", type=Path, required=True)
    q.add_argument("--attempt-id", required=True)
    q.add_argument("--shard-index", type=int, required=True)
    q.add_argument("--shards", type=int, required=True)
    q.add_argument("--indices", type=int, nargs="+", help="Optional bounded subset; default all members")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "plan":
        return plan(args)
    path, digest, manifest = load_manifest(args.manifest)
    if args.command == "run":
        result = run_one(path, digest, manifest, args.index, args.attempt_id, args.resume)
        if not result["completed"]:
            raise SystemExit(3)
        return result
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Require a valid worker shard index/count")
    indices = list(range(len(manifest["members"]))) if args.indices is None else args.indices
    if len(indices) != len(set(indices)) or any(not 0 <= index < len(manifest["members"]) for index in indices):
        raise ValueError("Require distinct valid member indices")
    results = []
    for index in indices:
        if index % args.shards == args.shard_index:
            result = run_one(path, digest, manifest, index, args.attempt_id)
            results.append(result)
            if not result["completed"]:
                raise SystemExit(3)
    return results


if __name__ == "__main__":
    main()
