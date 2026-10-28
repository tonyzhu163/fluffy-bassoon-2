"""DPT-inspired flow-matching warm start from immutable native interaction histories."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge
from .bridge_distillation import HISTORY_FORMAT, distillation_loss, load_history, update_distillation
from .checkpoint import restore_checkpoint, save_checkpoint
from .maniskill_rotation import validate_cpu_execution
from .warmstart import NATIVE, export_warmstart


def read_manifest(path, digest):
    if sha256(path) != digest:
        raise ValueError("History manifest checksum differs")
    manifest = json.loads(path.read_text())
    if manifest.get("format") != HISTORY_FORMAT or len(manifest.get("histories", [])) < 2:
        raise ValueError("Require at least two complete batches for disjoint training/validation histories")
    names = [item["file"] for item in manifest["histories"]]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("History filenames must be unique local basenames")
    if len({item["sha256"] for item in manifest["histories"]}) != len(names):
        raise ValueError("Duplicate histories cannot straddle train/validation")
    return manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--histories", type=Path, required=True, help="manifest.json exported by the live collector")
    p.add_argument("--histories-sha256", required=True)
    p.add_argument("--core", choices=("persistent_ctm", "transformer_xl"), default="persistent_ctm")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--ticks", type=int)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-elapsed-seconds", type=float, default=1800.)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Native warm-start computation requires allocated Linux resources")
    validate_cpu_execution(os.environ, os.uname().nodename, args.expected_hostname)
    args.ticks = (4 if args.core == "persistent_ctm" else 1) if args.ticks is None else args.ticks
    if (any(getattr(args, key) < 1 for key in ("updates", "eval_every", "width", "ticks", "threads"))
            or not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0):
        raise ValueError("Invalid native distillation configuration")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Require one allocated visible GPU")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Require the exact clean reviewed source")
    manifest = read_manifest(args.histories, args.histories_sha256)
    schema = manifest["compatibility"]
    # Every fifth complete trial batch validates imitation; with 2-4 batches reserve the last.
    validation = [i for i in range(len(manifest["histories"])) if i % 5 == 4] or [len(manifest["histories"]) - 1]
    training = [i for i in range(len(manifest["histories"])) if i not in validation]
    def history(index):
        entry = manifest["histories"][index]
        return load_history(args.histories.parent / entry["file"], entry["sha256"], schema)
    # Validate every shard before allocating output or mutating model weights.
    for index in range(len(manifest["histories"])):
        history(index)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = np.random.default_rng(args.seed)
    bridge_config = dict(core=args.core, width=args.width, ticks=args.ticks,
                         residual_scale=schema["bridge"]["residual_scale"])
    policy = StochasticPi05BridgePolicy(Pi05FeatureBridge(**bridge_config),
        noise_scale=schema["gaussian_policy"]["noise_scale"],
        active_dims=schema["gaussian_policy"]["active_dims"]).to(args.device).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    provenance = dict(manifest_sha256=args.histories_sha256, histories=manifest["histories"],
                      train_indices=training, validation_indices=validation,
                      split="complete batches; no held-out simulator evaluation data")
    contract = dict(schema, core=args.core, bridge=bridge_config, gaussian_policy=policy.config(),
        source_commit=source, phase="native_dpt_flow_warmstart", seed=args.seed,
        device=args.device, learning_rate=args.learning_rate, data=provenance)
    counters = dict(teacher_simulator_steps=sum(item["simulator_steps"] for item in manifest["histories"]),
                    optimizer_steps=0, replayed_policy_chunks=0, validation_policy_chunks=0)
    iteration = 0
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=policy, optimizer=optimizer,
                                                  environment_rng=sampler, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target must exceed the restored iteration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", dict(contract=contract, target_updates=args.updates,
        manifest_path=str(args.histories.resolve()), hostname=os.uname().nodename))
    started, stop = time.monotonic(), False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for name in ("SIGTERM", "SIGINT", "SIGUSR1"):
        signal.signal(getattr(signal, name), request_stop)
    def checkpoint():
        return save_checkpoint(args.output_dir / "latest.pt", policy=policy, optimizer=optimizer,
            environment_rng=sampler, iteration=iteration, counters=counters, contract=contract)
    @torch.no_grad()
    def evaluation():
        losses, sizes = [], []
        for index in validation:
            batch = history(index)
            losses.append(float(distillation_loss(policy, batch)))
            sizes.append(batch["rewards"].numel())
        counters["validation_policy_chunks"] += sum(sizes)
        return dict(validation_flow_matching_loss=float(np.average(losses, weights=sizes)),
                    behavioral_status="not measured; run frozen live normal/memory_wiped/reward_blind evaluation")
    checkpoint()
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        result = evaluation()
        log.write(json.dumps(dict(iteration=iteration, evaluation=result)) + "\n")
        while iteration < args.updates and not stop and time.monotonic() - started < args.max_elapsed_seconds:
            index = int(sampler.choice(training))
            batch = history(index)
            metrics = update_distillation(policy, optimizer, batch)
            iteration += 1
            counters["optimizer_steps"] += 1
            counters["replayed_policy_chunks"] += batch["rewards"].numel()
            row = dict(iteration=iteration, history_index=index, **metrics, elapsed_seconds=time.monotonic() - started)
            if iteration % args.eval_every == 0:
                result = evaluation()
                row["evaluation"] = result
                checkpoint()
            log.write(json.dumps(row, allow_nan=False) + "\n")
            if iteration % args.eval_every == 0:
                print(json.dumps(row, allow_nan=False), flush=True)
        result = evaluation()
        resume_digest = checkpoint()
        digest = export_warmstart(args.output_dir / "warmstart.pt", family=NATIVE, policy=policy,
            contract=contract, iteration=iteration, counters=counters, data=provenance, evaluation=result) if iteration else None
        write_json(args.output_dir / "summary.json", dict(completed=iteration == args.updates,
            iteration=iteration, counters=counters, evaluation=result, warmstart_sha256=digest,
            resume_checkpoint_sha256=resume_digest, elapsed_seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
