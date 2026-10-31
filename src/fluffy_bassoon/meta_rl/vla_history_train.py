"""Train a portable history/no-history controller from frozen-VLA teacher data.

Validation NLL is an imitation diagnostic, not evidence of robot success or
in-context adaptation. Evaluate frozen candidates on held-out simulator trials.
"""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .vla_history import (HistoryConfig, HistoryNoisePolicy, MODES, export_policy,
                          load_dataset, sample_batch)
from .warmstart import atomic_save


RESUME_FORMAT = "frozen_vla_history_resume_v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", type=Path, required=True)
    p.add_argument("--train-sha256", required=True)
    p.add_argument("--validation-data", type=Path, required=True)
    p.add_argument("--validation-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mode", choices=MODES, default="full")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--context-length", type=int, default=300)
    p.add_argument("--max-window", type=int, default=128)
    p.add_argument("--full-trial-batches", action="store_true")
    p.add_argument("--max-episodes-per-window", type=int)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--validation-batches", type=int, default=4)
    p.add_argument("--max-elapsed-seconds", type=float, default=1800.)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--source-commit", required=True)
    p.add_argument("--expected-hostname", required=True, help="Exact allocated Linux scheduler hostname")
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint-selection", choices=("validation-nll", "final"), default="validation-nll")
    return p


def supervised_update(policy, optimizer, batch, *, mode):
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    loss = policy.sequence_nll(*batch, mode=mode)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite history-controller imitation loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(value).all() for value in policy.parameters()):
        raise RuntimeError("Non-finite history-controller parameters")
    return dict(nll=float(loss.detach()), gradient_norm=float(norm))


@torch.no_grad()
def validate(policy, trials, *, mode, batch_size, batches, max_context, max_episodes=None):
    policy.eval()
    # The identical held-out windows are compared at every update and baseline.
    sampler = np.random.default_rng(734_001)
    values = [float(policy.sequence_nll(*sample_batch(trials, policy, sampler,
        batch_size=batch_size, max_context=max_context, mode=mode, max_episodes=max_episodes), mode=mode)) for _ in range(batches)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite held-out imitation loss")
    return float(np.mean(values))


def _rng_state(sampler):
    return dict(sampler=sampler.bit_generator.state, python=random.getstate(),
        numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state, sampler):
    sampler.bit_generator.state = state["sampler"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        raise ValueError("Run computation on the explicitly identified allocated Linux host")
    for key in ("updates", "batch_size", "eval_every", "validation_batches", "threads", "max_window"):
        if getattr(args, key) < 1:
            raise ValueError("Training/evaluation counts must be positive")
    if args.max_episodes_per_window is not None and (args.max_episodes_per_window < 1 or args.full_trial_batches):
        raise ValueError("Choose a positive complete-episode window cap or full-trial batches")
    if (not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0):
        raise ValueError("Invalid seed/optimization budget")
    root = Path(__file__).resolve().parents[3]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (source != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require the exact clean reviewed source commit")
    train, train_manifest = load_dataset(args.train_data, expected_sha256=args.train_sha256)
    validation, validation_manifest = load_dataset(args.validation_data, expected_sha256=args.validation_sha256)
    if (args.train_sha256 == args.validation_sha256
            or (train_manifest["observation_dim"], train_manifest["noise_dim"], train_manifest["objective"])
            != (validation_manifest["observation_dim"], validation_manifest["noise_dim"], validation_manifest["objective"])):
        raise ValueError("Require distinct compatible train and validation collections")
    # Collection provenance must expose exactly how held-out trials were separated.
    train_ids = train_manifest["metadata"].get("trial_ids")
    validation_ids = validation_manifest["metadata"].get("trial_ids")
    if (not isinstance(train_ids, list) or not isinstance(validation_ids, list)
            or len(train_ids) != len(train) or len(validation_ids) != len(validation)
            or any(not isinstance(value, str) or not value for value in train_ids + validation_ids)
            or len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids)
            or set(train_ids) & set(validation_ids)):
        raise ValueError("Require disjoint explicit train/validation trial identities in metadata")
    for key in ("motor_sha256", "feature_contract", "noise_contract"):
        if not train_manifest["metadata"].get(key) or train_manifest["metadata"].get(key) != validation_manifest["metadata"].get(key):
            raise ValueError("Require matching explicit frozen motor, feature and noise contracts")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler = np.random.default_rng(args.seed)
    config = HistoryConfig(train_manifest["observation_dim"], train_manifest["noise_dim"],
        width=args.width, layers=args.layers, heads=args.heads, context_length=args.context_length)
    policy = HistoryNoisePolicy(config).to(args.device)
    policy.fit_observation_normalization(train)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate, weight_decay=0.)
    max_context = None if args.full_trial_batches or args.max_episodes_per_window is not None else args.max_window
    contract = dict(source_commit=source, config=asdict(config), mode=args.mode,
        objective=train_manifest["objective"], train_sha256=args.train_sha256,
        validation_sha256=args.validation_sha256, train_metadata=train_manifest["metadata"],
        validation_metadata=validation_manifest["metadata"], seed=args.seed,
        batch_size=args.batch_size, max_context=max_context, learning_rate=args.learning_rate,
        max_episodes_per_window=args.max_episodes_per_window,
        schedule=dict(updates=args.updates, eval_every=args.eval_every,
            validation_batches=args.validation_batches, max_elapsed_seconds=args.max_elapsed_seconds,
            checkpoint_selection=args.checkpoint_selection),
        supervised_objective="Gaussian target-noise NLL; motor adapter applies its fixed noise map",
        status="candidate; imitation NLL does not establish robot performance")
    start_update, best, elapsed_before = 0, math.inf, 0.
    if args.resume:
        # Full trusted local training checkpoint, not an inference artifact.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved.get("format") != RESUME_FORMAT or saved.get("contract") != contract:
            raise ValueError("Resume contract differs")
        policy.load_state_dict(saved["policy"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        _restore_rng(saved["rng"], sampler)
        start_update, best, elapsed_before = saved["update"], saved["best_validation_nll"], saved["elapsed_seconds"]
        if args.output_dir.resolve() != args.resume.parent.resolve():
            raise ValueError("Resume must continue its original output identity")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2, allow_nan=False) + "\n")
    started = time.monotonic()
    for update in range(start_update + 1, args.updates + 1):
        batch = sample_batch(train, policy, sampler, batch_size=args.batch_size,
                             max_context=max_context, mode=args.mode, max_episodes=args.max_episodes_per_window)
        metrics = supervised_update(policy, optimizer, batch, mode=args.mode)
        elapsed = elapsed_before + time.monotonic() - started
        terminal = update == args.updates or elapsed >= args.max_elapsed_seconds
        if update == 1 or update % args.eval_every == 0 or terminal:
            heldout = validate(policy, validation, mode=args.mode, batch_size=args.batch_size,
                batches=args.validation_batches, max_context=max_context, max_episodes=args.max_episodes_per_window)
            row = dict(update=update, elapsed_seconds=elapsed, validation_nll=heldout, **metrics)
            with (args.output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            improved = heldout < best
            best = min(best, heldout)
            if ((args.checkpoint_selection == "validation-nll" and improved)
                    or (args.checkpoint_selection == "final" and update == args.updates)):
                export_policy(args.output_dir / "candidate.pt", policy,
                    manifest=dict(contract=contract, update=update, validation_nll=heldout,
                                  train_transitions=train_manifest["transitions"]))
            atomic_save(args.output_dir / "latest.pt", dict(format=RESUME_FORMAT,
                contract=contract, update=update, policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                rng=_rng_state(sampler), best_validation_nll=best, elapsed_seconds=elapsed))
        if terminal:
            break
    if start_update >= args.updates:
        raise ValueError("Resume has already completed the requested updates")


if __name__ == "__main__":
    main()
