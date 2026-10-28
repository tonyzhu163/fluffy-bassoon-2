"""Reduced copy-task qualification for the paper-based recurrence, not Copy-100 reproduction."""

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
from torch import nn

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.gpu_execution import validate_execution
from .checkpoint import restore_checkpoint, save_checkpoint
from .cmm import CMMCore
from .ppo import GRUCore, LoopedTransformerCore


def copy_batch(rng, batch, length, dimensions, device="cpu"):
    if any(type(value) is not int or value < 1 for value in (batch, length, dimensions)):
        raise ValueError("Require positive integer batch, length and dimensions")
    targets = torch.as_tensor(rng.uniform(-1, 1, (length, batch, dimensions)),
                              dtype=torch.float32, device=device)
    inputs = targets.new_zeros(2 * length, batch, dimensions + 2)
    inputs[:length, :, :dimensions] = targets
    inputs[:length, :, dimensions] = 1
    inputs[length:, :, dimensions + 1] = 1
    return inputs, targets


class CopyModel(nn.Module):
    def __init__(self, core, dimensions):
        super().__init__()
        self.core = core
        self.input = nn.Linear(dimensions + 2, core.input_dim)
        self.output = nn.Linear(core.output_dim, dimensions)

    def forward(self, sequence, *, reset_before_query=False):
        length, state, predictions = sequence.shape[0] // 2, None, []
        for index, observation in enumerate(sequence):
            if reset_before_query and index == length:
                state = None
            features, state = self.core(self.input(observation), state)
            if index >= length:
                predictions.append(self.output(features))
        return torch.stack(predictions)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core", choices=("gru", "ctm", "cmm", "looped_transformer"), default="cmm")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--dimensions", type=int, default=4)
    parser.add_argument("--ltm-tokens", type=int, default=16)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-trials", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-elapsed-seconds", type=float, default=1800)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run memory qualification on allocated Linux resources")
    validate_execution("slurm", os.environ)
    if any(getattr(args, name) < 1 for name in ("width", "length", "dimensions", "updates", "batch_size",
                                               "eval_every", "eval_trials", "checkpoint_every")):
        raise ValueError("Counts must be positive")
    if (not 0 <= args.seed < 2**32 or args.seed == 1_000_000_000
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0):
        raise ValueError("Invalid seed, learning rate or time cap")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean source")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    config = dict(width=args.width, ticks=1)
    if args.core == "looped_transformer":
        config.update(memory_tokens=8)
    elif args.core != "gru":
        config.update(stm_length=10, ltm_tokens=args.ltm_tokens, synapse_depth=8)
        if args.core == "ctm":
            config.update(layers=0, ltm_tokens=0)
    factory = {"gru": GRUCore, "looped_transformer": LoopedTransformerCore}.get(args.core, CMMCore)
    core = factory(64, **config)
    model = CopyModel(core, args.dimensions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    contract = dict(source_commit=source, core=args.core, core_config=config, seed=args.seed,
                    length=args.length, dimensions=args.dimensions, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, input_projection=64, device="cpu", dtype="float32",
                    optimizer="AdamW", weight_decay=0., gradient_clip=5., eval_trials=args.eval_trials,
                    eval_seed=1_000_000_000)
    iteration, counters = 0, dict(training_sequences=0, training_steps=0, evaluation_sequences=0)
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=model, optimizer=optimizer,
                                                  environment_rng=rng, contract=contract)
    if iteration >= args.updates:
        raise ValueError("Target updates must exceed restored iteration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state = core.initial_state(1)
    tensors = state.values() if isinstance(state, dict) else (state,)
    write_json(args.output_dir / "config.json", dict(contract=contract, target_updates=args.updates,
               parameters=sum(p.numel() for p in model.parameters()),
               retained_state_bytes_per_lane=sum(t.numel() * t.element_size() for t in tensors),
               max_elapsed_seconds=args.max_elapsed_seconds, eval_every=args.eval_every,
               checkpoint_every=args.checkpoint_every, job_id=os.environ["SLURM_JOB_ID"],
               resumed_from=None if args.resume is None else dict(path=str(args.resume), sha256=sha256(args.resume)),
               hostname=os.uname().nodename, runtime=dict(torch=torch.__version__, python=sys.version)))
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        signal.signal(getattr(signal, name), request_stop)
    started = time.monotonic()
    def checkpoint():
        return save_checkpoint(args.output_dir / "latest.pt", policy=model, optimizer=optimizer,
                               environment_rng=rng, iteration=iteration, counters=counters, contract=contract)
    @torch.no_grad()
    def evaluate():
        inputs, targets = copy_batch(np.random.default_rng(1_000_000_000), args.eval_trials,
                                     args.length, args.dimensions)
        model.eval()
        normal = model(inputs)
        reset = model(inputs, reset_before_query=True)
        counters["evaluation_sequences"] += 2 * args.eval_trials
        return dict(iteration=iteration, mse=float((normal - targets).square().mean()),
                    reset_before_query_mse=float((reset - targets).square().mean()),
                    zero_predictor_mse=float(targets.square().mean()))
    digest = checkpoint()
    last_evaluation = evaluate()
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        log.write(json.dumps(dict(iteration=iteration, evaluation=last_evaluation)) + "\n")
        while iteration < args.updates and not stopped and time.monotonic() - started < args.max_elapsed_seconds:
            inputs, targets = copy_batch(rng, args.batch_size, args.length, args.dimensions)
            model.train()
            loss = (model(inputs) - targets).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite copy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if hasattr(core, "constrain_parameters"):
                core.constrain_parameters()
            iteration += 1
            counters["training_sequences"] += args.batch_size
            counters["training_steps"] += 2 * args.length * args.batch_size
            record = dict(iteration=iteration, loss=float(loss.detach()), gradient_norm=float(norm))
            if iteration % args.eval_every == 0 or iteration == args.updates:
                last_evaluation = evaluate()
                record["evaluation"] = last_evaluation
                print(json.dumps(record, allow_nan=False), flush=True)
            if iteration % args.checkpoint_every == 0:
                digest = checkpoint()
            log.write(json.dumps(record, allow_nan=False) + "\n")
    digest = checkpoint()
    result = dict(completed=iteration == args.updates, iteration=iteration, source_commit=source,
                  elapsed_seconds=time.monotonic() - started, counters=counters,
                  checkpoint=dict(path=str(args.output_dir / "latest.pt"), sha256=digest),
                  last_evaluation=last_evaluation)
    write_json(args.output_dir / "summary.json", result)
    return result


if __name__ == "__main__":
    main()
