"""CMM few-shot linear regression calibration (arXiv:2610.07907 §4.3, App. D.5, Table 2).

Calibrates the repository's CMM port against published numbers before any CMM
comparison is interpreted. Paper rows at K=20 (query accuracy %, 3 seeds): CMM
30.63±0.24, CMM-STM 13.03±0.61, CTM-Ext. 8.00±0.07; at K=10: 9.86, 7.28, 5.84.

Stated by the paper: n=12 inputs in [-1,1]; a fresh linear function per episode;
K noisy support pairs then 20 queries; clean targets standardized per episode
(zero mean, unit sample std) before support noise; per-step input = 12 inputs,
observed value, support bit -> learned linear projection + ReLU; MSE over the
full episode; AdamW 1e-4, weight decay 0, batch 64, clip 5, no dropout, 10,000
fresh episodes per epoch, 1,000 epochs; metric = share of query predictions with
|y_hat - y| < 0.05; 10,000 test episodes; final-epoch result; seeds 0, 1, 2.

Unstated, chosen here and recorded in the contract: inputs uniform on [-1,1];
weights i.i.d. N(0,1) with no bias; support noise std 0.1 (reading N(0, 0.01) as
a variance; --noise-std changes it); query steps carry observed value 0; the
full-episode MSE targets the clean standardized value at every step; 4 heads;
one tick per input; width = largest multiple of 4 within the paper's parameter
count for that variant.
"""

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


INPUTS, QUERIES, THRESHOLD = 12, 20, .05
EPISODES_PER_EPOCH, TEST_EPISODES, TEST_SEED = 10_000, 10_000, 2_000_000_000
# Paper parameter counts (App. D.5) and the variant's memory layout.
VARIANTS = {
    "cmm": dict(target=496_129, stm_length=10, ltm_tokens=32, layers=1),
    "cmm_stm": dict(target=495_713, stm_length=42, ltm_tokens=0, layers=1),
    "ctm_ext": dict(target=499_585, stm_length=42, ltm_tokens=0, layers=0),
}


def training_batch_size(iteration, batch_size):
    """Keep the final partial batch so an epoch contains exactly 10,000 episodes."""
    updates = math.ceil(EPISODES_PER_EPOCH / batch_size)
    return min(batch_size, EPISODES_PER_EPOCH - (iteration % updates) * batch_size)


def regression_batch(rng, batch, shots, noise_std, device="cpu"):
    """Returns inputs [T,B,14], clean standardized targets [T,B], query mask [T]."""
    if any(type(v) is not int or v < 1 for v in (batch, shots)) or not noise_std >= 0:
        raise ValueError("Require positive batch/shots and nonnegative noise")
    steps = shots + QUERIES
    x = rng.uniform(-1, 1, (batch, steps, INPUTS))
    weights = rng.standard_normal((batch, INPUTS))
    y = np.einsum("bti,bi->bt", x, weights)
    y = (y - y.mean(1, keepdims=True)) / y.std(1, ddof=1, keepdims=True)
    observed = np.zeros_like(y)
    observed[:, :shots] = y[:, :shots] + noise_std * rng.standard_normal((batch, shots))
    support = np.zeros_like(y)
    support[:, :shots] = 1
    inputs = np.concatenate((x, observed[..., None], support[..., None]), -1)
    query = torch.zeros(steps, dtype=torch.bool, device=device)
    query[shots:] = True
    as_tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device).transpose(0, 1).contiguous()
    return as_tensor(inputs), as_tensor(y), query


class RegressionModel(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.input = nn.Sequential(nn.Linear(INPUTS + 2, core.input_dim), nn.ReLU())
        self.output = nn.Linear(core.output_dim, 1)

    def forward(self, sequence):
        state, predictions = None, []
        for observation in sequence:
            features, state = self.core(self.input(observation), state)
            predictions.append(self.output(features).squeeze(-1))
        return torch.stack(predictions)


def build(variant, width):
    layout = VARIANTS[variant]
    core = CMMCore(width, width=width, stm_length=layout["stm_length"], ltm_tokens=layout["ltm_tokens"],
                   layers=layout["layers"], heads=4, synapse_depth=8, ticks=1)
    return RegressionModel(core)


def parameters(model):
    return sum(p.numel() for p in model.parameters())


def select_width(variant):
    """Largest multiple of 4 whose total parameter count stays within the paper's."""
    target, best = VARIANTS[variant]["target"], None
    for width in range(16, 1024, 4):
        count = parameters(build(variant, width))
        if count > target:
            break
        best = (width, count)
    if best is None:
        raise ValueError("No width fits the paper's parameter count")
    return best


def query_metrics(predictions, targets, query):
    error = (predictions[query] - targets[query]).abs()
    return dict(query_accuracy=float((error < THRESHOLD).float().mean()),
                query_mse=float(error.square().mean()), queries=int(error.numel()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="cmm")
    parser.add_argument("--shots", type=int, choices=(10, 20), default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, help="Override the parameter-matched width")
    parser.add_argument("--noise-std", type=float, default=.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-every-epochs", type=int, default=25)
    parser.add_argument("--dev-episodes", type=int, default=1000)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=5)
    parser.add_argument("--max-elapsed-seconds", type=float, default=7*24*3600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int,
                        help="Execution-only CPU thread count; defaults to allocated Slurm CPUs or one")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run calibration on allocated Linux resources")
    execution = os.environ.get('FLUFFY_GPU_EXECUTION', 'slurm')
    if execution not in ('slurm', 'baige'):
        raise ValueError('Calibration requires allocated Slurm or Baige execution')
    validate_execution(execution, os.environ)
    if (any(getattr(args, n) < 1 for n in ("epochs", "batch_size", "eval_every_epochs", "dev_episodes",
                                            "checkpoint_every_epochs"))
            or not 0 <= args.seed < 2**31 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.noise_std) or args.noise_std < 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0):
        raise ValueError("Invalid counts, seed, learning rate, noise or time cap")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean source")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Require exactly one allocated GPU for --device cuda")
    threads = args.torch_threads if args.torch_threads is not None else int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if threads < 1:
        raise ValueError("Require a positive Torch thread count")
    torch.set_num_threads(threads)
    width, matched_count = select_width(args.variant)
    if args.width is not None:
        width = args.width
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build(args.variant, width).to(args.device)
    updates_per_epoch = math.ceil(EPISODES_PER_EPOCH / args.batch_size)
    target_updates = args.epochs * updates_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.)
    contract = dict(source_commit=source, task="few-shot linear regression", variant=args.variant,
        layout=VARIANTS[args.variant], width=width, heads=4, synapse_depth=8, ticks=1, shots=args.shots,
        queries=QUERIES, inputs=INPUTS, noise_std=args.noise_std, seed=args.seed,
        batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=0., gradient_clip=5.,
        epochs=args.epochs, episodes_per_epoch=EPISODES_PER_EPOCH, updates_per_epoch=updates_per_epoch,
        threshold=THRESHOLD, test_episodes=TEST_EPISODES, test_seed=TEST_SEED, device=args.device,
        unstated_choices=dict(inputs="uniform[-1,1]", weights="iid N(0,1), no bias",
            noise="std (reading N(0,0.01) as variance)", query_observed_value=0.,
            loss="MSE to clean standardized target at every step", heads=4, ticks=1,
            width="largest multiple of 4 within the paper parameter count"),
        paper_parameters=VARIANTS[args.variant]["target"])
    iteration, counters = 0, dict(training_episodes=0, evaluation_episodes=0)
    if args.resume:
        iteration, counters = restore_checkpoint(args.resume, policy=model, optimizer=optimizer,
                                                  environment_rng=rng, contract=contract)
    if iteration >= target_updates:
        raise ValueError("Target updates must exceed restored iteration")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "config.json", dict(contract=contract, target_updates=target_updates,
               parameters=parameters(model), parameter_matched_width=width if args.width is None else None,
               parameter_matched_count=matched_count,
               job_id=os.environ['SLURM_JOB_ID'] if execution == 'slurm' else os.uname().nodename.removesuffix('-master-0'),
               resumed_from=None if args.resume is None else dict(path=str(args.resume), sha256=sha256(args.resume)),
               hostname=os.uname().nodename, runtime=dict(torch=torch.__version__, python=sys.version,
                   torch_threads=torch.get_num_threads(), cuda=torch.version.cuda,
                   device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else None)))
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        signal.signal(getattr(signal, name), request_stop)
    started = time.monotonic()
    timing = dict(training_seconds=0., evaluation_seconds=0., checkpoint_seconds=0.,
                  data_host_seconds=0., forward_host_seconds=0., backward_host_seconds=0.,
                  optimizer_host_seconds=0.)

    def checkpoint():
        before = time.monotonic()
        digest = save_checkpoint(args.output_dir / "latest.pt", policy=model, optimizer=optimizer,
                               environment_rng=rng, iteration=iteration, counters=counters, contract=contract)
        timing["checkpoint_seconds"] += time.monotonic() - before
        return digest

    @torch.no_grad()
    def evaluate(episodes, seed):
        before = time.monotonic()
        model.eval()
        generator, totals = np.random.default_rng(seed), dict(correct=0., squared=0., queries=0)
        for start in range(0, episodes, 500):
            inputs, targets, query = regression_batch(generator, min(500, episodes - start), args.shots,
                                                      args.noise_std, args.device)
            metrics = query_metrics(model(inputs), targets, query)
            totals["correct"] += metrics["query_accuracy"] * metrics["queries"]
            totals["squared"] += metrics["query_mse"] * metrics["queries"]
            totals["queries"] += metrics["queries"]
        counters["evaluation_episodes"] += episodes
        elapsed = time.monotonic() - before
        timing["evaluation_seconds"] += elapsed
        return dict(iteration=iteration, epoch=iteration / updates_per_epoch, episodes=episodes,
                    elapsed_seconds=elapsed,
                    query_accuracy=totals["correct"] / totals["queries"],
                    query_mse=totals["squared"] / totals["queries"])

    digest = checkpoint()
    last = evaluate(args.dev_episodes, TEST_SEED + 1)
    with (args.output_dir / "metrics.jsonl").open("x", buffering=1) as log:
        log.write(json.dumps(dict(iteration=iteration, development=last)) + "\n")
        while iteration < target_updates and not stopped and time.monotonic() - started < args.max_elapsed_seconds:
            before = time.monotonic()
            # Sample stream timing without synchronizing each individual phase.
            # The final event is synchronized once after the complete update.
            sample_cuda = args.device == "cuda" and (iteration < 5 or (iteration + 1) % updates_per_epoch == 0)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if sample_cuda else None
            actual_batch = training_batch_size(iteration, args.batch_size)
            inputs, targets, _ = regression_batch(rng, actual_batch, args.shots, args.noise_std, args.device)
            model.train()
            data_done = time.monotonic()
            if events:
                events[0].record()
            loss = (model(inputs) - targets).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite regression loss")
            optimizer.zero_grad(set_to_none=True)
            forward_done = time.monotonic()
            if events:
                events[1].record()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            backward_done = time.monotonic()
            if events:
                events[2].record()
            optimizer.step()
            model.core.constrain_parameters()
            if events:
                events[3].record()
            iteration += 1
            counters["training_episodes"] += actual_batch
            record = dict(iteration=iteration, loss=float(loss.detach()), gradient_norm=float(norm))
            if events:
                events[-1].synchronize()
                record["cuda_stream_seconds"] = dict(
                    forward=events[0].elapsed_time(events[1]) / 1000,
                    backward_and_clip=events[1].elapsed_time(events[2]) / 1000,
                    optimizer_and_projection=events[2].elapsed_time(events[3]) / 1000)
            done = time.monotonic()
            record["update_seconds"] = done - before
            timing["training_seconds"] += done - before
            for key, elapsed in (("data_host_seconds", data_done - before),
                                 ("forward_host_seconds", forward_done - data_done),
                                 ("backward_host_seconds", backward_done - forward_done),
                                 ("optimizer_host_seconds", done - backward_done)):
                timing[key] += elapsed
            if iteration % updates_per_epoch == 0:
                epoch = iteration // updates_per_epoch
                record["timing"] = dict(timing)
                record["timing_note"] = "Host phase times include submission and existing scalar waits; CUDA stream samples time the complete phases"
                if epoch % args.eval_every_epochs == 0:
                    last = evaluate(args.dev_episodes, TEST_SEED + 1)
                    record["development"] = last
                    print(json.dumps(record, allow_nan=False), flush=True)
                if epoch % args.checkpoint_every_epochs == 0:
                    digest = checkpoint()
            log.write(json.dumps(record, allow_nan=False) + "\n")
    digest = checkpoint()
    completed = iteration == target_updates
    final = evaluate(TEST_EPISODES, TEST_SEED) if completed else None
    result = dict(completed=completed, iteration=iteration, source_commit=source,
                  elapsed_seconds=time.monotonic() - started, counters=counters,
                  timing=timing,
                  checkpoint=dict(path=str(args.output_dir / "latest.pt"), sha256=digest),
                  last_development=last, final_test=final)
    write_json(args.output_dir / "summary.json", result)
    return result


if __name__ == "__main__":
    main()
