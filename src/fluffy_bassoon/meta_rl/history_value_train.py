"""Independently resumable DIT / in-context IQL value and policy fitting.

This trains on an already verified offline corpus, never on new simulator
interactions. The operational time limit checkpoints both stages; it does not
stand in for validation convergence. Inference exports use the existing frozen
history controller, with no value updates or teacher labels at deployment.
"""
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .history_value import HistoryValueLearner, ValueConfig, learning_step, sample_value_batch
from .vla_history import HistoryConfig, export_policy, load_dataset
from .vla_history_train import parser as base_parser, advance_plateau, _rng_state, _restore_rng
from .warmstart import atomic_save


RESUME_FORMAT = "frozen_history_value_resume_v1"


def parser():
    result = base_parser()
    result.description = __doc__
    result.set_defaults(max_episodes_per_window=9)
    result.add_argument("--objective", choices=("dit", "ic_iql"), required=True)
    result.add_argument("--critic-updates", type=int, default=1000)
    result.add_argument("--discount", type=float, default=.99)
    result.add_argument("--expectile", type=float, default=.7)
    result.add_argument("--temperature", type=float, default=1.)
    result.add_argument("--maximum-weight", type=float, default=100.)
    result.add_argument("--target-tau", type=float, default=.005)
    result.add_argument("--bellman-weight", type=float, default=.1)
    return result


def checked_datasets(args):
    train, tm = load_dataset(args.train_data, expected_sha256=args.train_sha256)
    validation, vm = load_dataset(args.validation_data, expected_sha256=args.validation_sha256)
    if (tm["objective"] != args.objective or vm["objective"] != args.objective
            or args.train_sha256 == args.validation_sha256
            or (tm["observation_dim"], tm["noise_dim"]) != (vm["observation_dim"], vm["noise_dim"])):
        raise ValueError("Require distinct compatible datasets for the actual value objective")
    ids = []
    for trials, manifest in ((train, tm), (validation, vm)):
        group = manifest["metadata"].get("trial_ids")
        if (not isinstance(group, list) or len(group) != len(trials)
                or len(set(group)) != len(group) or any(not isinstance(x, str) or not x for x in group)):
            raise ValueError("Bind every complete train/validation stream identity")
        ids.append(set(group))
    if ids[0] & ids[1]:
        raise ValueError("Training and validation stream identities overlap")
    for key in ("motor_sha256", "feature_contract", "noise_contract", "value_contract"):
        if not tm["metadata"].get(key) or tm["metadata"][key] != vm["metadata"].get(key):
            raise ValueError("Frozen source/interface/value contracts differ")
    vc = tm["metadata"]["value_contract"]
    if vc.get("boundary") != "physical_done" or vc.get("discount") != args.discount:
        raise ValueError("Do not change the source's physical value boundary or discount")
    return train, validation, tm, vm


@torch.no_grad()
def validate(learner, trials, *, phase, batch_size, batches, max_episodes, mode):
    learner.eval()
    sampler = np.random.default_rng(734001)
    loss_function = learner.value_loss if phase == "value" else learner.actor_loss
    losses = []
    for _ in range(batches):
        batch = sample_value_batch(trials, learner.actor, sampler, batch_size=batch_size,
            max_episodes=max_episodes, discount=learner.config.discount, mode=mode)
        losses.append(float(loss_function(batch, mode=mode)[0]))
    if not all(math.isfinite(x) for x in losses):
        raise FloatingPointError("Nonfinite held-out value/policy objective")
    return float(np.mean(losses))


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        raise ValueError("Run computation on the identified allocated Linux host")
    if (args.full_trial_batches or args.max_episodes_per_window is None
            or args.max_episodes_per_window < 1 or args.backbone != "decoder"):
        raise ValueError("Value methods use complete-episode windows and the causal decoder")
    for key in ("updates", "critic_updates", "batch_size", "eval_every", "validation_batches", "threads"):
        if getattr(args, key) < 1:
            raise ValueError("Require positive optimization/evaluation counts")
    if (not 0 <= args.seed < 2**32 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_elapsed_seconds) or args.max_elapsed_seconds <= 0
            or args.execution_batch_size is not None and args.execution_batch_size < 1):
        raise ValueError("Invalid optimization, execution batch or invocation budget")
    advance_plateau({}, 0., 1, minimum_updates=args.minimum_updates,
        patience=args.validation_patience, relative_delta=args.validation_relative_delta)
    if args.until_converged and args.checkpoint_selection != "validation-nll":
        raise ValueError("Converged fits select their held-out objective")
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (head != args.source_commit
            or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()):
        raise ValueError("Require exact clean pinned source")
    train, validation, tm, vm = checked_datasets(args)
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sampler = np.random.default_rng(args.seed)
    config = HistoryConfig(tm["observation_dim"], tm["noise_dim"], width=args.width, layers=args.layers,
        heads=args.heads, context_length=args.context_length, backbone=args.backbone)
    value_config = ValueConfig(args.objective, args.discount, args.expectile, args.temperature,
                              args.maximum_weight, args.target_tau, args.bellman_weight)
    learner = HistoryValueLearner(config, value_config).to(args.device)
    learner.fit_normalization(train)
    optimizers = dict(value=torch.optim.AdamW([*learner.q.parameters(), *learner.v.parameters()],
        lr=args.learning_rate, weight_decay=0.), actor=torch.optim.AdamW(learner.actor.parameters(),
        lr=args.learning_rate, weight_decay=0.))
    contract = dict(source_commit=head, config=asdict(config), value_config=asdict(value_config),
        mode=args.mode, objective=args.objective, train_sha256=args.train_sha256,
        validation_sha256=args.validation_sha256, train_metadata=tm["metadata"], validation_metadata=vm["metadata"],
        seed=args.seed, batch_size=args.batch_size, max_context=None,
        max_episodes_per_window=args.max_episodes_per_window, learning_rate=args.learning_rate,
        schedule=dict(updates=None if args.until_converged else args.updates,
            critic_updates=None if args.until_converged else args.critic_updates,
            eval_every=args.eval_every, validation_batches=args.validation_batches,
            checkpoint_selection=args.checkpoint_selection, until_converged=args.until_converged,
            minimum_updates=args.minimum_updates, validation_patience=args.validation_patience,
            validation_relative_delta=args.validation_relative_delta),
        value_scheme="independent causal Q/V fit, then frozen-value weighted policy extraction",
        method="DIT with MC and conventional TD regularization" if args.objective == "dit" else
               "in-context IQL decoder adaptation; no S-ICQL world-model prompt",
        status="candidate; held-out offline loss does not establish robot success")
    phase, updates, best, plateaus = "value", dict(value=0, actor=0), dict(value=math.inf, actor=math.inf), {}
    best_value_state, elapsed_before = None, 0.
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (saved.get("format") != RESUME_FORMAT or saved.get("contract") != contract
                or args.output_dir.resolve() != args.resume.parent.resolve() or saved["converged"]):
            raise ValueError("Resume must preserve the same incomplete source/learner/output contract")
        learner.load_state_dict(saved["learner"], strict=True)
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])
        _restore_rng(saved["rng"], sampler)
        phase, updates, best, plateaus = (saved[key] for key in ("phase", "updates", "best", "plateaus"))
        best_value_state, elapsed_before = saved["best_value_state"], saved["elapsed_seconds"]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    started, converged = time.monotonic(), False

    def checkpoint():
        elapsed = elapsed_before + time.monotonic() - started
        atomic_save(args.output_dir / "latest.pt", dict(format=RESUME_FORMAT, contract=contract,
            phase=phase, updates=updates, best=best, plateaus=plateaus, best_value_state=best_value_state,
            learner=learner.state_dict(), optimizers={k: v.state_dict() for k, v in optimizers.items()},
            rng=_rng_state(sampler), elapsed_seconds=elapsed, converged=converged))
        status = dict(converged=converged, phase=phase, update=updates["actor"], updates=updates,
            best_validation_loss={k: v if math.isfinite(v) else None for k, v in best.items()}, elapsed_seconds=elapsed,
            stop_reason="validation_plateau" if converged and args.until_converged else
                        "fixed_updates" if converged else "operational_checkpoint")
        temporary = args.output_dir / "training-status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.output_dir / "training-status.json")

    while True:
        learner.train()
        if phase == "actor":
            learner.q.requires_grad_(False); learner.v.requires_grad_(False)
        batch = sample_value_batch(train, learner.actor, sampler, batch_size=args.batch_size,
            max_episodes=args.max_episodes_per_window, discount=args.discount, mode=args.mode)
        metrics = learning_step(learner, optimizers[phase], batch, phase=phase, mode=args.mode,
                                execution_batch_size=args.execution_batch_size)
        updates[phase] += 1
        step = updates[phase]
        limit = args.critic_updates if phase == "value" else args.updates
        timed_out = time.monotonic() - started >= args.max_elapsed_seconds
        scheduled = step == 1 or step % args.eval_every == 0
        fixed_end = not args.until_converged and step >= limit
        if scheduled or timed_out or fixed_end:
            heldout = validate(learner, validation, phase=phase, batch_size=args.batch_size,
                batches=args.validation_batches, max_episodes=args.max_episodes_per_window, mode=args.mode)
            improved = heldout < best[phase]
            best[phase] = min(best[phase], heldout)
            if args.until_converged and scheduled:
                plateaus[phase] = advance_plateau(plateaus.get(phase, {}), heldout, step,
                    minimum_updates=args.minimum_updates, patience=args.validation_patience,
                    relative_delta=args.validation_relative_delta)
            phase_complete = fixed_end or plateaus.get(phase, {}).get("converged", False)
            if phase == "value" and improved:
                best_value_state = {k: v.detach().cpu().clone() for k, v in learner.state_dict().items()}
            if phase == "actor" and (improved if args.checkpoint_selection == "validation-nll" else fixed_end):
                export_policy(args.output_dir / "candidate.pt", learner.actor, manifest=dict(contract=contract,
                    update=step, value_updates=updates["value"], validation_loss=heldout,
                    validation_nll=heldout, metric="frozen-value weighted Gaussian NLL",
                    train_transitions=tm["transitions"]))
            row = dict(phase=phase, update=step, validation_loss=heldout, **metrics)
            with (args.output_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            if phase_complete and phase == "value":
                learner.load_state_dict(best_value_state, strict=True)
                phase = "actor"
            elif phase_complete:
                converged = True
            checkpoint()
            if timed_out or converged:
                break


if __name__ == "__main__":
    main()
