"""Isolated checkpoint diagnostics; never resume or modify a comparison run."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .checkpoint import restore_checkpoint, save_checkpoint
from .metaworld_env import benchmark_pools, task_digest
from .metaworld_policy import make_policy
from .metaworld_workers import make_worker_lanes
from .ppo import collect_trials, generalized_advantages, replay_windows, update_ppo
from .screen import allocation, clean_source


def tensor_stats(value):
    value = value.detach().double()
    if not value.numel():
        return {"count": 0}
    if not torch.isfinite(value).all():
        raise FloatingPointError("Nonfinite diagnostic input")
    return dict(count=value.numel(), mean=float(value.mean()), std=float(value.std(unbiased=False)),
                rms=float(value.square().mean().sqrt()), minimum=float(value.min()), maximum=float(value.max()))


def explained_variance(target, value):
    variance = target.double().var(unbiased=False)
    return (float(1 - (target.double() - value.double()).var(unbiased=False) / variance)
            if variance > 1e-12 else None)


def advantage_data(rollout, optimization):
    advantages, returns = generalized_advantages(rollout["rewards"], rollout["values"],
        rollout["trial_ends"], gamma=optimization["gamma"], gae_lambda=optimization["gae_lambda"])
    mask = rollout["action_mask"]
    normalized = advantages
    if optimization["normalize_advantages"]:
        active = advantages[mask]
        normalized = (advantages - active.mean()) / (active.std(unbiased=False) + 1e-8)
    return advantages, torch.where(mask, normalized, 0.), returns


def _loss_terms(policy, rollout, optimization, lanes):
    _, advantages, returns = advantage_data(rollout, optimization)
    data, states, _ = replay_windows(dict(rollout, advantages=advantages, returns=returns),
                                    optimization["replay_chunk_length"])
    initial = ({k: v[lanes] for k, v in states.items()} if isinstance(states, dict)
               else states[lanes] if states is not None else None)
    log_prob, entropy, value = policy.evaluate_sequence(data["observations"][:, lanes],
        data["raw_actions"][:, lanes], data["resets"][:, lanes], initial_state=initial)
    mask, old_value = data["action_mask"][:, lanes], data["values"][:, lanes]
    ratio = torch.where(mask, log_prob - data["log_probs"][:, lanes], 0.).exp()
    advantage, target = data["advantages"][:, lanes], data["returns"][:, lanes]
    clip = optimization["clip"]
    actor = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(1-clip, 1+clip)).mean()
    clipped = old_value + (value - old_value).clamp(-clip, clip)
    critic = .5 * torch.maximum((value-target).square(), (clipped-target).square()).mean()
    actor = actor - optimization["entropy_coef"] * torch.where(mask, entropy, 0.).mean()
    return actor, optimization["value_coef"] * critic


def gradient_components(policy, rollout, optimization, *, all_windows=False):
    """Measure an actual cached minibatch, with the production loss's weighting."""
    policy.eval()
    length, batch = rollout["rewards"].shape
    chunks = length // optimization["replay_chunk_length"] if optimization["replay_chunk_length"] else 1
    count = batch*chunks
    # Independent generator: diagnostic ordering never consumes the learner RNG.
    order = torch.randperm(count, generator=torch.Generator().manual_seed(9187))
    lanes = order[:count if all_windows else optimization["minibatch_trials"]*chunks]
    lanes = lanes.to(rollout["rewards"].device)
    actor, critic = _loss_terms(policy, rollout, optimization, lanes)
    parameters = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    a = torch.autograd.grad(actor, [p for _, p in parameters], allow_unused=True, retain_graph=True)
    c = torch.autograd.grad(critic, [p for _, p in parameters], allow_unused=True)

    def group(prefix):
        aa, cc, ac = 0., 0., 0.
        for (name, _), x, y in zip(parameters, a, c, strict=True):
            if prefix and not name.startswith(prefix):
                continue
            if x is not None:
                aa += float(x.double().square().sum())
            if y is not None:
                cc += float(y.double().square().sum())
            if x is not None and y is not None:
                ac += float((x.double()*y.double()).sum())
        return dict(actor_norm=math.sqrt(aa), weighted_critic_norm=math.sqrt(cc),
                    joint_norm=math.sqrt(max(0., aa+cc+2*ac)),
                    cosine=ac/math.sqrt(aa*cc) if aa*cc > 0 else None)

    result = dict(samples=len(lanes)*(length//chunks), actor_loss=float(actor.detach()),
                  weighted_critic_loss=float(critic.detach()),
                  groups={name: group(prefix) for name, prefix in
                          (("all", ""), ("core", "core."), ("actor_head", "actor."), ("critic_head", "critic."))})
    norm = result["groups"]["all"]["joint_norm"]
    result["global_clip_multiplier"] = min(1., optimization["max_grad_norm"]/(norm+1e-6))
    result["interpretation"] = "Gradient decomposition only; Adam step size is not the clipping multiplier."
    return result


@torch.no_grad()
def replay_drift(policy, rollout, chunk_length):
    """Fresh current-policy prefixes versus saved behavior-policy boundary states."""
    full = policy.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])
    data, states, _ = replay_windows(rollout, chunk_length)
    cached = policy.evaluate_sequence(data["observations"], data["raw_actions"], data["resets"], initial_state=states)
    result = {}
    for name, reference, observed in (("log_probability", full[0], cached[0]), ("value", full[2], cached[2])):
        expected = replay_windows(dict(rollout, diagnostic=reference), chunk_length)[0]["diagnostic"]
        difference = observed-expected
        result[name] = dict(mean_absolute=float(difference.abs().mean()),
                            rms=float(difference.square().mean().sqrt()), maximum_absolute=float(difference.abs().max()))
    result["interpretation"] = "Saved-action log-probability difference, not distribution KL."
    return result


def family_scales(rollout, families, optimization):
    raw, normalized, targets = advantage_data(rollout, optimization)
    result = {}
    for family in sorted(set(families)):
        lanes = [i for i, f in enumerate(families) if f == family]
        result[family] = dict(rewards=tensor_stats(rollout["rewards"][:, lanes]),
            advantages=tensor_stats(raw[:, lanes]), normalized_advantages=tensor_stats(normalized[:, lanes]),
            gae_targets=tensor_stats(targets[:, lanes]), values=tensor_stats(rollout["values"][:, lanes]),
            value_explained_variance=explained_variance(targets[:, lanes], rollout["values"][:, lanes]))
    return result


def parameter_scales(policy):
    result = {}
    for name, value in policy.named_parameters():
        if "decay_params" not in name:
            continue
        rates = value.detach().clamp(0, 15)
        positive = rates[rates > 1e-6]
        result[name] = dict(**tensor_stats(rates), fraction_at_zero=float((rates <= 1e-6).float().mean()),
            positive_rate_half_life_steps=(tensor_stats(math.log(2)/positive) if positive.numel() else None))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    parent = args.parent_dir
    summary = json.loads((parent/"summary.json").read_text())
    config = json.loads((parent/"config.json").read_text())
    contract = config["contract"]
    if (not summary["completed"] or summary["iteration"] != 80
            or (parent.parent/"exit-code.txt").read_text().strip() != "0"
            or contract["benchmark"] != "ml10" or contract["core"] not in ("persistent_ctm", "cmm")
            or contract["num_envs"] != 50 or contract["optimization"]["replay_chunk_length"] != 100):
        raise ValueError("Require a terminal, successful, unchanged CTM/CMM ML10 comparison")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir/"parent-complete.pt"
    shutil.copyfile(parent/"latest.pt", checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != summary["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint checksum mismatch")
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools("ml10", contract["benchmark_seed"])
    if {k: task_digest(v[1]) for k,v in pools.items()} != contract["pool_hashes"]:
        raise ValueError("Parent benchmark pools differ")
    envs = make_worker_lanes(pools["train"], 50, benchmark="ml10", attempts=5, horizon=500,
                             workers=8, allocated_cpus=16)
    report = dict(diagnostic_source=source, parent_source=contract["source_commit"],
                  parent_checkpoint_sha256=digest, allocation=evidence,
                  interpretation="Isolated diagnostic update; excluded from the 10M comparison and its curves.")
    try:
        policy = make_policy(contract["core"], contract["core_config"], envs[0]).cuda().eval()
        options = {k:v for k,v in contract["optimizer"].items() if k != "name"}
        optimizer = torch.optim.Adam(policy.parameters(), **options)
        sampler = np.random.default_rng()
        iteration, counters = restore_checkpoint(checkpoint, policy=policy, optimizer=optimizer,
                                                 environment_rng=sampler, contract=contract)
        write_json(args.output_dir/"config.json", dict(**report, parent_contract=contract, parent_counters=counters))
        optimization = contract["optimization"]
        diagnostic_training_started = time.monotonic()
        seeds = sampler.integers(0, 1_000_000_000, size=50)
        rollout = collect_trials(policy, envs, seeds, replay_chunk_length=100)
        torch.save(rollout, args.output_dir/"diagnostic-rollout.pt")
        report["diagnostic_environment_steps"] = rollout["rewards"].numel()
        report["family_scales"] = family_scales(rollout, [e.family for e in envs], optimization)
        report["gradient_components_before"] = gradient_components(policy, rollout, optimization)
        report["decay_parameters_before"] = parameter_scales(policy)
        report["cached_state_scales"] = {k: tensor_stats(v) for k,v in rollout["replay_states"].items()}
        report["executed_actions"] = tensor_stats(rollout["raw_actions"].tanh())
        report["fraction_abs_action_above_095"] = float((rollout["raw_actions"].tanh().abs() > .95).float().mean())
        report["gaussian_std"] = policy.log_std.detach().clamp(-5, 2).exp().cpu().tolist()
        report["replay_drift_before"] = replay_drift(policy, rollout, 100)
        if max(v["maximum_absolute"] for v in report["replay_drift_before"].values() if isinstance(v, dict)) > 2e-5:
            raise ValueError("Frozen cached/full-prefix replay mismatch")
        write_json(args.output_dir/"progress.json", report)
        print(json.dumps({"phase":"frozen_diagnostics_complete", "gradient":report["gradient_components_before"]}), flush=True)
        before = time.monotonic()
        report["isolated_update"] = update_ppo(policy, optimizer, rollout, **optimization)
        torch.cuda.synchronize()
        report["isolated_update_seconds"] = time.monotonic()-before
        report["replay_drift_after"] = replay_drift(policy, rollout, 100)
        report["gradient_components_after"] = gradient_components(policy, rollout, optimization)
        # Delay the normalizer update exactly until both replay paths were compared.
        policy.core.observe_after_update(rollout["observations"])
        counters["environment_steps"] += rollout["rewards"].numel()
        counters["optimizer_steps"] += report["isolated_update"]["optimizer_steps"]
        counters["training_seconds"] += time.monotonic()-diagnostic_training_started
        counters["cumulative_elapsed_seconds"] += time.monotonic()-started
        diagnostic_contract = dict(contract, protocol="metaworld_isolated_ppo_diagnostic_v1",
            source_commit=source, parent_source=contract["source_commit"], parent_checkpoint_sha256=digest,
            parent_iteration=iteration, comparison_eligible=False)
        report["diagnostic_checkpoint_sha256"] = save_checkpoint(args.output_dir/"diagnostic-after-update.pt",
            policy=policy, optimizer=optimizer, environment_rng=sampler, iteration=iteration+1,
            counters=counters, contract=diagnostic_contract)
        report["completed"] = True
        report["diagnostic_active_seconds"] = time.monotonic()-started
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(args.output_dir/"summary.json", report)
        print(json.dumps({"phase":"diagnostic_complete", "seconds":report["diagnostic_active_seconds"]}), flush=True)
    except BaseException as error:
        write_json(args.output_dir/"failure.json", dict(error=type(error).__name__, message=str(error), parent_preserved=True))
        raise
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
