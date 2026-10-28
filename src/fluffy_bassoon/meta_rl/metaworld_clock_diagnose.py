"""Frozen ML10 public-clock intervention, separate from comparison scores."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from gymnasium.spaces.utils import flatdim
import numpy as np
import torch
from torch import nn

from fluffy_bassoon.evaluation_stages import sha256, write_json
from .metaworld_env import benchmark_pools, task_digest
from .metaworld_policy import make_policy
from .metaworld_train import preserve_random_streams
from .metaworld_workers import make_worker_lanes
from .ppo import collect_trials
from .screen import allocation, clean_source


class CappedEpisodeClock(nn.Module):
    """Evaluation-only adapter; cap the normalized episode index, not step time."""

    def __init__(self, policy, observation_space):
        super().__init__()
        self.policy = policy
        offset = 0
        self.clock_index = None
        for name, space in observation_space.spaces.items():
            if name == "clock":
                if space.shape != (2,):
                    raise ValueError("Require episode and within-episode public clocks")
                self.clock_index = offset
            offset += flatdim(space)
        if self.clock_index is None:
            raise ValueError("Missing public clock")

    @property
    def core(self):
        return self.policy.core

    def effective_observations(self, observations):
        result = observations.clone()
        result[..., self.clock_index].clamp_(max=1.)
        return result

    def step(self, observation, state=None, **kwargs):
        if not kwargs.get("deterministic", False) or kwargs.get("raw_action") is not None:
            raise ValueError("Clock intervention is deterministic frozen evaluation only")
        return self.policy.step(self.effective_observations(observation), state, **kwargs)


def tensor_hash(items):
    digest = hashlib.sha256()
    for key, tensor in sorted(items):
        digest.update(f"{key}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fingerprint(policy):
    # Includes normalizer buffers as well as trainable weights.
    return tensor_hash(policy.state_dict().items())


@torch.no_grad()
def evaluate_condition(policy, envs, *, capped, reset, training_attempts, seed_start):
    adapter = CappedEpisodeClock(policy, envs[0].observation_space) if capped else policy
    with preserve_random_streams():
        rollout = collect_trials(adapter, envs, np.arange(seed_start, seed_start + len(envs)),
            deterministic=True, reset_each_attempt=reset,
            clock_scale=np.array([max(training_attempts - 1, 1), envs[0].horizon]))
    if capped:
        # The collector records its input before the adapter. Hash the effective
        # policy observations explicitly; these records are never PPO replay data.
        rollout["observations"] = adapter.effective_observations(rollout["observations"])
    outcomes = np.asarray([env.outcomes for env in envs], dtype=bool)
    returns = np.asarray([env.native_returns for env in envs], dtype=float)
    families = [env.family for env in envs]
    by_family = {name: outcomes[[x == name for x in families]].mean(0).tolist()
                 for name in sorted(set(families))}
    curve = np.asarray(list(by_family.values())).mean(0)
    prefix = training_attempts * envs[0].horizon
    return dict(success=float(curve.mean()), success_by_episode=curve.tolist(),
        per_family=by_family, outcomes=outcomes.tolist(), native_returns=returns.tolist(),
        family_by_lane=families, environment_steps=int(rollout["action_mask"].sum()),
        trained_prefix_sha256=tensor_hash((k, v[:prefix]) for k, v in rollout.items()),
        complete_trajectory_sha256=tensor_hash(rollout.items()),
        policy_input="episode clock capped at normalized1" if capped else "original extrapolated clock")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--parent-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    parent = json.loads((args.parent_dir/"config.json").read_text())["contract"]
    summary = json.loads((args.parent_dir/"summary.json").read_text())
    checkpoint = args.parent_dir/"latest.pt"
    if (not summary["completed"] or summary["iteration"] != 80
            or summary["counters"]["environment_steps"] != 10_000_000
            or parent["benchmark"] != "ml10" or parent["attempts"] != 5
            or parent["horizon"] != 500 or parent["core"] not in ("persistent_ctm", "cmm")
            or sha256(checkpoint) != summary["checkpoint_sha256"]):
        raise ValueError("Require the unchanged, complete ML10 parent")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools("ml10", parent["benchmark_seed"])
    assert {k: task_digest(v[1]) for k,v in pools.items()} == parent["pool_hashes"]
    report = dict(source_commit=source, parent_source=parent["source_commit"],
        parent_checkpoint_sha256=summary["checkpoint_sha256"], allocation=evidence,
        protocol="metaworld_frozen_clock_diagnostic_v1", comparison_eligible=False,
        training_attempts=5, evaluation_attempts=25, lanes=10, seed_start=1_000_000_000,
        interpretation="One frozen evaluation block; original comparison outcomes remain unchanged.",
        records={}, paired_prefix_checks=0, original_outcome_checks=0)
    write_json(args.output_dir/"config.json", report)
    started, policy, original = time.monotonic(), None, None
    try:
        for split in ("train_eval", "test"):
            envs = make_worker_lanes(pools[split], 10, benchmark="ml10", attempts=25,
                                    horizon=500, workers=8, allocated_cpus=16)
            try:
                if policy is None:
                    policy = make_policy(parent["core"], parent["core_config"], envs[0]).cuda().eval()
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    assert saved["contract"] == parent
                    policy.load_state_dict(saved["policy"], strict=True)
                    del saved
                    original = fingerprint(policy)
                report["records"][split] = {}
                for condition in ("normal", "memory_reset"):
                    pair = {}
                    for capped in (False, True):
                        before = time.monotonic()
                        row = evaluate_condition(policy, envs, capped=capped,
                            reset=condition == "memory_reset", training_attempts=5,
                            seed_start=1_000_000_000)
                        row["active_seconds"] = time.monotonic()-before
                        assert fingerprint(policy) == original, "Frozen policy or normalizer changed"
                        pair["capped" if capped else "original"] = row
                        if not capped:
                            reference = summary["last_evaluation"][split][condition]
                            for key in ("family_by_lane", "outcomes", "native_returns"):
                                assert row[key] == reference[key], f"Original evaluation differs: {split}/{condition}/{key}"
                                report["original_outcome_checks"] += 1
                        report["records"][split][condition] = pair
                        write_json(args.output_dir/"progress.json", report)
                        print(json.dumps(dict(split=split, condition=condition, capped=capped,
                                              success=row["success"], seconds=row["active_seconds"])), flush=True)
                    assert pair["original"]["trained_prefix_sha256"] == pair["capped"]["trained_prefix_sha256"]
                    report["paired_prefix_checks"] += 1
            finally:
                for env in envs:
                    env.close()
        assert sha256(checkpoint) == summary["checkpoint_sha256"]
        report.update(completed=True, frozen_policy_sha256=original,
                      active_seconds=time.monotonic()-started,
                      evaluation_steps=sum(v["environment_steps"] for cs in report["records"].values()
                                           for pair in cs.values() for v in pair.values()),
                      training_steps=0, optimizer_steps=0)
        write_json(args.output_dir/"summary.json", report)
    except BaseException as error:
        write_json(args.output_dir/"failure.json", dict(error=type(error).__name__, message=str(error),
                                                       parent_preserved=True))
        raise


if __name__ == "__main__":
    main()
