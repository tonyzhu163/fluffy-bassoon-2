"""Paired CTM replay timing on one real Meta-World rollout, inside an allocation."""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.evaluation_stages import write_json
from .metaworld_env import benchmark_pools, make_lanes
from .metaworld_policy import make_policy
from .metaworld_train import core_settings, replay_error
from .ppo import RecurrentActorCritic, collect_trials, update_ppo
from .screen import allocation, clean_source


class SerialHeadReference(RecurrentActorCritic):
    """The pilot's previous per-decision heads and distributions, for timing."""

    def evaluate_sequence(self, observations, raw_actions, resets=None, *, bounded_entropy=False,
                          initial_state=None):
        state, rows = initial_state, []
        for step in range(len(observations)):
            _, _, log_prob, entropy, value, state = self.step(observations[step], state,
                raw_action=raw_actions[step], reset=None if resets is None else resets[step],
                bounded_entropy=bounded_entropy)
            rows.append((log_prob, entropy, value))
        return tuple(torch.stack([row[i] for row in rows]) for i in range(3))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ctm-ticks", type=int, default=1)
    args = p.parse_args(argv)
    evidence = allocation()
    source = clean_source(args.source_commit)
    if args.ctm_ticks < 1:
        raise ValueError("Require positive CTM ticks")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pools = benchmark_pools("ml10", 0)
    envs = make_lanes(pools["train"], 10, benchmark="ml10", attempts=5, horizon=500)
    config = dict(core_settings("persistent_ctm"), width=84, ticks=args.ctm_ticks)
    policy = make_policy("persistent_ctm", config, envs[0]).cuda().eval()
    optimization = dict(epochs=1, minibatch_trials=2, clip=.2, gamma=.99, gae_lambda=.95,
                        entropy_coef=0., value_coef=.5, max_grad_norm=.5)
    result = dict(source_commit=source, allocation=evidence, core_config=config,
                  seed=1000, benchmark="ml10", training_transitions=25000,
                  optimizer_steps_per_update=5, optimization=optimization,
                  methodology="one paired rollout per collector mode; one optimizer update per replay mode; no convergence claim",
                  rollouts={}, updates={})
    write_json(args.output_dir / "config.json", result)
    try:
        # Warm CUDA and the policy before measuring collection.
        with torch.no_grad():
            state = None
            for _ in range(8):
                _, state = policy.core(torch.zeros(10, policy.core.input_dim, device="cuda"), state)
        torch.cuda.synchronize()
        baseline = None
        for name, length in (("original_collection", 0), ("cached_collection", 100)):
            torch.manual_seed(2000)
            torch.cuda.synchronize()
            started = time.monotonic()
            rollout = collect_trials(policy, envs, np.arange(3000, 3010), replay_chunk_length=length)
            torch.cuda.synchronize()
            result["rollouts"][name] = dict(seconds=time.monotonic() - started,
                                            environment_steps=rollout["rewards"].numel())
            if baseline is None:
                baseline = {key: value for key, value in rollout.items()}
            else:
                for key in baseline:
                    torch.testing.assert_close(rollout[key], baseline[key], rtol=0, atol=0)
            print(json.dumps({"phase": name, **result["rollouts"][name]}), flush=True)
            write_json(args.output_dir / "progress.json", result)
        del baseline
        frozen = deepcopy(policy.state_dict())
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        for name, length in (("original_serial_full_trial", 0),
                             ("batched_heads_full_trial", 0), ("cached_chunks_100", 100)):
            model = make_policy("persistent_ctm", config, envs[0]).cuda().eval()
            model.load_state_dict(frozen)
            if name == "original_serial_full_trial":
                model.__class__ = SerialHeadReference
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)
            torch.manual_seed(4000)
            # Validate frozen behavior before every timed optimizer update.
            error = replay_error(model, rollout, length)
            if max(error.values()) > 2e-5:
                raise RuntimeError(f"Behavior replay error in {name}: {error}")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            metrics = update_ppo(model, optimizer, rollout, replay_chunk_length=length, **optimization)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            assert metrics["optimizer_steps"] == 5 and metrics["forward_samples"] == 25000
            result["updates"][name] = dict(seconds=seconds, replay_error=error, metrics=metrics,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gradient_horizon_observations=length or 2500)
            print(json.dumps({"phase": name, **result["updates"][name]}), flush=True)
            write_json(args.output_dir / "progress.json", result)
            del optimizer, model
            gc.collect()
            torch.cuda.empty_cache()
        original = result["updates"]["original_serial_full_trial"]["seconds"]
        for row in result["updates"].values():
            row["speedup_vs_serial"] = original / row["seconds"]
        write_json(args.output_dir / "summary.json", result)
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
