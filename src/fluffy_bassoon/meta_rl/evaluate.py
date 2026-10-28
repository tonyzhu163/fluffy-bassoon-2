"""Read-only CPU evaluation of compact checkpoints with greedy and seeded exploration."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from gymnasium.spaces.utils import flatdim

from fluffy_bassoon.evaluation_stages import sha256, write_json
from fluffy_bassoon.gpu_execution import validate_execution
from .cmm import CMMCore
from .transformer_xl import TransformerXLCore
from .ppo import GRUCore, LoopedTransformerCore, PersistentCTMVectorCore, RecurrentActorCritic
from .train import evaluate_trials, make_env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[2_000_000_000, 2_000_000_001, 2_000_000_002])
    args = parser.parse_args(argv)
    if sys.platform != "linux":
        raise ValueError("Run checkpoint evaluation on allocated Linux resources")
    validate_execution("slurm", os.environ)
    if (len(set(args.policy_seeds)) != len(args.policy_seeds)
            or any(not 0 <= seed < 2**32 for seed in args.policy_seeds)):
        raise ValueError("Require distinct valid policy seeds")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if source != args.source_commit or subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("Require exact clean evaluator source")
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    contract = config["contract"]
    digest = sha256(args.checkpoint)
    # These are the project's own complete learner snapshots, including non-tensor RNG state.
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (saved.get("version") != 1 or saved.get("boundary") != "between_trials"
            or saved["contract"] != contract or sha256(args.checkpoint) != digest):
        raise ValueError("Checkpoint/config mismatch or checkpoint changed while reading")
    if contract["device"] != "cpu" or contract["dtype"] != "float32":
        raise ValueError("This diagnostic supports saved CPU float32 compact policies")
    encoding = contract.get("observation_encoding")
    if encoding not in (None, "clock_by_declared_max_v1"):
        raise ValueError("Unknown saved observation encoding")
    normalize_clock = encoding is not None  # Format-one predecessors used raw clocks.
    task, attempts, horizon = (contract[key] for key in ("task", "attempts", "horizon"))
    arms = contract.get("arms") or 2
    env = make_env(task, attempts, horizon, arms)
    factory = {"gru": GRUCore, "ctm": CMMCore, "cmm": CMMCore, "persistent_ctm": PersistentCTMVectorCore,
               "looped_transformer": LoopedTransformerCore, "transformer_xl": TransformerXLCore}[contract["core"]]
    policy = RecurrentActorCritic(factory(int(flatdim(env.observation_space)), **contract["core_config"]),
                                   env.action_space)
    env.close()
    policy.load_state_dict(saved["policy"], strict=True)
    before = deepcopy(policy.state_dict())
    trials = contract["evaluation"]["trials"] if args.trials is None else args.trials
    if trials < 1:
        raise ValueError("Require positive evaluation trials")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_start = contract["evaluation"]["seed_start"]
    seeds = np.arange(seed_start, seed_start + trials)
    result = dict(evaluator_source=source, training_source=contract["source_commit"],
                  checkpoint=dict(path=str(args.checkpoint), sha256=digest, iteration=saved["iteration"]),
                  observation_encoding="raw_legacy" if encoding is None else encoding,
                  trials=trials, seed_start=seed_start, policy_seeds=args.policy_seeds,
                  job_id=os.environ["SLURM_JOB_ID"], hostname=os.uname().nodename,
                  runtime=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
                  conditions={}, evaluation_steps=0, completed_run_greedy_reference_matched=False)
    normal_reset = contract["mode"] == "episode"
    def condition(name, deterministic, policy_seed, reset):
        summary, raw = evaluate_trials(policy, task, attempts, horizon, seeds, arms=arms,
                         reset_each_attempt=reset, deterministic=deterministic,
                         policy_seed=policy_seed, normalize_clock=normalize_clock)
        np.savez_compressed(args.output_dir / f"{name}.npz", **raw)
        result["conditions"][name] = summary
        result["evaluation_steps"] += summary["environment_steps"]
        print(name, summary.get("mean_reward", summary.get("mean_success")), flush=True)
        return summary
    greedy = condition("greedy", True, args.policy_seeds[0], normal_reset)
    summary_path = args.checkpoint.parent / "summary.json"
    if summary_path.exists() and trials == contract["evaluation"]["trials"]:
        previous = json.loads(summary_path.read_text())
        if previous.get("completed") and previous["checkpoint"]["sha256"] == digest:
            expected = previous["last_evaluation"]["normal"]
            for key in ("mean_reward", "mean_success", "trial_reward", "cumulative_expected_regret",
                        "reward_by_attempt", "success_by_attempt"):
                if key in expected:
                    np.testing.assert_allclose(greedy[key], expected[key], rtol=0, atol=1e-7)
            result["completed_run_greedy_reference_matched"] = True
    if not normal_reset:
        condition("greedy_reset", True, args.policy_seeds[0], True)
    for policy_seed in args.policy_seeds:
        condition(f"sampled-{policy_seed}", False, policy_seed, normal_reset)
        if not normal_reset:
            condition(f"sampled-{policy_seed}-reset", False, policy_seed, True)
    if any(not torch.equal(value, before[key]) for key, value in policy.state_dict().items()):
        raise RuntimeError("Evaluation modified a model tensor")
    result["model_tensors_unchanged"] = True
    result["completed"] = True
    write_json(args.output_dir / "summary.json", result)
    return result


if __name__ == "__main__":
    main()
