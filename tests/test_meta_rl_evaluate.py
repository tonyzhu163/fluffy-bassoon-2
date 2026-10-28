"""Exploration evaluation must be reproducible and must not consume training RNG."""

from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from gymnasium.spaces.utils import flatdim

from fluffy_bassoon.evaluation_stages import sha256
from fluffy_bassoon.meta_rl.environments import BernoulliBandit
from fluffy_bassoon.meta_rl.ppo import GRUCore, RecurrentActorCritic
from fluffy_bassoon.meta_rl.train import evaluate_trials


def test_sampled_evaluation_has_its_own_rng_and_preserves_every_model_tensor():
    torch.manual_seed(13)
    env = BernoulliBandit(attempts=8, arms=3)
    policy = RecurrentActorCritic(GRUCore(int(flatdim(env.observation_space)), width=8), env.action_space)
    before_model = deepcopy(policy.state_dict())
    before_rng = torch.get_rng_state().clone()
    first, raw_first = evaluate_trials(policy, "bernoulli", 8, 1, [1, 2, 3, 4], arms=3,
                                       deterministic=False, policy_seed=17)
    torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
    torch.rand(11)
    other_rng = torch.get_rng_state().clone()
    second, raw_second = evaluate_trials(policy, "bernoulli", 8, 1, [1, 2, 3, 4], arms=3,
                                         deterministic=False, policy_seed=17)
    assert first == second
    for key in raw_first:
        np.testing.assert_array_equal(raw_first[key], raw_second[key])
    torch.testing.assert_close(torch.get_rng_state(), other_rng, rtol=0, atol=0)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before_model[key], rtol=0, atol=0)
    _, third = evaluate_trials(policy, "bernoulli", 8, 1, [1, 2, 3, 4], arms=3,
                               deterministic=False, policy_seed=18)
    assert not np.array_equal(raw_first["raw_actions"], third["raw_actions"])


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Evaluator CLI requires an allocated Linux step")
def test_read_only_evaluator_preserves_checkpoint_and_matches_training_greedy_reference(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    training, output = tmp_path / "training", tmp_path / "evaluation"
    run = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.train",
                           "--source-commit", source, "--output-dir", str(training),
                           "--updates", "1", "--num-envs", "2", "--width", "8", "--eval-trials", "4",
                           "--ppo-epochs", "1", "--minibatch-trials", "2"],
                          capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    checkpoint = training / "latest.pt"
    digest = sha256(checkpoint)
    run = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.evaluate",
                           "--source-commit", source, "--checkpoint", str(checkpoint),
                           "--output-dir", str(output), "--policy-seeds", "17"],
                          capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert sha256(checkpoint) == digest
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed_run_greedy_reference_matched"] and summary["model_tensors_unchanged"]
    assert summary["evaluation_steps"] == 64
    with np.load(output / "sampled-17.npz") as raw:
        assert raw["rewards"].shape == (4, 4) and raw["raw_actions"].shape == (4, 4)
