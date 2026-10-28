"""Explicit CUDA qualification after the repository's allocation-local GPU probe."""

from copy import deepcopy
import os

import numpy as np
import pytest
import torch
from gymnasium.spaces.utils import flatdim

from fluffy_bassoon.gpu_execution import validate_execution
from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, save_checkpoint
from fluffy_bassoon.meta_rl.cmm import CMMCore
from fluffy_bassoon.meta_rl.transformer_xl import TransformerXLCore
from fluffy_bassoon.meta_rl.environments import BernoulliBandit, HiddenGainReach
from fluffy_bassoon.meta_rl.ppo import (
    GRUCore, LoopedTransformerCore, RecurrentActorCritic, collect_trials, update_ppo,
)

pytestmark = pytest.mark.skipif(os.environ.get("FLUFFY_META_RL_CUDA_QUALIFICATION") != "1",
                                reason="Requires an explicitly probed one-GPU scheduler allocation")


def policy_for(env, kind):
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'slurm'), os.environ)
    assert torch.cuda.device_count() == 1
    inputs = int(flatdim(env.observation_space))
    if kind == "gru":
        core = GRUCore(inputs, width=8, ticks=2)
    elif kind == "transformer_xl":
        core = TransformerXLCore(inputs, width=8, memory_length=4, heads=2)
    elif kind == "looped_transformer":
        core = LoopedTransformerCore(inputs, width=8, ticks=2, memory_tokens=2)
    else:
        core = CMMCore(inputs, width=8, ticks=2, stm_length=3, heads=2, synapse_depth=2,
                       layers=0 if kind == "ctm" else 1, ltm_tokens=0 if kind == "ctm" else 2)
    return RecurrentActorCritic(core, env.action_space)


@pytest.mark.parametrize("kind", ["gru", "ctm", "cmm", "looped_transformer", "transformer_xl"])
@pytest.mark.parametrize("factory", [lambda: BernoulliBandit(attempts=8, arms=3),
                                     lambda: HiddenGainReach(attempts=2, horizon=2)])
def test_cuda_replays_likelihoods_matches_cpu_inference_and_updates(kind, factory):
    torch.manual_seed(51)
    envs = [factory() for _ in range(3)]
    cpu = policy_for(envs[0], kind).eval()
    cuda = deepcopy(cpu).to("cuda")
    expected = collect_trials(cpu, envs, [1, 2, 3], deterministic=True)
    rollout = collect_trials(cuda, envs, [1, 2, 3], deterministic=True)
    for key in expected:
        torch.testing.assert_close(rollout[key].cpu(), expected[key], rtol=3e-5, atol=3e-5)
    log_probs, _, values = cuda.evaluate_sequence(rollout["observations"], rollout["raw_actions"], rollout["resets"])
    torch.testing.assert_close(log_probs, rollout["log_probs"], rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(values, rollout["values"], rtol=3e-5, atol=3e-5)
    before = {key: value.clone() for key, value in cuda.state_dict().items()}
    metrics = update_ppo(cuda, torch.optim.AdamW(cuda.parameters(), lr=1e-3), rollout,
                         epochs=1, minibatch_trials=3)
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(not torch.equal(value, before[key]) for key, value in cuda.state_dict().items())


@pytest.mark.parametrize("kind", ["gru", "ctm", "cmm", "looped_transformer", "transformer_xl"])
def test_cuda_checkpoint_restores_sampling_optimizer_and_next_update(tmp_path, kind):
    torch.manual_seed(37)
    envs = [BernoulliBandit(attempts=8, arms=3) for _ in range(3)]
    policy = policy_for(envs[0], kind).to("cuda")
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(19)
    contract = dict(qualification=True, device="cuda", kind=kind)
    def update(model, optim):
        seeds = rng.integers(0, 1_000_000_000, size=3)
        batch = collect_trials(model, envs, seeds)
        update_ppo(model, optim, batch, epochs=1, minibatch_trials=3)
        return batch
    update(policy, optimizer)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, policy=policy, optimizer=optimizer, environment_rng=rng,
                    iteration=1, counters={}, contract=contract)
    expected = update(policy, optimizer)
    expected_state = deepcopy(policy.state_dict())
    restored = policy_for(envs[0], kind).to("cuda")
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restore_checkpoint(path, policy=restored, optimizer=restored_optimizer,
                       environment_rng=rng, contract=contract)
    actual = update(restored, restored_optimizer)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_state[key], rtol=3e-6, atol=3e-6)
