"""Real cached-flow fitting leaves an exact recoverable gate snapshot."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from fluffy_bassoon.evaluation_stages import sha256
from fluffy_bassoon.meta_rl.bridge_distillation import make_dpt_record, update_distillation
from fluffy_bassoon.meta_rl.checkpoint import restore_checkpoint, restore_rng
from fluffy_bassoon.meta_rl.pi05_interface_gate import residual_headroom, slice_history_lanes
from fluffy_bassoon.meta_rl.pi05_support_probe_run import fit_candidate
from fluffy_bassoon.meta_rl.warmstart import NATIVE, initialize_from_warmstart
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


class TinyMotor(nn.Module):
    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None].expand(-1, 3, 1, -1)
        return None, torch.ones(key.shape[:2], dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return actions

    def velocity_from_suffix(self, suffix):
        return .2 * suffix


def policy_and_contract():
    config = dict(core="transformer_xl", kv_dim=8, suffix_dim=8, action_dim=8,
                  width=8, ticks=1, memory_length=4, residual_scale=1.)
    policy = StochasticPi05BridgePolicy(Pi05FeatureBridge(**config)).eval()
    contract = dict(source_commit="a" * 40, task="synthetic_native_flow_fixture", objects=[0, 1],
        train_theta_degrees=[-45., 0., 45.], train_overlays=list(range(12)), attempts=1,
        episode_steps=2, action_chunk=1, denoise_steps=4, feedback="public fixture feedback",
        reward="fixture terminal reward", core="transformer_xl", bridge=config,
        gaussian_policy=policy.config(), motor_sha256="b" * 64, norm_stats_sha256="c" * 64,
        rlinf_sources={"synthetic": "d" * 64})
    return policy, contract


def cached_history():
    torch.manual_seed(74)
    motor, generator = TinyMotor().eval(), torch.Generator().manual_seed(21)
    records = [make_dpt_record(motor, SimpleNamespace(state=torch.randn(2, 4)),
        torch.randn(2, 11), torch.randn(2, 2, 8), generator=generator) for _ in range(2)]
    ends = torch.tensor([[False, False], [True, True]])
    return dict(records=records, rewards=torch.zeros(2, 2), resets=torch.zeros_like(ends),
                attempt_ends=ends.clone(), trial_ends=ends, supervised_action_steps=1)


def test_sliced_teacher_history_is_compatible_with_headroom_and_real_fit():
    selected = slice_history_lanes(cached_history(), [1])
    diagnostics = residual_headroom(selected, residual_scale=1.)
    assert diagnostics["scalar_count"] == 2 * 1 * 1 * 7
    assert diagnostics["policy_chunks"] == selected["policy_chunks"] == 2
    policy, _ = policy_and_contract()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    metrics = update_distillation(policy, optimizer, selected)
    assert np.isfinite(metrics["loss"]) and metrics["gradient_norm"] > 0


def test_fit_candidate_snapshot_restores_optimizer_sampler_and_exact_next_update(tmp_path):
    data = cached_history()
    policy, contract = policy_and_contract()
    output = tmp_path / "fit"
    result = fit_candidate(policy, [data, deepcopy(data)], [data], output=output,
        contract=contract, provenance=dict(simulator_steps=8, fixture="immutable test histories"),
        updates=2, fit_seconds=30., learning_rate=1e-3, seed=23, should_stop=lambda: False)
    assert result["completed"] and result["iteration"] == 2
    assert result["counters"] == dict(teacher_simulator_steps=8, optimizer_steps=2,
        replayed_policy_chunks=8, validation_policy_chunks=8)
    assert sha256(output / "latest.pt") == result["checkpoint_sha256"]
    assert sha256(output / "warmstart.pt") == result["warmstart_sha256"]
    saved = torch.load(output / "latest.pt", weights_only=False)
    assert saved["boundary"] == "between_trials" and saved["optimizer"]["state"]
    saved_contract = json.loads((output / "config.json").read_text())["contract"]
    assert saved["contract"] == saved_contract
    # An optimizer continued from the emitted snapshot must match a second
    # independent restoration, including the cached-history sampler stream.
    expected, _ = policy_and_contract()
    expected.load_state_dict(saved["policy"])
    expected_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-3, weight_decay=0.)
    expected_optimizer.load_state_dict(saved["optimizer"])
    expected_sampler = np.random.default_rng()
    expected_sampler.bit_generator.state = saved["environment_rng"]
    restore_rng(saved["rng"])
    expected_choice = int(expected_sampler.integers(2))
    expected_noise = torch.randn(3)
    update_distillation(expected, expected_optimizer, data)
    restored, _ = policy_and_contract()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, weight_decay=0.)
    restored_sampler = np.random.default_rng(999)
    iteration, counters = restore_checkpoint(output / "latest.pt", policy=restored,
        optimizer=restored_optimizer, environment_rng=restored_sampler, contract=saved_contract)
    assert iteration == 2 and counters == result["counters"]
    assert int(restored_sampler.integers(2)) == expected_choice
    assert torch.equal(torch.randn(3), expected_noise)
    update_distillation(restored, restored_optimizer, data)
    torch.testing.assert_close(restored.state_dict(), expected.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored_optimizer.state_dict(), expected_optimizer.state_dict(), rtol=0, atol=0)
    # The separately exported initialization is usable without transferring
    # optimizer state or recurrent trial memory.
    initialized, _ = policy_and_contract()
    provenance = initialize_from_warmstart(output / "warmstart.pt", result["warmstart_sha256"],
        family=NATIVE, policy=initialized, contract=contract)
    assert provenance["optimizer"] == "fresh"
    torch.testing.assert_close(initialized.state_dict(), policy.state_dict(), rtol=0, atol=0)
