"""Behavior/label separation, recurrent gradients and portable initialization gates."""

from copy import deepcopy
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.evaluation_stages import sha256
from fluffy_bassoon.meta_rl.bridge_distillation import corrected_action_label, make_dpt_record, distillation_loss, update_distillation
from fluffy_bassoon.meta_rl.cores import build_policy
from fluffy_bassoon.meta_rl.environments import HiddenRewardBandit
from fluffy_bassoon.meta_rl.teachers import expert_action
from fluffy_bassoon.meta_rl.train import make_env, DEFAULT_HORIZONS
from fluffy_bassoon.meta_rl.warmstart import COMPACT, export_warmstart, initialize_from_warmstart
from fluffy_bassoon.meta_rl.warmstart_train import generate_histories, supervised_loss, update_supervised
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


@pytest.mark.parametrize("task", ["bandit", "bernoulli", "nonstationary", "reach", "darkroom", "keydoor"])
def test_labels_are_separate_from_real_history_and_reproducible(task):
    factory = lambda: make_env(task, 3, DEFAULT_HORIZONS.get(task, 1), 3)
    seeds = np.arange(6)
    data = generate_histories(factory, task, seeds, arms=3)
    repeated = generate_histories(factory, task, seeds, arms=3)
    for key in data:
        np.testing.assert_array_equal(data[key], repeated[key])
    assert np.any(data["raw_actions"] != data["target_actions"])
    # Replaying executed actions must reproduce every input and reward, despite different expert labels.
    from gymnasium.spaces.utils import flatten
    for lane, seed in enumerate(seeds):
        env = factory()
        obs, _ = env.reset(seed=int(seed))
        for step in range(data["observations"].shape[0]):
            encoded = dict(obs, clock=obs["clock"] / np.maximum(env.observation_space["clock"].high, 1))
            # atanh/tanh serialization of continuous actions incurs float32 rounding.
            np.testing.assert_allclose(flatten(env.observation_space, encoded), data["observations"][step, lane],
                                       atol=2e-6, rtol=1e-5)
            action = data["raw_actions"][step, lane]
            if task == "reach":
                action = np.tanh(action).astype(np.float32)
            obs, reward, _, _, _ = env.step(action)
            assert reward == pytest.approx(float(data["rewards"][step, lane]), abs=2e-6)


def test_grid_oracle_reaches_hidden_target_and_reach_oracle_uses_gain():
    for task in ("darkroom", "keydoor", "reach"):
        env = make_env(task, 2, DEFAULT_HORIZONS[task])
        obs, _ = env.reset(seed=19)
        rng = np.random.default_rng(8)
        successes = []
        for _ in range(env.attempts * env.horizon):
            obs, _, _, _, info = env.step(expert_action(env, task, obs, rng=rng))
            if info["attempt_end"]:
                successes.append(info["is_success"])
        assert all(successes)


@pytest.mark.parametrize("core", ["persistent_ctm", "gru"])
def test_supervised_update_uses_labels_and_trains_recurrence(core):
    torch.manual_seed(3)
    factory = lambda: HiddenRewardBandit(attempts=4)
    data = generate_histories(factory, "bandit", np.arange(8))
    config = dict(width=8, ticks=1)
    if core == "persistent_ctm":
        config.update(history_length=3, synapse_depth=2, bounded_synchronisation=True, decay_per_observation=True)
    policy = build_policy(core, config, factory()).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.003)
    before = deepcopy(policy.core.state_dict())
    loss = supervised_loss(policy, data, np.arange(8))
    changed = dict(data, raw_actions=1 - data["raw_actions"])
    torch.testing.assert_close(loss, supervised_loss(policy, changed, np.arange(8)), rtol=0, atol=0)
    update_supervised(policy, optimizer, data, np.arange(8))
    assert any(not torch.equal(value, before[key]) for key, value in policy.core.state_dict().items())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())


@pytest.mark.parametrize("bad", [None, "family", "task", "sha", "nan"])
def test_initialization_is_weights_only_atomic_compatible_and_fresh(tmp_path, bad):
    env = HiddenRewardBandit()
    contract = dict(task="bandit", attempts=4, horizon=1, arms=2, observation_encoding="clock_by_declared_max_v1",
                    core="gru", core_config=dict(width=8, ticks=1), dtype="float32", source_commit="a" * 40)
    parent = build_policy("gru", contract["core_config"], env)
    path = tmp_path / "warmstart.pt"
    digest = export_warmstart(path, family=COMPACT, policy=parent, contract=contract, iteration=1,
        counters=dict(optimizer_steps=1, teacher_environment_steps=32), data={"sha256": "b" * 64},
        evaluation={"validation_nll": .2})
    target_contract = dict(contract, source_commit="c" * 40)
    family = COMPACT
    if bad == "family":
        family = "unknown"
    elif bad == "task":
        target_contract["task"] = "bernoulli"
    elif bad == "sha":
        digest = "0" * 64
    elif bad == "nan":
        payload = torch.load(path, weights_only=True)
        payload["policy"]["actor.weight"].fill_(float("nan"))
        torch.save(payload, path)
        digest = sha256(path)
    policy = build_policy("gru", contract["core_config"], env)
    before, rng = deepcopy(policy.state_dict()), torch.get_rng_state().clone()
    if bad:
        with pytest.raises(ValueError):
            initialize_from_warmstart(path, digest, family=family, policy=policy, contract=target_contract)
        for key in before:
            torch.testing.assert_close(policy.state_dict()[key], before[key], rtol=0, atol=0)
    else:
        result = initialize_from_warmstart(path, digest, family=family, policy=policy, contract=target_contract)
        for key, value in parent.state_dict().items():
            if key.startswith("critic."):
                assert not policy.state_dict()[key].any()
            else:
                torch.testing.assert_close(policy.state_dict()[key], value, rtol=0, atol=0)
        assert result["optimizer"] == "fresh"
        assert torch.equal(rng, torch.get_rng_state())


class TinyMotor:
    def decode_actions(self, actions, state):
        # Nontrivial scale + offset catches incorrect rotation in normalized model coordinates.
        return actions[..., :7] * actions.new_tensor([.2, .7, 1., 1., 1., 1., 1.]) + .1

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None, :4].expand(-1, 3, 1, -1)
        return None, torch.ones(key.shape[:2], dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, actions, time, cache, mask):
        return actions

    def velocity_from_suffix(self, suffix):
        return .2 * suffix


def test_native_labels_invert_actual_decoder_without_affecting_other_commands():
    from fluffy_bassoon.rl.maniskill_pi05 import rotate_translation
    motor = TinyMotor()
    actions = torch.randn(2, 2, 8)
    state = torch.zeros(2, 4)
    target = corrected_action_label(motor, state, actions, [-45., 45.])
    decoded = motor.decode_actions(actions, state)
    corrected = motor.decode_actions(target, state)
    torch.testing.assert_close(rotate_translation(corrected, [-45., 45.]), decoded, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(corrected[..., 2:], decoded[..., 2:], rtol=0, atol=0)


@pytest.mark.parametrize("core", ["persistent_ctm", "transformer_xl"])
def test_native_flow_loss_replays_context_and_does_not_feed_labels_to_recurrence(core):
    torch.manual_seed(7)
    bridge = Pi05FeatureBridge(core=core, kv_dim=8, suffix_dim=8, action_dim=8, width=8, ticks=1, memory_length=4)
    policy = StochasticPi05BridgePolicy(bridge).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=.003)
    motor, generator = TinyMotor(), torch.Generator().manual_seed(8)
    records = [make_dpt_record(motor, SimpleNamespace(state=torch.randn(2, 4)), torch.randn(2, 11),
               torch.randn(2, 2, 8), generator=generator) for _ in range(3)]
    data = dict(records=records)
    loss = distillation_loss(policy, data)
    # target_actions is provenance; only target_velocity supervises, and neither is a recurrent input.
    altered = deepcopy(data)
    for record in altered["records"]:
        record["target_actions"].fill_(1e6)
    torch.testing.assert_close(loss, distillation_loss(policy, altered), rtol=0, atol=0)
    update_distillation(policy, optimizer, data)
    before = deepcopy(bridge.core.state_dict())
    update_distillation(policy, optimizer, data)
    assert any(not torch.equal(value, before[key]) for key, value in bridge.core.state_dict().items())
    assert all(parameter.grad is None for parameter in policy.critic.parameters())


@pytest.mark.skipif(sys.platform != "linux" or not (
    os.environ.get("SLURM_STEP_ID") or os.environ.get("KUBERNETES_SERVICE_HOST")),
    reason="CLI warm-start qualification requires allocated Linux")
def test_real_warmstart_cli_resumes_and_initializes_ppo(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    common = [sys.executable, "-m", "fluffy_bassoon.meta_rl.warmstart_train",
        "--source-commit", source, "--expected-hostname", os.uname().nodename,
        "--task", "bandit", "--core", "persistent_ctm", "--width", "8", "--ticks", "1",
        "--attempts", "4", "--train-trials", "8", "--validation-trials", "4",
        "--batch-trials", "2", "--eval-trials", "4", "--eval-every", "2", "--threads", "1"]
    first, resumed, direct, ppo = (tmp_path / name for name in ("first", "resumed", "direct", "ppo"))
    for output, updates, extra in ((first, 2, []), (resumed, 3,
            ["--resume", str(first / "latest.pt"), "--dataset-dir", str(first)]), (direct, 3, [])):
        result = subprocess.run(common + ["--output-dir", str(output), "--updates", str(updates)] + extra,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(resumed / "latest.pt", weights_only=False)
    b = torch.load(direct / "latest.pt", weights_only=False)
    torch.testing.assert_close(a["policy"], b["policy"], atol=0, rtol=0)
    torch.testing.assert_close(a["optimizer"], b["optimizer"], atol=0, rtol=0)
    result = subprocess.run([sys.executable, "-m", "fluffy_bassoon.meta_rl.train",
        "--source-commit", source, "--output-dir", str(ppo), "--task", "bandit",
        "--core", "persistent_ctm", "--width", "8", "--ticks", "1", "--attempts", "4",
        "--bounded-synchronisation", "--decay-per-observation", "--updates", "1",
        "--num-envs", "2", "--minibatch-trials", "2", "--ppo-epochs", "1",
        "--eval-trials", "4", "--eval-every", "1", "--checkpoint-every", "1",
        "--warmstart", str(resumed / "warmstart.pt"), "--warmstart-sha256", sha256(resumed / "warmstart.pt")],
        capture_output=True, text=True, timeout=120,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="0") if os.environ.get("KUBERNETES_SERVICE_HOST") else None)
    assert result.returncode == 0, result.stdout + result.stderr
    contract = json.loads((ppo / "config.json").read_text())["contract"]
    assert contract["initialization"]["parent"]["iteration"] == 3
    assert json.loads((ppo / "summary.json").read_text())["iteration"] == 1
