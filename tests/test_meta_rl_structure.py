"""Structural CMM diagnostics: component lesions, longer trials, state export, probes and AD data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.cores import build_policy, core_config_from_args, policy_size
from fluffy_bassoon.meta_rl.evaluate import LESIONS, lesion_keys
from fluffy_bassoon.meta_rl.ppo import collect_trials
from fluffy_bassoon.meta_rl.probe import probe_archive, ridge_probe
from fluffy_bassoon.meta_rl.screen import CMM_STRUCTURE_VARIANTS, VARIANT_OVERRIDES, matching_width
from fluffy_bassoon.meta_rl.train import evaluate_trials, make_env
from fluffy_bassoon.meta_rl.warmstart_train import generate_histories, warmstart_core_config


CMM_CONFIG = dict(width=8, ticks=1, stm_length=3, ltm_tokens=2, synapse_depth=2)


def random_policy(core="cmm", task="darkroom", attempts=3, horizon=4, config=CMM_CONFIG):
    env = make_env(task, attempts, horizon)
    policy = build_policy(core, dict(config), env).eval()
    env.close()
    torch.manual_seed(5)
    with torch.no_grad():
        # Move off the zero/identity initialization so every state component matters.
        for parameter in policy.parameters():
            parameter.normal_(0, .5)
    return policy


def rollout(policy, attempts=3, horizon=4, seeds=(1, 2, 3, 4), **options):
    envs = [make_env("darkroom", attempts, horizon) for _ in seeds]
    try:
        return collect_trials(policy, envs, list(seeds), deterministic=True, **options)
    finally:
        for env in envs:
            env.close()


def test_resetting_every_component_equals_a_full_attempt_reset():
    policy = random_policy()
    keys = tuple(policy.core.initial_state(1))
    full = rollout(policy, reset_each_attempt=True, record_states=True)
    named = rollout(policy, reset_each_attempt=True, reset_keys=keys, record_states=True)
    for key in ("raw_actions", "rewards", "observations", "resets"):
        torch.testing.assert_close(named[key], full[key], rtol=0, atol=0)
    for key, value in full["attempt_states"].items():
        torch.testing.assert_close(named["attempt_states"][key], value, rtol=0, atol=0)


def test_one_component_lesion_resets_only_that_entry_at_attempt_boundaries():
    policy = random_policy()
    normal = rollout(policy, record_states=True)
    lesioned = rollout(policy, reset_each_attempt=True, reset_keys=("long_term",), record_states=True)
    # The first attempt precedes any boundary, so it is unchanged.
    for key, value in normal["attempt_states"].items():
        torch.testing.assert_close(lesioned["attempt_states"][key][0], value[0], rtol=0, atol=0)
    assert not torch.equal(lesioned["attempt_states"]["long_term"][1:], normal["attempt_states"]["long_term"][1:])
    assert not torch.equal(lesioned["attempt_states"]["alpha"][1:], normal["attempt_states"]["alpha"][1:])
    full = rollout(policy, reset_each_attempt=True, record_states=True)
    assert not torch.equal(lesioned["attempt_states"]["alpha"][1:], full["attempt_states"]["alpha"][1:])


def test_component_lesions_reject_tensor_states_unknown_entries_and_missing_attempt_resets():
    policy = random_policy()
    with pytest.raises(ValueError, match="Component resets"):
        rollout(policy, reset_keys=("long_term",))
    with pytest.raises(ValueError, match="Component resets"):
        rollout(policy, reset_each_attempt=True, reset_keys=("memory",))
    gru = random_policy("gru", config=dict(width=8, ticks=1))
    with pytest.raises(ValueError, match="Component resets"):
        rollout(gru, reset_each_attempt=True, reset_keys=("state",))
    with pytest.raises(ValueError, match="no state"):
        lesion_keys(gru, "trace")
    assert lesion_keys(policy, "synchronization") == ("alpha", "beta")
    assert set(LESIONS) == {"long_term", "trace", "activated", "synchronization"}


def test_longer_trial_with_training_clock_scale_reproduces_the_trained_attempts():
    policy = random_policy()
    trained = rollout(policy, attempts=3)
    scale = np.maximum(make_env("darkroom", 3, 4).observation_space["clock"].high, 1)
    longer = rollout(policy, attempts=6, clock_scale=scale)
    steps = trained["raw_actions"].shape[0]
    assert longer["raw_actions"].shape[0] == 2 * steps
    for key in ("observations", "raw_actions", "rewards"):
        torch.testing.assert_close(longer[key][:steps], trained[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="clock scale"):
        rollout(policy, attempts=6, clock_scale=scale, normalize_clock=False)


@pytest.mark.parametrize("task,label", [("darkroom", "goal_cell"), ("keydoor", "door_cell"),
                                        ("bernoulli", "best_arm"), ("nonstationary", "best_arm_by_attempt")])
def test_exported_states_are_trial_major_with_private_probe_targets(task, label):
    horizon = {"darkroom": 4, "keydoor": 4}.get(task, 1)
    policy = random_policy(task=task, horizon=horizon)
    summary, raw = evaluate_trials(policy, task, 3, horizon, np.arange(5), arms=2, record_states=True)
    assert raw["state_long_term"].shape == (5, 3, 2, 8)
    assert raw["state_alpha"].shape == (5, 3, 8)
    assert raw[f"hidden_{label}"].shape == ((5, 3) if task == "nonstationary" else (5,))
    assert summary["reset_keys"] is None
    _, plain = evaluate_trials(policy, task, 3, horizon, np.arange(5), arms=2)
    np.testing.assert_array_equal(plain["raw_actions"], raw["raw_actions"])
    assert not any(key.startswith(("state_", "hidden_")) for key in plain)


def test_ridge_probe_reads_encoded_labels_and_not_noise():
    rng = np.random.default_rng(0)
    labels = rng.integers(4, size=400)
    encoded = np.eye(4)[labels] + .05 * rng.normal(size=(400, 4))
    noise = rng.normal(size=(400, 4))
    assert ridge_probe(encoded[:200], labels[:200], encoded[200:], labels[200:])["accuracy"] == 1.
    assert ridge_probe(noise[:200], labels[:200], noise[200:], labels[200:])["accuracy"] < .4


def test_probe_archive_tracks_decodability_by_attempt_and_skips_empty_stores():
    rng = np.random.default_rng(1)
    goal = rng.integers(9, size=200)
    # The goal becomes readable only from the second attempt.
    alpha = rng.normal(size=(200, 3, 9))
    alpha[:, 1:] += 4 * np.eye(9)[goal][:, None]
    archive = dict(state_alpha=alpha, state_beta=np.ones((200, 3, 9)),
                   state_long_term=np.zeros((200, 3, 0, 8)), hidden_goal_cell=goal)
    result = probe_archive(archive)
    rows = result["probes"]["goal_cell"]
    assert set(rows) == {"alpha", "beta", "alpha_over_beta"}
    accuracy = rows["alpha"]["accuracy_by_attempt"]
    assert accuracy[0] < .3 and min(accuracy[1:]) > .9
    with pytest.raises(ValueError, match="export-states"):
        probe_archive(dict(state_alpha=alpha))


def test_ad_noise_histories_label_executed_actions_and_anneal_exploration():
    factory = lambda: make_env("darkroom", 4, 20)
    seeds = np.arange(64)
    distilled = generate_histories(factory, "darkroom", seeds, objective="ad_noise")
    cloned = generate_histories(factory, "darkroom", seeds, objective="behavior_cloning")
    np.testing.assert_array_equal(distilled["target_actions"], distilled["raw_actions"])
    # Identical fraction and random stream until the annealing first differs.
    np.testing.assert_array_equal(distilled["raw_actions"][:20], cloned["raw_actions"][:20])
    returns = distilled["rewards"].reshape(4, 20, len(seeds)).sum(1).mean(1)
    assert returns[-1] > returns[0]
    assert returns[-1] > cloned["rewards"].reshape(4, 20, len(seeds)).sum(1).mean(1)[-1]


@pytest.mark.parametrize("core", ["cmm", "ctm"])
def test_warmstart_cmm_configs_match_the_ppo_construction_path(core):
    options = dict(core=core, width=16, ticks=1, stm_length=10, ltm_tokens=32, synapse_depth=8)
    assert warmstart_core_config(SimpleNamespace(memory_length=16, **options)) == core_config_from_args(
        SimpleNamespace(**options))


@pytest.mark.parametrize("label,core,ticks", CMM_STRUCTURE_VARIANTS)
def test_structure_variants_fit_the_parameter_cap_on_dark_room(label, core, ticks):
    env = make_env("darkroom", 16, 20)
    settings = dict(ticks=ticks, stm_length=32, ltm_tokens=32, synapse_depth=8,
                    bounded_synchronisation=False, decay_per_observation=False,
                    transformer_memory_length=320, transformer_layers=2, looped_memory_tokens=8)
    settings.update(VARIANT_OVERRIDES.get(label, {}))
    selected = matching_width(core, env, settings, 100_000)
    assert 85_000 <= selected["parameters"] <= 100_000
    if label == "cmm-stm":
        assert selected["core_config"]["ltm_tokens"] == 0
        full = build_policy("cmm", dict(selected["core_config"], ltm_tokens=32), env)
        assert selected["retained_state_bytes_per_lane"] < policy_size(full)["retained_state_bytes_per_lane"]
    env.close()
