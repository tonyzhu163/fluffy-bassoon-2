"""History data and deployment causality; run on allocated Linux resources."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.vla_history import (FrozenHistoryController, HistoryConfig,
    HistoryNoisePolicy, export_policy, file_sha256, load_dataset, load_policy,
    sample_batch, save_dataset, trial_tokens, validate_trial)
from fluffy_bassoon.meta_rl.vla_history_train import supervised_update


def trial(length=6, *, offset=0.):
    done = np.zeros(length, dtype=bool)
    done[1::2] = True
    done[-1] = True
    starts = np.concatenate(([True], done[:-1]))
    noise = np.arange(length * 2, dtype=np.float32).reshape(length, 2) / 10 + offset
    return dict(observations=np.arange(length * 3, dtype=np.float32).reshape(length, 3) / 10,
        executed_noise=noise, target_noise=noise.copy(), rewards=np.arange(length, dtype=np.float32),
        dones=done, episode_starts=starts, teacher_steps=np.repeat(np.arange((length + 1) // 2), 2)[:length].astype(np.int64))


def policy():
    return HistoryNoisePolicy(HistoryConfig(3, 2, width=8, heads=2, layers=1, context_length=8))


def test_tokens_contain_only_previous_real_outcomes():
    data = trial()
    tokens = trial_tokens(data)
    assert np.array_equal(tokens[0, :3], data["observations"][0])
    assert np.array_equal(tokens[0, 3:], [0, 0, 0, 0, 1])
    assert np.array_equal(tokens[2, 3:], [*data["executed_noise"][1], data["rewards"][1], 1, 1])
    changed = deepcopy(data)
    changed["target_noise"][:] = 1000
    changed["rewards"][3:] = -1000
    changed["executed_noise"][3:] = -2000
    assert np.array_equal(trial_tokens(changed)[:4], tokens[:4])
    assert np.array_equal(trial_tokens(data, mode="episode")[2, 3:], [0, 0, 0, 0, 1])
    assert not trial_tokens(data, mode="static")[:, 3:-1].any()


def test_dataset_rejects_fixed_teacher_ad_and_fake_boundaries():
    data = trial()
    assert validate_trial(data, objective="ad") == (6, 3, 2)
    data["teacher_steps"][:] = 0
    with pytest.raises(ValueError, match="learning trajectory"):
        validate_trial(data, objective="ad")
    validate_trial(data, objective="behavior_cloning")
    data["target_noise"] += 1
    validate_trial(data, objective="dpt")
    with pytest.raises(ValueError, match="actually executed"):
        validate_trial(data, objective="behavior_cloning")
    data["episode_starts"][2] = False
    with pytest.raises(ValueError, match="complete episodes"):
        validate_trial(data, objective="dpt")


def test_ragged_serialization_and_checksum(tmp_path):
    path = tmp_path / "train.npz"
    manifest = save_dataset(path, [trial(), trial(3)], objective="ad",
                            metadata={"trial_ids": ["train:0", "train:1"]})
    rows, loaded = load_dataset(path, expected_sha256=manifest["sha256"])
    assert loaded == manifest
    assert [len(row["observations"]) for row in rows] == [6, 3]
    for key in rows[1]:
        assert np.array_equal(rows[1][key], trial(3)[key])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)


def test_sequence_padding_does_not_affect_loss_or_gradients():
    torch.manual_seed(7)
    first, second = policy(), policy()
    second.load_state_dict(first.state_dict())
    batch = sample_batch([trial(3)], first, np.random.default_rng(7), batch_size=2)
    loss = first.sequence_nll(*batch)
    loss.backward()
    tokens, targets, valid, starts = batch
    padded = (torch.cat((tokens, torch.full_like(tokens, float("nan")))),
        torch.cat((targets, torch.full_like(targets, float("nan")))),
        torch.cat((valid, torch.zeros_like(valid))), torch.cat((starts, torch.zeros_like(starts))))
    other = second.sequence_nll(*padded)
    other.backward()
    torch.testing.assert_close(loss, other, rtol=0, atol=0)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)


def test_frozen_deployment_weights_and_causal_reset_controls():
    torch.manual_seed(12)
    base = policy()
    before = {key: value.clone() for key, value in base.state_dict().items()}
    full = FrozenHistoryController(deepcopy(base), mode="full")
    episode = FrozenHistoryController(deepcopy(base), mode="episode")
    static = FrozenHistoryController(deepcopy(base), mode="static")
    obs = np.ones(3, np.float32)
    initial = []
    for controller in (full, episode, static):
        initial.append(controller.act(obs, deterministic=True))
        with pytest.raises(RuntimeError, match="Observe"):
            controller.act(obs)
        controller.observe(np.array([.9, -.8], np.float32), 1., True)
    assert np.array_equal(initial[0], initial[1])
    assert np.array_equal(initial[0], initial[2])
    outputs = [controller.act(obs, deterministic=True) for controller in (full, episode, static)]
    assert np.array_equal(outputs[1], initial[1])
    assert np.array_equal(outputs[2], initial[2])
    assert not np.array_equal(outputs[0], outputs[1])
    assert all(not parameter.requires_grad for parameter in full.policy.parameters())
    for key, value in full.policy.state_dict().items():
        assert torch.equal(value, before[key])
    full.reset_trial()
    assert np.array_equal(full.act(obs, deterministic=True), initial[0])


def test_variable_context_batches_use_masked_real_labels():
    learner = policy()
    samples = [sample_batch([trial()], learner, np.random.default_rng(seed),
        batch_size=4, max_context=4) for seed in range(4)]
    assert len({int(batch[2].sum(0)[0]) for batch in samples}) > 1
    for batch in samples:
        assert batch[0].shape[0] <= 4
        assert torch.isfinite(learner.sequence_nll(*batch))


def test_supervised_path_fits_labels_and_exports_frozen_candidate(tmp_path):
    torch.manual_seed(31)
    learner = policy()
    data = trial(4)
    # A constant coordinate target gives a small, deterministic functionality check.
    data["target_noise"][:] = .4
    learner.fit_observation_normalization([data])
    batch = sample_batch([data], learner, np.random.default_rng(0), batch_size=1, mode="static")
    before = float(learner.sequence_nll(*batch, mode="static").detach())
    optimizer = torch.optim.AdamW(learner.parameters(), lr=.01, weight_decay=0.)
    for _ in range(12):
        supervised_update(learner, optimizer, batch, mode="static")
    after = float(learner.sequence_nll(*batch, mode="static").detach())
    assert after < before - .2
    path = tmp_path / "candidate.pt"
    digest = export_policy(path, learner, manifest={"status": "candidate", "update": 12})
    loaded, manifest = load_policy(path, expected_sha256=digest)
    assert digest == file_sha256(path) and manifest["update"] == 12
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for key, value in learner.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
