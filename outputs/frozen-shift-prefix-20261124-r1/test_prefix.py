"""CPU checks on a tiny PyTorch stand-in. Rewards in the fixture are synthetic; nothing here is task success."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]
import adapter as lane
import fit as fitting
import prefix as px
import standin
import ur5e_motor
from fixture import Plant, case, pack
from harness import TASK, Adapter, collect_support, rollout


@pytest.fixture
def engine():
    return standin.Engine(standin.Tiny())


@pytest.fixture
def fitted(engine, tmp_path, monkeypatch):
    rows = standin.rows(48, "fit", 1)
    for arm in px.ARMS:
        params, receipt = fitting.fit(engine, rows, arm, updates=40, batch=8, lr=.02)
        px.save(tmp_path / f"prefix-{arm}.npz", params, receipt)
    monkeypatch.setenv(lane.FIT, str(tmp_path))
    monkeypatch.setattr(lane, "BUILD", standin.sampler)
    monkeypatch.setattr(lane.Prefixed, "policy_class", standin.Policy)
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)
    lane.REGISTRY.clear()
    yield tmp_path
    lane.uninstall(standin.Policy)


@pytest.mark.parametrize("arm, parameters", [("shift", standin.WIDTH), ("tokens", px.TOKENS * standin.WIDTH)])
def test_fit_changes_only_the_prefix(engine, arm, parameters):
    rows, held = standin.rows(48, "fit", 1), standin.rows(24, "held", 2)
    before = {name: value.clone() for name, value in engine.model.state_dict().items()}
    params, receipt = fitting.fit(engine, rows, arm, updates=40, batch=8, lr=.02)
    after = engine.model.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    assert not any(p.requires_grad or p.grad is not None for p in engine.model.parameters())
    assert receipt["parameters"] == px.count(params) == parameters
    assert receipt["base_leaves"] == len(before) and receipt["base_leaves_changed"] == 0
    assert receipt["last_loss"] < receipt["first_loss"]
    score = fitting.heldout(engine, held, {arm: params})
    assert score["prefixes"][arm]["minus_stock"] < 0 and score["blocks"] == 3


def test_fit_stops_when_a_base_parameter_moves(engine):
    class Leaky(standin.Engine):
        def value_and_grad(self, prefix, row, draw):
            self.model.out.bias += 1e-6
            return super().value_and_grad(prefix, row, draw)

    with pytest.raises(fitting.FrozenBaseChanged):
        fitting.fit(Leaky(engine.model), standin.rows(8, "fit", 1), "shift", updates=1, batch=2)


def test_zero_shift_is_the_unmodified_policy_and_tokens_extend_the_prefix(engine):
    row = standin.rows(1, "fit", 3)[0]
    model, obs = engine.model, row["obs"]
    noise = torch.randn(1, standin.HORIZON, standin.DIM)
    zero = {"shift": torch.zeros(standin.WIDTH)}
    assert torch.equal(model.sample_actions(obs, noise), model.sample_actions(obs, noise, zero))
    tokens, mask, ar = model.embed_prefix(obs)
    extra = {"tokens": torch.ones(px.TOKENS, standin.WIDTH)}
    wide, wide_mask, wide_ar = px.inject(torch.cat, standin.cast, tokens, mask, ar, extra, standin.TEXT)
    assert torch.equal(wide[:, :tokens.shape[1]], tokens) and wide.shape[1] == tokens.shape[1] + px.TOKENS
    assert wide_mask[:, -px.TOKENS:].all() and not wide_ar.any() and len(wide_ar) == wide.shape[1]
    shifted, same_mask, _ = px.inject(torch.cat, standin.cast, tokens, mask, ar,
                                      {"shift": torch.ones(standin.WIDTH)}, standin.TEXT)
    moved = (shifted != tokens).any(-1)
    assert torch.equal(moved[:, -standin.TEXT:], mask[:, -standin.TEXT:]) and not moved[:, :-standin.TEXT].any()
    assert torch.equal(same_mask, mask)
    with pytest.raises(ValueError):
        px.inject(torch.cat, standin.cast, tokens, mask, ar, {"shift": torch.ones(3)}, standin.TEXT)


def test_prefix_file_is_bound_to_its_receipt(tmp_path):
    params = px.initial("tokens", np.ones(standin.WIDTH), 7)
    meta = px.save(tmp_path / "p.npz", params, dict(seed=7))
    loaded, read = px.load(tmp_path / "p.npz")
    assert read == meta and np.array_equal(loaded["tokens"], params["tokens"])
    control = px.random_like({"shift": np.full(4, 2., np.float32)}, 7)
    assert np.isclose(np.linalg.norm(control["shift"]), 4) and not np.allclose(control["shift"], 2)
    np.savez(tmp_path / "bad.npz", meta=np.array(__import__("json").dumps(meta)), tokens=params["tokens"] + 1)
    with pytest.raises(ValueError):
        px.load(tmp_path / "bad.npz")


def episode(adapter, policy):
    sim = Plant()
    support = collect_support(sim, case("Kinova3", 11))
    return rollout(sim, policy, support, case("Kinova3", 2), adapter)


def test_adapters_inject_the_prefix_and_leave_untreated_calls_alone(fitted):
    assert [a.name for a in lane.ADAPTERS] == ["prefix-shift", "prefix-tokens", "prefix-random"]
    reference_policy = standin.Policy()
    reference = episode(Adapter(), reference_policy)  # before anything is installed
    base = standin.Engine(standin.Tiny()).digest()
    commands = {}
    for make in lane.ADAPTERS:
        policy = standin.Policy()
        record = episode(make(), policy)
        assert all(event["prompt"] == TASK + lane.tag(make.name) for event in record["replans"])
        assert not any(event["inputs_changed"] for event in record["replans"])
        assert set(policy.prompts) == {TASK}  # the tag never reaches the policy's language input
        assert record["untreated_opening_proposal_sha256"] == reference["untreated_opening_proposal_sha256"]
        assert record["commands_sha256"] != reference["commands_sha256"]
        assert standin.Engine(policy._model).digest() == base
        commands[make.name] = record["commands_sha256"]
    assert len(set(commands.values())) == 3
    after = episode(Adapter(), standin.Policy())  # wrapper installed, no tag
    assert after["commands_sha256"] == reference["commands_sha256"]
    assert [e["proposal_sha256"] for e in after["replans"]] == [e["proposal_sha256"] for e in reference["replans"]]


def test_missing_fit_is_an_error_not_an_untreated_run(monkeypatch):
    monkeypatch.delenv(lane.FIT, raising=False)
    with pytest.raises(RuntimeError):
        lane.Shift()
