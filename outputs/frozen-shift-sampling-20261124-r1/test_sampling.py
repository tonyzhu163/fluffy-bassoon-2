"""CPU fixture checks for the sampling lane. Rewards in the fixture are synthetic."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "frozen-vla-proposal-shift-20261124-r1"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import adapter
import critic
import fixture
import harness
import headroom
import panel
import ur5e_motor


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", fixture.pack)

    def go(made, layout=14):
        sim = fixture.Plant()
        support = harness.collect_support(sim, fixture.case("Kinova3", 11))
        return harness.rollout(sim, fixture.Policy(), support, fixture.case("Kinova3", layout), made)
    return go


def test_registered_names_and_discovery():
    factories, manifest = panel.discover(names=["critic-k4", "random-k4", "single-k1", *(f"noise-{k}" for k in range(8))])
    assert len(factories) == 11
    assert not [entry for entry in manifest if entry["lane"] == HERE.name and "skipped" in entry]


def test_only_sampling_records_move(run):
    raw = run(harness.Adapter())
    for make, count in ((adapter.Critic, 4), (adapter.Random, 4), (adapter.Single, 1), (lambda: adapter.Stream(3), 1)):
        record = run(make())
        assert record["candidates"] == count
        assert record["reset"] == raw["reset"]
        assert record["untreated_opening_proposal_sha256"] == raw["untreated_opening_proposal_sha256"]
        for event in record["replans"]:
            assert event["prompt"] == event["original_prompt"] and not event["inputs_changed"]
            assert event["command_delta_linf"] == 0 and event["base_noise_sha256"] not in event["noise_sha256"]


def test_critic_random_and_single_share_candidates(run):
    chosen, random, single = run(adapter.Critic()), run(adapter.Random()), run(adapter.Single())
    first = [record["replans"][0]["noise_sha256"] for record in (chosen, random, single)]
    assert first[0] == first[1] and first[2] == first[0][:1]
    # The fixture ignores which sample was executed when packing the next input, so the whole stream is shared.
    assert [e["noise_sha256"] for e in chosen["replans"]] == [e["noise_sha256"] for e in random["replans"]]
    assert len({event["chosen"] for event in random["replans"]}) > 1
    assert run(adapter.Random())["commands_sha256"] == random["commands_sha256"]


def test_noise_streams_are_distinct_and_repeat_per_start(run):
    digests = {k: [e["noise_sha256"][0] for e in run(adapter.Stream(k))["replans"]] for k in range(3)}
    assert len({tuple(value) for value in digests.values()}) == 3
    assert [e["noise_sha256"][0] for e in run(adapter.Stream(1))["replans"]] == digests[1]
    assert [e["noise_sha256"][0] for e in run(adapter.Stream(1), layout=15)["replans"]] != digests[1]


def test_critic_choice_is_the_argmax_of_the_stored_weights(run):
    made = adapter.Critic()
    record = run(made)
    assert [event["chosen"] for event in record["replans"]] == [int(np.argmax(s)) for s in made.scores]
    artifact = json.loads(critic.ARTIFACT.read_text())
    assert len(artifact["weights"]) == critic.features(np.zeros(3), 0.).shape[-1]
    assert not any("Kinova3" in name or "IIWA" in name for name in artifact["fitted_on"])


def test_calibration_gain_recovers_a_linear_response():
    truth = np.array([[.01, 0, 0], [0, .02, 0], [.001, 0, .015]])
    means = np.array([[.4, 0, 0], [0, .4, 0], [0, 0, .4]])
    inputs = np.c_[means, np.zeros((3, 3))]
    assert np.allclose(critic.response_gain(inputs, 5 * means @ truth, np.eye(3)), truth)
    assert np.allclose(critic.response_gain(np.zeros((3, 6)), np.zeros((3, 3)), np.eye(3)), np.eye(3))


def test_exact_interval():
    low, high = headroom.exact_interval(0, 8)
    assert low == 0 and high == pytest.approx(1 - .025 ** (1 / 8), abs=1e-6)
    assert headroom.exact_interval(8, 8)[1] == 1
