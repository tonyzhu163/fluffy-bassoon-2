"""CPU fixture checks of the body-view adapters. Rewards are synthetic; no MuJoCo render runs here."""
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]
import harness  # puts the motor directory on sys.path
import ur5e_motor
from fixture import Plant, Policy, case, pack
from harness import Adapter, collect_support, rollout
import panel
import adapter as lane
from twin import Session


@pytest.fixture(autouse=True)
def packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)


class Backend:
    """Fixture twin: a second Plant. `panda` adds one grey level to the edited cameras."""

    def __init__(self, drift=None, tagged=False):
        self.plant, self.resets, self.issued, self.drift, self.tagged = Plant(), [], [], drift, tagged

    def reset(self, layout, seed):
        self.resets.append(layout)
        obs = self.plant.call("create", dict(case("Kinova3", layout), embodiment="nom", seed=seed))[1]["observation"]
        if self.tagged:
            obs["agentview_image"] = obs["agentview_image"] + np.uint8(layout)
        return obs

    def step(self, issued):
        self.issued.append(issued.tolist())
        obs = self.plant.call("chunk", issued)["observation"]
        if self.drift is not None and self.plant.tick >= self.drift:
            obs["robot0_eef_pos"] = obs["robot0_eef_pos"] + 1e-9
        return obs

    def render(self, mode, cameras):
        obs = self.plant.observation()
        return ({cam: obs[cam + "_image"] + np.uint8(mode == "panda") for cam in cameras},
                {cam: dict(changed=int(mode == "panda") * 64) for cam in cameras})


class Local:
    def __init__(self, backend):
        self.session, self.backend = Session(backend), backend

    def call(self, op, value):
        if op == "start":
            return self.session.start(value["observation"], value["candidates"])
        if op == "advance":
            return self.session.advance(value)
        return self.session.view(value["observation"], value["mode"], value["cameras"])


def episode(make, layout=14, arm="Kinova3", **backend):
    sim, twin = Plant(), Local(Backend(**backend))
    support = collect_support(sim, case(arm, 11))
    return rollout(sim, Policy(), support, case(arm, layout), make(twin)), twin.backend


def commands(record):
    return [event["executed_commands"] for event in record["replans"]]


def test_kinova_rerender_is_the_untreated_branch():
    raw, _ = episode(lambda twin: Adapter())
    record, backend = episode(lane.Kinova)
    assert commands(record) == commands(raw) and record["changed_replans"] == 0
    assert backend.issued == commands(record)


@pytest.mark.parametrize("make,changed", [
    (lane.PandaExternal, ["observation/image"]),
    (lane.PandaBoth, ["observation/image", "observation/wrist_image"]),
])
def test_panda_changes_only_its_cameras_and_twin_follows_consumed_commands(make, changed):
    record, backend = episode(make)
    assert [event["inputs_changed"] for event in record["replans"]] == [changed] * len(record["replans"])
    assert all(event["prompt"] == event["original_prompt"] and event["candidates"] == 1
               and event["command_delta_linf"] == 0 for event in record["replans"])
    assert backend.issued == commands(record) and backend.resets == [14]


def test_early_stop_inside_a_window_keeps_the_twin_on_the_consumed_commands():
    sim, twin = Plant(), Local(Backend())
    support = collect_support(sim, case("Kinova3", 11))
    sim.stop_at = twin.backend.plant.stop_at = 13
    record = rollout(sim, Policy(), support, case("Kinova3", 14), lane.PandaExternal(twin))
    assert record["controls"] == 13 and twin.backend.issued == commands(record)


def test_twin_off_the_public_trajectory_is_a_hook_fault():
    with pytest.raises(ValueError, match="left the public trajectory by control 10"):
        episode(lane.PandaExternal, drift=10)


def test_panda_start_is_untreated_and_other_arms_are_refused():
    raw, _ = episode(lambda twin: Adapter(), layout=1, arm="Panda")
    record, backend = episode(lane.PandaBoth, layout=1, arm="Panda")
    assert commands(record) == commands(raw) and record["changed_replans"] == 0 and backend.resets == []
    with pytest.raises(ValueError, match="registered for Kinova3"):
        episode(lane.PandaBoth, layout=1, arm="UR5e")


def test_start_is_identified_from_the_public_reset_observation():
    backend = Backend(tagged=True)
    session = Session(backend)
    public = lambda layout: {k: v for k, v in Backend(tagged=True).reset(layout, 0).items()
                             if k.endswith("_image") or k == "robot0_eef_pos"}
    assert session.start(public(15), lane.CANDIDATES) == dict(resets=2)
    assert session.start(public(15), lane.CANDIDATES) == dict(resets=1)   # other branches of the start
    assert session.start(public(17), lane.CANDIDATES) == dict(resets=1)   # next unseen start
    assert session.start(public(14), lane.CANDIDATES) == dict(resets=1)   # cached
    assert backend.resets == [14, 15, 15, 17, 14]
    with pytest.raises(ValueError, match="No registered start"):
        session.start(public(11), lane.CANDIDATES[:3])
    with pytest.raises(ValueError, match="without a matched start"):
        session.advance(np.zeros((5, 7)))


def test_panel_discovers_the_three_adapters():
    factories, manifest = panel.discover(names=["view_kinova", "view_panda_external", "view_panda_both"])
    assert list(factories) == ["view_kinova", "view_panda_external", "view_panda_both"]
    assert lane.LAYOUTS == tuple(layout for arm, layout in (*panel.STARTS["development"], *panel.STARTS["locked"]))
