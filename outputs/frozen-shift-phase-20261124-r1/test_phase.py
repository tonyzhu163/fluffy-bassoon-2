"""CPU checks of the phase-prompt adapters. Fixture rewards are synthetic and are not asserted on."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]
import adapter as lane
import panel
import schedule
import ur5e_motor
from fixture import Plant, Policy, case, pack
from harness import TASK, Adapter, collect_support, rollout

O, P = lane.OPENING, lane.PLACING
RESET_EDGE = 40
BAND = 19.25  # between the lowest and highest source crossing
# Drawer-front displacement in pixels at each replan; None hides the edge.
SCRIPT = [0, 0, 20, None, BAND, 20, None, 0, BAND, 20]


def frame(displacement, edge=RESET_EDGE):
    """Dark drawer body, bright front at a subpixel column, in the native 256-pixel image."""
    image = np.zeros((256, 256, 3), np.uint8)
    if displacement is not None:
        x = 2 * (edge + displacement)
        whole = int(x)
        image[:, whole:] = 200
        image[:, whole] = round(200 * (1 - (x - whole)))
    return image


class Drawer(Plant):
    script, edge = SCRIPT, RESET_EDGE

    def observation(self):
        raw = super().observation()
        raw["agentview_image"] = frame(self.script[min(self.tick // 5, len(self.script) - 1)], self.edge)
        return raw


@pytest.fixture(autouse=True)
def packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)


def episode(adapter, sim=None, horizon=50):
    sim = sim or Drawer()
    sim.horizon = 20  # calibration needs 15 uninterrupted controls
    support = collect_support(sim, case("Kinova3", 11))
    sim.horizon = horizon
    record = rollout(sim, Policy(), support, case("Kinova3", 1), adapter)
    json.dumps(record, allow_nan=False)
    return record


def prompts(record):
    return [event["prompt"] for event in record["replans"]]


def test_synthetic_frames_read_as_scripted():
    reset, _ = lane.read(None, frame(0))
    for displacement, score in ((0, 0), (18.5, 0), (20, 1)):
        front, got = lane.read(reset, frame(displacement))
        assert front - reset == pytest.approx(displacement, abs=.02) and got == score
    assert 0 < lane.read(reset, frame(BAND))[1] < .5
    assert lane.read(reset, frame(None)) == (None, 0)
    with pytest.raises(ValueError):
        lane.read(reset, frame(0).astype(np.float32) / 255)


def test_held_prompt_advances_holds_and_returns_on_reclosure():
    """A hidden edge or a displacement inside the source band keeps the prompt;
    a visible edge below every source crossing returns to opening."""
    assert prompts(episode(lane.Reader())) == [O, O, P, P, P, P, P, O, O, P]
    # `Signals.cue` applied each replan drops to opening whenever the edge is hidden or in the band.
    assert prompts(episode(lane.Instant())) == [O, O, P, O, O, P, O, O, O, P]
    assert prompts(episode(lane.Open())) == [O] * 10 and prompts(episode(lane.Place())) == [P] * 10
    assert prompts(episode(Adapter())) == [TASK] * 10


def test_hidden_reset_edge_never_leaves_opening():
    sim = Drawer()
    sim.script = [None, 20, 20]
    assert prompts(episode(lane.Reader(), sim, 15)) == prompts(episode(lane.Instant(), sim, 15)) == [O] * 3


def test_clock_switches_once_at_the_source_fitted_control():
    assert [lane.clock_open(t) for t in range(0, 300, 5)] == [False] * 12 + [True] * 48
    assert lane.clock_open(10_000)
    sim = Drawer()
    sim.script = [0]  # the clock ignores the camera
    assert prompts(episode(lane.Clock(), sim, 70)) == [O] * 12 + [P] * 2


def test_state_is_cleared_between_episodes():
    adapter = lane.Reader()
    first = episode(adapter)
    sim = Drawer()
    sim.edge = 55  # a different reset edge: the first episode's reference would read this as open
    assert prompts(episode(adapter, sim)) == prompts(first)


@pytest.mark.parametrize("make", lane.ADAPTERS, ids=lambda make: make.name)
def test_only_the_prompt_reaches_the_policy_changed(make):
    untreated, record = episode(Adapter()), episode(make())
    assert record["reset"] == untreated["reset"]
    assert record["untreated_opening_proposal_sha256"] == untreated["untreated_opening_proposal_sha256"]
    assert record["changed_replans"] == len(record["replans"]) == 10
    for event in record["replans"]:
        assert event["prompt"] in (O, P) and event["original_prompt"] == TASK
        assert event["inputs_changed"] == [] and event["command_delta_linf"] == 0
        assert (event["candidates"], event["chosen"]) == (1, 0)
        assert event["noise_sha256"] == [event["base_noise_sha256"]]


def test_panel_registers_the_lane():
    factories, manifest = panel.discover(names=[make.name for make in lane.ADAPTERS])
    assert list(factories) == ["phase-reader", "phase-instant", "phase-clock", "phase-open", "phase-place"]
    assert all(isinstance(make(), Adapter) for make in factories.values())
    assert not any("skipped" in entry for entry in manifest if entry["lane"] == HERE.name)


def test_trace_records_public_reader_values(tmp_path, monkeypatch):
    monkeypatch.setenv(lane.TRACE_ENV, str(tmp_path))
    monkeypatch.setattr(lane.Reader, "episodes", 0)
    record = episode(lane.Reader())
    trace = json.loads((tmp_path / "phase-reader-000.json").read_text())
    assert [row["prompt"] for row in trace["replans"]] == prompts(record)
    assert [row["control"] for row in trace["replans"]] == [event["control"] for event in record["replans"]]
    assert set(trace["replans"][0]) == {"control", "front_px", "reset_front_px", "score", "prompt"}


def test_archived_kinova_frames_agree_with_the_audit_drawer_joint(tmp_path):
    """Reference: simulator drawer joint from the CPU replay of the same native failure tapes."""
    report = schedule.main(tmp_path / "schedule.json")
    assert list(report["layouts"]) == [14, 15, 17, 36, 37, 38]
    for layout, expected in ((14, [O, O, O, P, P]), (15, [O] * 5)):
        entry = report["layouts"][layout]
        assert [row["control"] for row in entry["frames"]] == [0, 30, 45, 60, 90]
        assert [row["hysteretic"] for row in entry["frames"]] == expected
        assert entry["reader_agrees_with_audit_on_frames"]
    assert report["layouts"][14]["audit"]["first_open_replan"] == 60
    assert report["layouts"][15]["audit"]["first_open_replan"] is None
    assert json.loads((tmp_path / "schedule.json").read_text()) == json.loads(
        (HERE / "schedule.json").read_text())
