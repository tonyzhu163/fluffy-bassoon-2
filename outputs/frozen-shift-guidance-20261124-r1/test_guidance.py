"""CPU checks of the source-guidance lane. Fixture rewards are synthetic; nothing here is task evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1")]

import adapter
import analyze
import donors
import panel
import run_guidance
import ur5e_motor
from fixture import Plant, Policy, case, pack
from harness import PROPRIO, collect_support, rollout

PUBLIC = {"agentview_image", "robot0_eye_in_hand_image", *PROPRIO}


@pytest.fixture(autouse=True)
def packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)


@pytest.fixture(scope="module")
def tapes():
    return donors.library()


def tape(layout, moves, drawer=None, jaw=None, arm="IIWA", rotation=None):
    """A donor whose window w issues the constant translation moves[w] / 5."""
    moves = np.asarray(moves, dtype=np.float64)
    commands = np.zeros((5 * len(moves), 7))
    commands[:, :3] = np.repeat(moves / 5, 5, axis=0)
    commands[:, 6] = np.repeat(np.where(jaw if jaw is not None else np.zeros(len(moves), bool), 1, -1), 5)
    drawer = np.zeros(len(moves), bool) if drawer is None else np.asarray(drawer)
    return donors.Tape(arm, layout, commands, drawer, "test", rotation)


def first(image=0, quat=(0, 0, 0, 1)):
    return dict(observation=dict(agentview_image=np.full((8, 8, 3), image, np.uint8), robot0_eef_quat=quat))


def proposals(*moves):
    out = np.zeros((len(moves), 10, 7), np.float32)
    out[:, :5, :3] = np.asarray(moves)[:, None] / 5
    out[:, :, 6] = -1
    return list(out)


def shut(reset_front, image):
    return False


def test_archived_donors_are_verified_successes_with_the_expected_phase_order(tapes, monkeypatch):
    assert [t.name for t in tapes] == ["IIWA-34", "IIWA-35", "IIWA-36", "IIWA-37", "Panda-14", "Panda-15"]
    assert [len(t.commands) for t in tapes] == [171, 184, 248, 251, 173, 233]
    for t in tapes:
        keys, coordinates, targets = t.windows()
        assert t.success and keys[0] == (False, False) and keys[-1] == (True, True)
        assert len(keys) == len(coordinates) == len(targets) == len(t.commands) // 5
    assert donors.cue_tape("Kinova3", 36).success is False
    monkeypatch.setitem(donors.SHA256, "raw-tapes.tar.gz", "0" * 64)
    with pytest.raises(ValueError, match="digest"):
        donors.cue_tape("IIWA", 34)


def test_progress_restarts_on_a_phase_change_and_ignores_elapsed_time():
    progress = donors.Progress()
    still = np.zeros((5, 7))
    still[:, 6] = -1
    move = still.copy()
    move[:, 0] = .2
    for _ in range(7):  # waiting leaves the coordinate where it was
        assert progress.at_replan(False) == ((False, False), pytest.approx(np.zeros(3)))
        progress.issued(still)
    progress.issued(move)
    assert progress.at_replan(False)[1] == pytest.approx([1, 0, 0])
    assert progress.at_replan(True) == ((True, False), pytest.approx(np.zeros(3)))
    grip = move.copy()
    grip[:, 6] = 1
    progress.issued(grip)
    assert progress.at_replan(True) == ((True, True), pytest.approx(np.zeros(3)))


def test_alignment_follows_progress_where_time_alignment_follows_the_clock():
    # The donor advances one unit of x per window; the query has stalled for three windows.
    donor = tape(34, [[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    samples = proposals([0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1])
    chosen = {}
    for align in ("progress", "time"):
        guide = adapter.Guidance("g", [donor], align=align, reader=shut)
        guide.reset({}, first())
        stall = np.zeros((5, 7))
        stall[:, 6] = -1
        for _ in range(3):
            guide.choose(first()["observation"], samples)
            guide.observe(None, stall, None)
        chosen[align] = guide.choose(first()["observation"], samples)
        assert [event["chosen"] for event in guide.log[:3]] == [2, 2, 2]
    assert chosen == dict(progress=2, time=1)


def test_nearest_donor_window_wins_and_a_missing_phase_keeps_the_untreated_sample():
    guide = adapter.Guidance("g", [tape(34, [[1, 0, 0]]), tape(35, [[0, 0, 2]])], reader=shut)
    guide.reset({}, first())
    assert guide.choose(first()["observation"], proposals([0, 1, 0], [0, 0, 1.6], [.2, 0, 0], [0, -1, 0])) == 1
    assert guide.log[-1]["donor"] == "IIWA-35" and not guide.log[-1]["fallback"]
    opened = adapter.Guidance("g", [tape(34, [[1, 0, 0]])], reader=lambda reset_front, image: True)
    opened.reset({}, first())
    assert opened.choose(first()["observation"], proposals([0, 1, 0], [1, 0, 0])) == 0
    assert opened.log[-1]["fallback"] and opened.log[-1]["donor_windows"] == {}


def test_tool_frame_follows_tool_rotation_since_reset_and_does_not_cancel_a_base_mount():
    quarter = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])  # the query arm is mounted a quarter turn round
    donor = tape(14, [[1, 0, 0]], arm="Panda", rotation=np.eye(3)[None])
    samples = proposals([1, 0, 0], [0, 1, 0])  # in the query base frame the donor's motion is +y
    start = first(quat=(0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)))
    assert adapter.rotation_matrix(start["observation"]["robot0_eef_quat"]) == pytest.approx(quarter)
    # Rotation is measured from reset, so a differently mounted base is not cancelled in either frame.
    for frame in ("reset", "tool"):
        guide = adapter.Guidance("g", [donor], frame=frame, reader=shut)
        guide.reset({}, start)
        assert guide.choose(start["observation"], samples) == 0
    turned = dict(start["observation"], robot0_eef_quat=(0, 0, 1, 0))  # the tool has since turned a quarter
    guide = adapter.Guidance("g", [donor], frame="tool", reader=shut)
    guide.reset({}, start)
    assert guide.choose(turned, samples) == 1
    assert donors.displacement(samples[1], quarter) == pytest.approx([1, 0, 0])
    with pytest.raises(ValueError, match="orientation"):
        adapter.Guidance("g", [tape(34, [[1, 0, 0]])], frame="tool")


def test_donors_exclude_the_query_layout_and_the_same_layout_reference_uses_only_it(tapes):
    for layout in (14, 15, 17, 34, 36, 38):
        made = adapter.for_layout(layout, tapes)
        names = [name for name, *_ in made["guidance"]().donors]
        assert f"IIWA-{layout}" not in names and f"Panda-{layout}" not in names
        assert len(names) == 6 - sum(t.layout == layout for t in tapes)
        assert [name for name, *_ in made["guidance_time"]().donors] == names
        if layout in (17, 38):
            assert "guidance_same" not in made
        else:
            assert {name.split("-")[1] for name, *_ in made["guidance_same"]().donors} == {str(layout)}
    shared, _ = panel.discover(names=["guidance_iiwa3435", "guidance_random"])
    assert [name for name, *_ in shared["guidance_iiwa3435"]().donors] == ["IIWA-34", "IIWA-35"]
    assert not {34, 35} & {layout for _, layout in panel.STARTS["locked"]}
    assert not {34, 35} & {14, 15, 17, 36, 37, 38}


def test_four_candidates_share_one_stream_and_the_random_control_ignores_the_observation():
    base = np.zeros((10, 7), np.float32)
    drawn = [adapter.Sampler().noises(base, np.random.default_rng(3)) for _ in range(2)]
    assert len(drawn[0]) == adapter.K == 4 and drawn[0][0] is base
    assert all(np.array_equal(a, b) for a, b in zip(*drawn))
    assert len({sample.tobytes() for sample in drawn[0]}) == 4
    control = adapter.Random()
    control.reset({}, first())
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(200):
        samples = list(rng.normal(size=(4, 10, 7)).astype(np.float32))
        picks.append(control.choose({}, samples))
        assert control.choose({"anything": 1}, samples) == picks[-1]
    assert np.bincount(picks, minlength=4).min() > 30


class Spy(adapter.Guidance):
    def reset(self, support, first):
        self.seen = [set(first["observation"])]
        super().reset(support, first)

    def choose(self, raw, proposals):
        self.seen.append(set(raw))
        return super().choose(raw, proposals)

    def observe(self, before, issued, after):
        self.seen += [set(before), set(after)]
        super().observe(before, issued, after)


@pytest.mark.parametrize("frame", ["reset", "tool"])
def test_fixture_rollout_records_four_candidates_and_only_public_inputs_reach_the_adapter(tapes, frame):
    sim, policy = Plant(), Policy()
    support = collect_support(sim, case("Kinova3", 11))
    raw = rollout(sim, policy, support, case("Kinova3", 14), adapter.Sampler())
    chosen = [t for t in tapes if t.layout != 14 and (frame == "reset" or t.rotation is not None)]
    spy = Spy("guidance", chosen, frame=frame)  # the checked-in camera reader runs on the fixture frames
    record = rollout(sim, policy, support, case("Kinova3", 14), spy)
    json.dumps(record, allow_nan=False)
    events = record["replans"]
    assert all(keys <= PUBLIC for keys in spy.seen)
    assert [event["candidates"] for event in events] == [4] * len(events)
    assert [event["chosen"] for event in events] == [event["chosen"] for event in spy.log]
    assert not any(event["inputs_changed"] or event["command_delta_linf"] for event in events)
    assert record["untreated_opening_proposal_sha256"] == raw["untreated_opening_proposal_sha256"]
    assert events[0]["noise_sha256"] == raw["replans"][0]["noise_sha256"]
    for event, log in zip(events, spy.log):
        assert event["selected_proposal_sha256"] == event["proposal_sha256"][log["chosen"]]
        assert log["fallback"] or log["chosen"] == int(np.argmin(log["distance"]))


def test_lane_driver_runs_every_arm_of_the_comparison_on_the_fixture(tapes, tmp_path):
    report = run_guidance.run(Plant(), Policy(), tmp_path, [("Kinova3", 14), ("Kinova3", 17)], tapes)
    cells = {layout: json.loads((tmp_path / f"Kinova3-{layout}.json").read_text()) for layout in (14, 17)}
    assert set(cells[14]["branches"]) == {"raw", "repeat", "guidance", "guidance_time", "guidance_random", "guidance_same"}
    assert "guidance_same" not in cells[17]["branches"]
    choices = json.loads((tmp_path / "Kinova3-14-choices.json").read_text())
    assert all(len(log) == len(cells[14]["branches"][name]["replans"]) for name, log in choices.items())
    assert json.loads((tmp_path / "summary.json").read_text()) == json.loads(json.dumps(report))
    with pytest.raises(ValueError, match="Locked"):
        run_guidance.main([str(tmp_path), "--starts", "locked", "--list"])


def test_saved_offline_analysis_matches_the_archives(tapes):
    saved = json.loads((HERE / "analysis.json").read_text())
    rows = analyze.heldout(tapes)
    for name, row in rows.items():
        assert row == pytest.approx(saved["leave_layout_out"]["per_query"][name])
    again = analyze.errors(tapes[0], tapes[0])
    assert again["progress"] == pytest.approx(0) and again["time"] == pytest.approx(0)
