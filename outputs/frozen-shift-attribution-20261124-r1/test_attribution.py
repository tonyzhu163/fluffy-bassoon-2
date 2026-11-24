"""CPU checks of the teacher-forced diagnostic on the harness fixture. Its rewards are synthetic."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teacher
from teacher import FRESH, PERMUTATION, array_digest, conditions, harness

import fixture
from fixture import Plant, Policy, case


class Viewed(Plant):
    def observation(self):
        raw = super().observation()
        raw["panda_agentview_image"] = raw["agentview_image"] + 40
        raw["panda_robot0_eye_in_hand_image"] = raw["robot0_eye_in_hand_image"] + 80
        return raw


def pack(raw, prompt, canonicalize_quat=False):
    entry = fixture.pack(raw, prompt)
    entry["observation/state"] = entry["observation/state"] + (1. if canonicalize_quat else 0.) + np.arange(9) * .01
    return entry


def tape(lengths=(5, 5, 5, 5), column=3, value=40 / 255):
    """A donor whose commands differ from the fixture's raw proposal only in `column`."""
    rng = np.random.default_rng(7)
    noise = rng.standard_normal((len(lengths), 10, 32)).astype(np.float32)
    sent = []
    for index, count in enumerate(lengths):
        rows = np.zeros((count, 7), dtype=np.float32)
        rows[:, 0] = .3
        rows[:, 2] = np.clip(.1 * noise[index, 0, 0], -.2, .2)
        rows[:, column] = value
        sent.append(rows)
    return dict(row=35, seed=42710335, prompt=harness.TASK, noise=noise, sent=sent,
                first_five=[rows + .001 for rows in sent], commands_sha256=array_digest(np.concatenate(sent)))


def run(layout=34, rules=None, **kw):
    sim, policy = Viewed(), Policy()
    record = teacher.teacher(sim, policy, tape(**kw), case("Kinova3", layout), rules or conditions(), pack)
    json.dumps(record, allow_nan=False)
    return record, sim, policy


def test_replay_consumes_the_donor_tape_and_reward_is_recorded_not_used():
    donor = tape()
    record, sim, policy = run()
    np.testing.assert_array_equal(np.concatenate(sim.executed), np.concatenate(donor["sent"]))
    assert record["controls"] == record["tape_controls"] == 20
    assert record["replay_success"] and record["first_reward_control"] == 20  # synthetic fixture reward
    assert len(policy.calls) == 4 * 10 + 1
    assert [len(event["proposals"]) for event in record["boundaries"]] == [11, 10, 10, 10]


def test_partial_final_window_and_tape_end():
    record, sim, _ = run(lengths=(5, 5, 5, 4))
    assert record["controls"] == 19 and not record["replay_success"] and record["first_reward_control"] is None
    assert teacher.score([record])["replay_rewarded"]["layouts"] == []


def test_each_condition_changes_only_its_own_input():
    record, _, _ = run()
    for event in record["boundaries"]:
        cells = event["proposals"]
        raw = cells["raw"]
        changed = {}
        for name, cell in cells.items():
            keys = {key for key in raw["inputs"] if cell["inputs"][key] != raw["inputs"][key]}
            if cell["prompt"] != raw["prompt"]:
                keys.add("prompt")
            if cell["noise_sha256"] != raw["noise_sha256"]:
                keys.add("noise")
            changed[name] = keys
        assert raw["noise_sha256"] == event["donor_noise_sha256"]
        assert changed["raw"] == set() and changed.get("raw_repeat", set()) == set()
        assert changed["noise_a"] == changed["noise_b"] == {"noise"}
        assert cells["noise_a"]["noise_sha256"] != cells["noise_b"]["noise_sha256"]
        assert changed["panda_external"] == {"observation/image"}
        assert changed["panda_both"] == {"observation/image", "observation/wrist_image"}
        for name in ("state_mapped", "state_zero", "state_shuffled"):
            assert changed[name] == {"observation/state"}
        assert changed["prompt_open"] == changed["prompt_put"] == {"prompt"}
        assert cells["prompt_open"]["prompt"] == teacher.OPEN and cells["prompt_put"]["prompt"] == teacher.PUT
    first = record["boundaries"][0]["proposals"]
    assert first["raw_repeat"]["proposal_sha256"] == first["raw"]["proposal_sha256"]


def test_state_conditions():
    raw = Viewed()
    raw.call("create", dict(case("Kinova3", 34), embodiment="nom"))
    obs = raw.observation()
    rules = {rule["name"]: rule for rule in conditions()}
    base = teacher.build(obs, "p", rules["raw"], pack)["observation/state"]
    assert not teacher.build(obs, "p", rules["state_zero"], pack)["observation/state"].any()
    np.testing.assert_array_equal(teacher.build(obs, "p", rules["state_shuffled"], pack)["observation/state"],
                                  base[list(PERMUTATION)])
    assert all(index != target for index, target in enumerate(PERMUTATION)) and sorted(PERMUTATION) == list(range(8))
    np.testing.assert_allclose(teacher.build(obs, "p", rules["state_mapped"], pack)["observation/state"], base + 1)


def test_missing_panda_render_is_recorded_and_views_can_be_dropped():
    record = teacher.teacher(Plant(), Policy(), tape(), case("Kinova3", 34), conditions(), pack)
    assert all(event["missing"] == ["panda_external", "panda_both"] for event in record["boundaries"])
    report = teacher.score([record])
    assert "panda_both" not in report["all"]["conditions"] and report["replays"][0]["missing_views"] == 4
    rules = conditions(views=False)
    assert not any(rule["view"] for rule in rules)
    record = teacher.teacher(Plant(), Policy(), tape(), case("Kinova3", 34), rules, pack)
    assert not any(event["missing"] for event in record["boundaries"])


def test_archived_source_view_proposals_score_against_the_donor():
    import archive
    report = archive.score()
    assert len(report["frames"]) == 10 and sum(row["on_donor_state"] for row in report["frames"]) == 2
    for row in report["frames"]:
        cells = row["donors"]["34"]
        assert cells["raw"] == cells["raw_repeat"] == cells["control"] and row["panda_minus_raw_rms"] > 0
    cell = report["summary"]["iiwa34/all_frames/donor/all"]
    assert cell["frames"] == 10 and cell["panda_minus_raw"]["layouts"] == 2


def test_lane_adapter_condition_sees_public_observations_only():
    seen = []

    class Probe(harness.Adapter):
        name = "probe"

        def observation(self, raw):
            seen.append(set(raw))
            return dict(raw, robot0_eef_pos=raw["robot0_eef_pos"] + 1)

        def observe(self, before, issued, after):
            seen.append(set(before) | set(after))

    record, _, _ = run(rules=conditions(adapters=[Probe()]))
    allowed = set(harness.PROPRIO) | set(teacher.CAMERAS)
    assert seen and all(keys <= allowed for keys in seen)
    cell = record["boundaries"][1]["proposals"]
    assert cell["adapter:probe"]["inputs"]["observation/state"] != cell["raw"]["inputs"]["observation/state"]


def test_score_orders_conditions_by_distance_to_the_donor():
    # The donor's column 3 equals what the fixture policy outputs for the Panda external image at tick 0.
    records = [run(layout)[0] for layout in (34, 35)]
    report = teacher.score(records)["all"]["conditions"]
    assert report["raw"]["donor/all"]["minus_raw"]["mean"] == 0
    assert report["prompt_open"]["donor/all"]["minus_raw"]["mean"] > 0
    assert report["prompt_open"]["donor/all"]["minus_fresh_noise"]["mean"] > 0
    first = teacher.distances(records[0])[0]["conditions"]
    assert first["panda_external"]["donor"]["all"] < first["raw"]["donor"]["all"]
    assert first["panda_external"]["donor"]["xyz"] == first["raw"]["donor"]["xyz"]
    assert first["noise_a"]["raw"]["all"] > 0 and first["raw"]["raw"]["all"] == 0
    assert report["raw"]["source/all"]["distance"]["layouts"] == 2
    assert teacher.score(records)["replays"][0]["replay_success"]


def test_layout_interval_resamples_whole_layouts():
    result = teacher.layout_interval({1: [0., 0., 0.], 2: [1.]})
    assert result["mean"] == .5 and result["layout_interval95"] == pytest.approx([.0375, .9625]) and result["boundaries"] == 4
    assert teacher.rms(np.ones((4, 7)), np.zeros((5, 7))) == 1.


def test_registered_donor_tapes_and_query_count():
    lengths = {row: teacher.load_tape(row) for row in (34, 35)}
    assert [sum(map(len, lengths[row]["sent"])) for row in (34, 35)] == [171, 184]
    plan = teacher.plan(list(teacher.PAIRS), conditions())
    assert plan["policy_calls"] == 1464 and sum(plan["pairs"].values()) == 146
    with pytest.raises(ValueError):
        teacher.parse_pairs("36:35")
