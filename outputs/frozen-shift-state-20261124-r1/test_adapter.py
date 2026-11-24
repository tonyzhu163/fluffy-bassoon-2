"""CPU checks of the state adapters on the harness fixture. Its rewards are synthetic."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "frozen-vla-proposal-shift-20261124-r1"))
sys.path.insert(0, str(HERE))
import adapter
import harness
import panel
import ur5e_motor
from fixture import Plant, Policy, case
from harness import Adapter, array_digest, collect_support, rollout

# Hand quaternion Kinova3 reported at the reset of layout 14 (measurement.json).
KINOVA3 = np.array([-0.9987, -0.0152, 0.0443, -0.0206])
KINOVA3 /= np.linalg.norm(KINOVA3)
MEAN = np.array([-0.08, 0.01, 1.09, 3.09, -0.13, -0.06, 0.039, -0.040])


def pack(raw, prompt):
    """The rotation arithmetic of ur5e_motor.to_openpi_observation without its image resize."""
    quat = np.asarray(raw["robot0_eef_quat_xyzw"], dtype=np.float64)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    rotation = quat[:3] * 2.0 * np.arccos(w) / den if den > 1e-10 else np.zeros(3)
    return {"prompt": prompt, "observation/state": np.concatenate((raw["robot0_eef_pos"], rotation, raw["robot0_gripper_qpos"])),
            "observation/image": raw["agentview_image"], "observation/wrist_image": raw["robot0_eye_in_hand_image"]}


class Arm(Plant):
    """The fixture with a hand that tilts over time and reports one quaternion sign."""

    sign = -1.

    def observation(self):
        raw = super().observation()
        tilt = adapter.quat_of_rotvec([0., .02 * (self.tick + 1), 0.])
        x, y, z, w = KINOVA3 * -1
        a, b, c, d = tilt
        quat = np.array([w * a + x * d + y * c - z * b, w * b - x * c + y * d + z * a,
                         w * c + x * b - y * a + z * d, w * d - x * a - y * b - z * c])
        raw["robot0_eef_quat_xyzw"] = self.sign * quat
        return raw


@pytest.fixture(autouse=True)
def packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)
    monkeypatch.setattr(adapter, "CHECKPOINT_STATS", HERE / "absent.json")


def episode(treatment, sign=-1., layout=2):
    """Run one fixture episode. Returns the record and, per replan, the public fields the hook
    received and the state packed from what it returned, checked against the harness receipt."""
    sim = Arm()
    sim.sign = sign
    support = collect_support(sim, case("Kinova3", 11))
    hook, received, given = treatment.observation, [], []

    def observation(raw):
        received.append({key: np.array(value) for key, value in raw.items()})
        seen = hook(raw)
        given.append(pack(seen, "")["observation/state"])
        return seen

    treatment.observation = observation
    record = rollout(sim, Policy(), support, case("Kinova3", layout), treatment)
    json.dumps(record, allow_nan=False)
    assert [event["policy_inputs"]["observation/state"] for event in record["replans"]] == [array_digest(s) for s in given]
    return record, received, given, support


def matrix(rotation):
    angle = np.linalg.norm(rotation)
    k = np.cross(np.eye(3), rotation / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_panel_finds_the_lane_and_runs_it_without_faults():
    names = ["state-mapped", "state-dropped", "state-shuffled", "state-mapped-shuffled"]
    factories, _ = panel.discover(names=names)
    assert list(factories) == names
    sim = Arm()
    support = collect_support(sim, panel.case("Kinova3", panel.CALIBRATION))
    cell = panel.run_start(sim, Policy(), support, panel.case("Kinova3", 14), factories)
    assert not cell["errors"] and set(cell["branches"]) == {panel.RAW, panel.REPEAT, *names}


def test_only_the_state_changes_and_nothing_downstream_is_written():
    for make in adapter.ADAPTERS:
        record, received, _, _ = episode(make())
        for event, raw in zip(record["replans"], received, strict=True):
            # A shuffled draw can equal the current state on the fixture, whose resets all coincide.
            assert event["inputs_changed"] in (["observation/state"], [] if "shuffled" in make.name else None)
            assert event["prompt"] == event["original_prompt"] and event["candidates"] == 1
            assert event["noise_sha256"] == [event["base_noise_sha256"]] and event["command_delta_linf"] == 0
            assert not set(raw) - {*harness.PROPRIO, "agentview_image", "robot0_eye_in_hand_image"}


def test_mapped_is_the_same_rotation_on_the_source_branch():
    reported = episode(Adapter())[2]
    record, _, mapped, _ = episode(adapter.StateMapped())
    assert record["changed_replans"] == len(record["replans"])
    for before, after in zip(reported, mapped, strict=True):
        assert before[3] < 0 < after[3]
        assert np.allclose(matrix(before[3:6]), matrix(after[3:6]), atol=1e-9)
        assert np.array_equal(before[[0, 1, 2, 6, 7]], after[[0, 1, 2, 6, 7]])
    assert np.linalg.norm(mapped[0] - reported[0]) == pytest.approx(2 * np.pi)
    # A body already on the source branch is left untreated.
    same = episode(adapter.StateMapped(), sign=1.)[0]
    assert same["changed_replans"] == 0 and same["commands_sha256"] == episode(Adapter(), sign=1.)[0]["commands_sha256"]


def test_dropped_is_the_source_mean_at_every_replan():
    record, _, given, _ = episode(adapter.StateDropped(MEAN))
    assert record["changed_replans"] == len(given) == 4
    assert all(np.allclose(state, MEAN, atol=1e-12) for state in given)
    assert len({array_digest(state) for state in given}) == 1


def test_dropped_reads_the_checkpoint_mean_and_refuses_the_other_branch(tmp_path, monkeypatch):
    assert adapter.source_mean()[1] == "archived:source_reference.json"
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {"state": {"mean": MEAN.tolist()}}}))
    monkeypatch.setenv(adapter.STATS_ENV, str(path))
    mean, origin = adapter.source_mean()
    assert np.array_equal(mean, MEAN) and origin == f"checkpoint:{path}"
    path.write_text(json.dumps({"state": {"mean": (MEAN * [1, 1, 1, -1, 1, 1, 1, 1]).tolist()}}))
    with pytest.raises(ValueError, match="Unusable"):
        adapter.source_mean()


@pytest.mark.parametrize("make", [adapter.StateShuffled, adapter.StateMappedShuffled])
def test_shuffled_gives_whole_earlier_states_of_this_body(make):
    treatment = make()
    record, received, given, support = episode(treatment)
    assert record["commands_sha256"] == episode(make())[0]["commands_sha256"]
    blank = dict(agentview_image=0, robot0_eye_in_hand_image=0)
    pool = [dict(frame, **blank) for frame in support["frames"]]
    for state, index, raw in zip(given, treatment.drawn, received, strict=True):
        assert index < len(pool)  # the current replan is not in the pool yet
        frame = adapter.to_source(dict(pool[index])) if make.source else pool[index]
        assert np.allclose(state, pack(frame, "")["observation/state"], atol=1e-12)
        assert (state[3] > 0) == make.source
        pool.append(raw)
    assert len(set(treatment.drawn)) > 1 and max(treatment.drawn) >= len(support["frames"])
