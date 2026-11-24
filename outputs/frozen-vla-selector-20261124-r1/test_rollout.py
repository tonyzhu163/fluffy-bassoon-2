"""CPU integration fixture for the native motor loop. Its rewards are synthetic."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"))
import ur5e_motor
import experiment
from rollout import TASK, SelectorEpisode, collect_support, rollout


class Policy:
    def __init__(self):
        self.calls = []

    def infer(self, item, *, noise):
        self.calls.append((item["prompt"], ur5e_motor.array_digest(noise), item["observation/state"].tobytes()))
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = .3
        if item["prompt"] != TASK:
            actions[:, 1] = 1.
        return {"actions": actions}


class Plant:
    def __init__(self):
        self.horizon = 20
        self.stop_at = None
        self.action_schema = dict(low=[-1.] * 7, high=[1.] * 7)
        self.corrupt = False

    def call(self, op, value):
        if op == "create":
            self.tick, self.cued = 0, False
            self.executed = []
            joints = 6 if value["arm"] == "UR5e" else 7
            self.image = np.zeros((32, 32, 3), np.uint8)
            self.raw = dict(robot0_eef_pos=np.zeros(3), robot0_gripper_qpos=np.zeros(2),
                            robot0_joint_pos=np.zeros(joints), agentview_image=self.image)
            self.physics = f"fixture-{value['arm']}-{value['init_state_id']}"
            return dict(max_steps=self.horizon), dict(
                observation=self.raw, anchor_eef_pos=self.raw["robot0_eef_pos"], language_instruction=TASK,
                action_schema=self.action_schema, agentview_64=self.image,
                environment_reproducibility=dict(scene_fingerprint="fixture", perturbation_plan_fingerprint="fixture",
                                                 initial_state_fingerprint=self.physics),
                model_identity=dict(arm=value["arm"], model_class="fixture", joint_count=joints),
                renderer_vendor="fixture", renderer="fixture", worker_step_index=0)
        if op == "fork_digest":
            if self.corrupt:
                return np.random.default_rng().bytes(8).hex()
            return self.physics + ur5e_motor.array_digest(self.raw["robot0_eef_pos"])
        assert op == "chunk"
        end = self.horizon if self.stop_at is None else min(self.horizon, self.stop_at)
        issued = value[:end - self.tick]
        self.executed.append(issued.copy())
        self.cued |= bool((issued[:, 1] == 1.).any())
        self.tick += len(issued)
        self.raw = dict(self.raw, robot0_eef_pos=self.raw["robot0_eef_pos"] + .01 * issued[:, :3].sum(axis=0))
        terminal = self.tick == end
        success = self.cued and terminal
        return dict(observation=self.raw, steps=len(issued), rewards=[0.] * (len(issued) - 1) + [float(success)],
                    terminated=success, truncated=terminal and not success, success=success,
                    language_instruction=TASK, positions=[self.raw["robot0_eef_pos"]], agentview_64=self.image,
                    gripper_positions=[self.raw["robot0_gripper_qpos"]], joint_positions=[self.raw["robot0_joint_pos"]])


@pytest.fixture(autouse=True)
def public_packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", lambda raw, prompt: {
        "prompt": prompt, "observation/state": raw["robot0_eef_pos"], "observation/image": raw["agentview_image"]})


def test_one_requery_same_noise_and_observation_then_restore():
    sim, policy = Plant(), Policy()
    support = collect_support(sim, experiment.case("UR5e", 11))
    result = rollout(sim, policy, support, experiment.case("UR5e", 0), 5, "cue")
    assert result["fork"] is not None
    calls = policy.calls
    assert [entry[0] for entry in calls] == [TASK, TASK, "open the top drawer", TASK, TASK]
    assert calls[1][1:] == calls[2][1:]
    assert result["record"]["success"] and result["record"]["controls"] == sim.horizon
    event, = result["events"]
    assert (event["control"], event["branch"], event["prompt"]) == (5, "cue", calls[2][0])
    assert event["noise_sha256"] == calls[1][1]
    np.testing.assert_array_equal(event["nominal_bounded"], sim.executed[0])
    np.testing.assert_array_equal(event["selected_bounded"], sim.executed[1])
    np.testing.assert_array_equal(event["executed_commands"], sim.executed[1])
    assert event["executed_delta_linf"] == 1.


def test_cue_changes_raw_proposal_but_clipping_and_unused_tail_leave_commands_unchanged(monkeypatch):
    sim, policy = Plant(), Policy()
    support = collect_support(sim, experiment.case("Panda", 11))
    sim.action_schema["high"][0] = .75
    nominal = np.zeros((10, 7), dtype=np.float32)
    nominal[:, :2] = [2., -2.]
    cued = nominal.copy()
    cued[:5, :2] = [3., -3.]
    cued[5:, 2] = .5
    buffer = np.empty_like(nominal)

    def infer(item, *, noise):
        buffer[:] = nominal if item["prompt"] == TASK else cued
        return {"actions": buffer}

    monkeypatch.setattr(policy, "infer", infer)
    result = rollout(sim, policy, support, experiment.case("Panda", 0), 5, "cue")
    event, = result["events"]
    assert event["nominal_proposal_sha256"] == result["fork"]["proposal"] == ur5e_motor.array_digest(nominal)
    assert event["selected_proposal_sha256"] == ur5e_motor.array_digest(cued)
    expected = np.zeros((5, 7), dtype=np.float32)
    expected[:, :2] = [.75, -1.]
    for key in ("nominal_bounded", "selected_bounded", "executed_commands"):
        np.testing.assert_array_equal(event[key], expected)
    np.testing.assert_array_equal(event["executed_commands"], sim.executed[1])
    assert event["executed_delta_linf"] == 0.


def test_execution_receipt_uses_commands_at_simulator_handoff():
    sim = Plant()
    support = collect_support(sim, experiment.case("Panda", 11))
    sim.stop_at = 7
    episode = SelectorEpisode(sim, Policy(), support, 5, "continue")
    episode.call("create", dict(experiment.case("Panda", 0), embodiment="nom"))
    step = episode.call("chunk", np.zeros((5, 7), dtype=np.float32))
    item = ur5e_motor.to_openpi_observation(step["observation"], TASK)
    proposal = episode.infer(item, noise=np.zeros((10, 32), dtype=np.float32))["actions"]
    # Distinguish the command at the motor/simulator handoff from the selected proposal.
    submitted = proposal[:5].copy()
    submitted[:, 3] = [0., .25, .5, .75, 1.]
    episode.call("chunk", submitted)
    event, = episode.events
    np.testing.assert_array_equal(event["selected_bounded"], proposal[:5])
    np.testing.assert_array_equal(event["executed_commands"], sim.executed[-1])
    assert (event["submitted_steps"], event["executed_steps"], event["executed_delta_linf"]) == (5, 2, .25)


@pytest.mark.parametrize(("horizon", "stop_at"), [(18, None), (20, 17)])
def test_repeated_decisions_measure_only_consumed_commands(monkeypatch, horizon, stop_at):
    sim, policy = Plant(), Policy()
    support = collect_support(sim, experiment.case("Panda", 11))
    sim.horizon, sim.stop_at = horizon, stop_at
    branches = ("correct", "continue", "correct")
    choices = iter(branches)
    monkeypatch.setattr("rollout.choose", lambda *_: next(choices))

    def correct(self, raw, canonical):
        changed = canonical.copy()
        changed[2:, 1] = .5
        return changed

    monkeypatch.setattr("rollout.Signals.correct", correct)
    result = rollout(sim, policy, support, experiment.case("Panda", 0), 5, "generic",
                     model={"kind": "generic"}, repeated=True)
    events = result["events"]
    end = horizon if stop_at is None else stop_at
    assert result["record"]["truncated"] and result["record"]["controls"] == end
    assert [e["control"] for e in events] == [5, 10, 15]
    assert [e["branch"] for e in events] == list(branches)
    assert [e["submitted_steps"] for e in events] == [5, 5, horizon - 15]
    assert [e["executed_steps"] for e in events] == [5, 5, end - 15]
    for event, call, issued in zip(events, policy.calls[1:], sim.executed[1:], strict=True):
        assert (event["nominal_prompt"], event["noise_sha256"]) == call[:2]
        nominal = sim.executed[0]
        selected = nominal.copy()
        if event["branch"] == "correct":
            selected[2:, 1] = .5
        np.testing.assert_array_equal(event["nominal_bounded"], nominal)
        np.testing.assert_array_equal(event["selected_bounded"], selected)
        np.testing.assert_array_equal(event["executed_commands"], issued)
        np.testing.assert_array_equal(issued, selected[:len(issued)])
        assert event["executed_delta_linf"] == np.max(np.abs(issued - nominal[:len(issued)]))


def test_calibration_separation_and_fork_identity():
    sim, policy = Plant(), Policy()
    query = experiment.case("Panda", 0)
    with pytest.raises(ValueError, match="independent calibration"):
        rollout(sim, policy, collect_support(sim, query), query, 5, "continue")
    support = collect_support(sim, experiment.case("Panda", 11))
    sim.corrupt = True
    with pytest.raises(ValueError, match="unmatched prefix"):
        experiment.paired_case(sim, policy, support, query, 5, {})


def test_early_terminal_remains_in_denominator():
    sim, policy = Plant(), Policy()
    support = collect_support(sim, experiment.case("Panda", 11))
    sim.horizon = 3
    cell = experiment.paired_case(sim, policy, support, experiment.case("Panda", 0), 5, {})
    assert experiment.training_row(cell) is None
    assert all(trial["events"] == [] for trial in cell["branches"].values())
    assert experiment.summarize([cell])["one_choice_hindsight"]["total"] == 1


def test_full_train_and_held_arm_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "BOUNDARIES", (5,))
    reports = experiment.run(Plant(), Policy(), tmp_path)
    models = json.loads((tmp_path / "selectors.json").read_text())
    assert set(models) == set(experiment.HEADS)
    for mode in ("generic", "factorized", "generic_online", "factorized_online", "one_choice_hindsight"):
        assert reports["Kinova3"][mode]["rescues"] == 4
    assert reports["Kinova3"]["repeat"]["rescues"] == 0
    trial = json.loads((tmp_path / "Kinova3-36.json").read_text())["branches"]["factorized_online"]
    assert [e["control"] for e in trial["events"]] == [5, 10, 15]


@pytest.mark.parametrize(("layout", "expected"), [(14, "put the bowl in the top drawer"), (15, "open the top drawer")])
def test_phase_cue_on_archived_kinova_camera_frames(layout, expected):
    from signals import Signals

    folder = Path(__file__).resolve().parents[1] / "frozen-vla-source-view-20261123-r1/evidence/cpu01"
    def frame(control):
        with np.load(folder / f"s{layout}-t{control:03d}.npz") as data:
            return dict(agentview_image=data["raw_agentview"], robot0_joint_pos=data["robot0_joint_pos"],
                        robot0_gripper_qpos=data["robot0_gripper_qpos"])

    # The cue does not consume motion calibration. Use a zero support for this camera-only check.
    support = dict(inputs=np.zeros((3, 6)), response=np.zeros((3, 3)), joints=np.zeros((3, 7)))
    signals = Signals(support, frame(0))
    assert signals.cue(frame(60)) == expected
