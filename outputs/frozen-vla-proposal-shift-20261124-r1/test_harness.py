"""CPU checks of the adapter harness on the deterministic fixture. Its rewards are synthetic."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness
import panel
import ur5e_motor
from fixture import Plant, Policy, case, pack
from harness import PROPRIO, TASK, Adapter, array_digest, collect_support, get_embodiment_spec, rollout


@pytest.fixture(autouse=True)
def packer(monkeypatch):
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", pack)


def episode(adapter, layout=1, arm="Kinova3", **plant):
    sim, policy = Plant(), Policy()
    support = collect_support(sim, case(arm, 11))
    for key, value in plant.items():
        setattr(sim, key, value)
    record = rollout(sim, policy, support, case(arm, layout), adapter)
    json.dumps(record, allow_nan=False)
    return record, sim, policy


def consumed(record):
    return [event["executed_commands"] for event in record["replans"]]


class Image(Adapter):
    name = "image"

    def observation(self, raw):
        return dict(raw, agentview_image=np.full_like(raw["agentview_image"], 200))


class State(Adapter):
    name = "state"

    def observation(self, raw):
        return dict(raw, robot0_eef_pos=raw["robot0_eef_pos"] + .5)


class Prompt(Adapter):
    name = "prompt"

    def prompt(self, raw, original):
        return "open the top drawer"


class Sample(Adapter):
    name = "sample"

    def noises(self, base, rng):
        self.drawn = [base, *rng.standard_normal((2, *base.shape)).astype(np.float32)]
        return self.drawn

    def choose(self, raw, proposals):
        self.proposals = proposals
        return 2


class Command(Adapter):
    name = "command"

    def command(self, raw, proposal):
        window = proposal[:5].copy()
        window[:, 2] = -3.
        return window


@pytest.mark.parametrize(("horizon", "stop_at"), [(20, None), (18, None), (20, 7)])
def test_identity_adapter_reproduces_the_plain_motor_loop(horizon, stop_at):
    plain_sim, plain_policy = Plant(), Policy()
    plain_sim.horizon, plain_sim.stop_at = horizon, stop_at
    plain = ur5e_motor.run_episode(plain_sim, plain_policy, case("Kinova3", 1), get_embodiment_spec("nom"))
    record, sim, policy = episode(Adapter(), horizon=horizon, stop_at=stop_at)
    assert len(sim.executed) == len(plain_sim.executed)
    for mine, theirs, event in zip(sim.executed, plain_sim.executed, record["replans"], strict=True):
        np.testing.assert_array_equal(mine, theirs)
        np.testing.assert_array_equal(event["executed_commands"], theirs)
    assert policy.calls == plain_policy.calls
    for key in ("reset", "native_rewards", "controls", "success", "terminated", "truncated",
                "opening_proposal_sha256"):
        assert record[key] == plain[key]
    assert record["untreated_opening_proposal_sha256"] == plain["opening_proposal_sha256"]
    assert record["controls"] == (horizon if stop_at is None else stop_at)
    assert (record["changed_replans"], record["candidates"]) == (0, 1)
    assert record["policy_calls"] == len(record["replans"]) == len(policy.calls)
    for event, call in zip(record["replans"], policy.calls, strict=True):
        assert (event["prompt"], event["original_prompt"], event["chosen"]) == (TASK, TASK, 0)
        assert event["noise_sha256"] == [call["noise"]] == event["proposal_sha256"][:0] + [event["base_noise_sha256"]]
        assert event["policy_inputs"] == call["inputs"] and event["untreated"]
        assert event["selected_proposal_sha256"] == event["proposal_sha256"][0]
    assert sum(event["executed_steps"] for event in record["replans"]) == record["controls"]


CHANGED = {
    "image": {"policy_inputs", "inputs_changed", "proposal_sha256", "selected_proposal_sha256", "executed_commands"},
    "state": {"policy_inputs", "inputs_changed", "proposal_sha256", "selected_proposal_sha256", "executed_commands"},
    "prompt": {"prompt", "proposal_sha256", "selected_proposal_sha256", "executed_commands"},
    "sample": {"noise_sha256", "proposal_sha256", "candidates", "chosen", "selected_proposal_sha256",
               "executed_commands"},
    "command": {"command_delta_linf", "executed_commands"},
}


@pytest.mark.parametrize("adapter", [Image(), State(), Prompt(), Sample(), Command()], ids=lambda a: a.name)
def test_each_hook_changes_only_what_it_owns(adapter):
    untreated, *_ = episode(Adapter())
    record, sim, policy = episode(adapter)
    first, reference = record["replans"][0], untreated["replans"][0]
    assert {key for key in first if first[key] != reference[key]} == CHANGED[adapter.name] | {"untreated"}
    assert record["reset"] == untreated["reset"]
    assert record["untreated_opening_proposal_sha256"] == untreated["untreated_opening_proposal_sha256"]
    assert record["changed_replans"] == len(record["replans"])
    # The receipt is what the simulator consumed.
    for event, issued in zip(record["replans"], sim.executed, strict=True):
        np.testing.assert_array_equal(event["executed_commands"], issued)
    nominal = np.asarray(reference["executed_commands"], dtype=np.float32)
    issued = np.asarray(first["executed_commands"], dtype=np.float32)
    column = dict(image=3, state=5, prompt=1, sample=2, command=2)[adapter.name]
    others = [c for c in range(7) if c != column]
    np.testing.assert_array_equal(issued[:, others], nominal[:, others])
    assert (issued[:, column] != nominal[:, column]).all()
    if adapter.name == "image":
        assert first["inputs_changed"] == ["observation/image"]
        assert first["policy_inputs"]["observation/image"] == array_digest(np.full((8, 8, 3), 200, np.uint8))
    if adapter.name == "state":
        assert first["inputs_changed"] == ["observation/state"]
    if adapter.name == "command":
        assert first["command_delta_linf"] == 1 + nominal[0, 2] and (issued[:, 2] == -1).all()
        assert [call["inputs"] for call in policy.calls[:1]] == [reference["policy_inputs"]]
    # One extra policy call recovers the untreated opening when the first query differs.
    extra = {"image": 1, "state": 1, "prompt": 1, "sample": 2 * len(record["replans"]), "command": 0}[adapter.name]
    assert record["policy_calls"] == len(policy.calls) == len(record["replans"]) + extra


def test_candidate_noise_and_chosen_index_are_recorded():
    adapter = Sample()
    record, sim, policy = episode(adapter)
    again, *_ = episode(Sample())
    assert consumed(record) == consumed(again)  # the candidate stream is seeded by the start
    assert record["candidates"] == 3
    for event in record["replans"]:
        assert (event["candidates"], event["chosen"]) == (3, 2)
        assert event["noise_sha256"][0] == event["base_noise_sha256"]
        assert len(set(event["noise_sha256"])) == len(set(event["proposal_sha256"])) == 3
        assert event["selected_proposal_sha256"] == event["proposal_sha256"][2]
    last = record["replans"][-1]
    assert last["noise_sha256"] == [array_digest(noise) for noise in adapter.drawn]
    assert last["proposal_sha256"] == [array_digest(proposal) for proposal in adapter.proposals]
    assert [call["noise"] for call in policy.calls[-3:]] == last["noise_sha256"]
    np.testing.assert_array_equal(last["executed_commands"], np.clip(adapter.proposals[2][:5], -1, 1))


def test_adapters_receive_public_inputs_only():
    seen, feedback = [], []

    class Spy(Adapter):
        name = "spy"

        def reset(self, support, first):
            self.support, self.first = support, first
            seen.append(first["observation"])

        def observation(self, raw):
            seen.append(raw)
            return raw

        def prompt(self, raw, original):
            seen.append(raw)
            for value in raw.values():
                value[:] = 9  # a private copy: neither the simulator nor the other hooks see this
            return original

        def choose(self, raw, proposals):
            seen.append(raw)
            proposals[0][:] = 5.
            return 0

        def command(self, raw, proposal):
            seen.append(raw)
            return proposal

        def observe(self, before, issued, after):
            seen.extend((before, after))
            feedback.append((before, issued, after))

    adapter = Spy()
    record, sim, _ = episode(adapter)
    allowed = {*PROPRIO, "agentview_image", "robot0_eye_in_hand_image"}
    assert seen and all(set(raw) == allowed - {"robot0_eef_quat"} for raw in seen)
    assert set(adapter.first) == {"observation", "language_instruction", "action_schema", "arm", "joint_count",
                                  "max_steps"}
    assert "physics" not in adapter.support and set(adapter.support["frames"][0]) <= set(PROPRIO)
    assert len(adapter.support["frames"]) == 4 and np.shape(adapter.support["commands"]) == (3, 5, 7)
    for (before, issued, after), executed in zip(feedback, sim.executed, strict=True):
        np.testing.assert_array_equal(issued, executed)
        np.testing.assert_allclose(after["robot0_eef_pos"] - before["robot0_eef_pos"],
                                   .01 * executed[:, :3].sum(axis=0))
    # The spy's writes into its copies reached neither the policy inputs nor the commands.
    untreated, *_ = episode(Adapter())
    assert consumed(record) == consumed(untreated) and record["changed_replans"] == 0


def test_support_extends_the_selector_calibration_record():
    sys.path.append(str(Path(__file__).resolve().parents[1] / "frozen-vla-selector-20261124-r1"))
    import rollout as selector

    mine = collect_support(Plant(), case("Kinova3", 11))
    reference = selector.collect_support(Plant(), case("Kinova3", 11))
    assert {key: mine[key] for key in reference} == reference
    assert harness.TASK == selector.TASK


@pytest.mark.parametrize(("hook", "value", "message"), [
    ("observation", None, "dictionary"), ("prompt", 3, "string"), ("noises", [], "nonempty"),
    ("noises", [np.zeros((2, 2))], "shape"), ("choose", 1, "outside"), ("choose", .5, "integer"),
    ("command", np.zeros((4, 7)), "at least five"), ("command", np.full((5, 7), np.nan), "finite"),
])
def test_invalid_hook_output_stops_the_episode(hook, value, message):
    adapter = Adapter()
    setattr(adapter, hook, lambda *_: value)
    with pytest.raises(ValueError, match=message):
        episode(adapter)


def test_calibration_must_be_independent_of_the_query():
    sim = Plant()
    query = case("Kinova3", 14)
    with pytest.raises(ValueError, match="independent calibration"):
        rollout(sim, Policy(), collect_support(sim, query), query, Adapter())
    with pytest.raises(ValueError, match="independent calibration"):
        rollout(sim, Policy(), collect_support(sim, case("Panda", 11)), query, Adapter())


LANE = '''
import numpy as np
from harness import Adapter

class Prompt(Adapter):
    name = "cue"
    def prompt(self, raw, original):
        return "open the top drawer"

class Push(Adapter):
    name = "push"
    def command(self, raw, proposal):
        proposal[:5, 2] = -1.
        return proposal

class Broken(Adapter):
    name = "broken"
    def choose(self, raw, proposals):
        return len(proposals)

ADAPTERS = [Prompt, Push(), Broken]
'''


@pytest.fixture
def lanes(tmp_path):
    for name, text in (("a", LANE), ("b", None), ("c", "raise RuntimeError('lane fault')"),
                       ("d", "from harness import Adapter\nADAPTERS = [Adapter]")):
        folder = tmp_path / f"frozen-shift-{name}-20261124-r1"
        folder.mkdir()
        if text is not None:
            (folder / "adapter.py").write_text(text)
    (tmp_path / "frozen-other-20261124-r1").mkdir()
    return tmp_path


def test_discovery_skips_missing_and_faulty_lanes_and_filters(lanes):
    factories, manifest = panel.discover(lanes)
    assert list(factories) == ["cue", "push", "broken"]
    assert [entry["adapters"] for entry in manifest] == [["cue", "push", "broken"], [], [], []]
    assert manifest[1]["skipped"] == "no adapter.py"
    assert "lane fault" in manifest[2]["skipped"] and "reserved" in manifest[3]["skipped"]
    assert list(panel.discover(lanes, ["push"])[0]) == ["push"]
    assert panel.discover(lanes, [])[0] == {}
    with pytest.raises(ValueError, match="Unknown adapters"):
        panel.discover(lanes, ["absent"])
    assert factories["cue"]() is not factories["cue"]() and factories["push"]() is factories["push"]()


def test_registered_starts_and_filters():
    assert panel.select_starts("development") == [("Kinova3", n) for n in (14, 15, 17, 36, 37, 38, 16, 31, 34, 35, 39)]
    assert panel.select_starts("locked") == [("Kinova3", n) for n in range(40, 48)]
    assert panel.select_starts("37,panda,37") == [("Kinova3", 37), *(("Panda", n) for n in range(4))]
    assert [panel.group_of(*start) for start in panel.select_starts("14,16,40,0")] == [
        "development_failures", "development_successes", "locked", "panda"]
    for text in ("11", "48", "Kinova3"):
        with pytest.raises(ValueError, match="Unregistered"):
            panel.select_starts(text)
    with pytest.raises(ValueError, match="Locked starts"):
        panel.main(["unused", "--starts", "locked", "--list"])


def test_panel_pairs_branches_and_counts_rescues_and_losses(lanes, tmp_path):
    factories, _ = panel.discover(lanes)
    out = tmp_path / "attempt"
    sim, policy = Plant(), Policy()
    report = panel.run(sim, policy, out, panel.select_starts("15,37,16,34,panda"), factories)
    assert report == json.loads((out / "summary.json").read_text())
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["Kinova3-support.json", "Panda-support.json", "summary.json", *(f"Kinova3-{n}.json" for n in (15, 37, 16, 34)),
         *(f"Panda-{n}.json" for n in range(4))])
    failures, successes = report["development_failures"], report["development_successes"]
    assert (failures["raw"]["successes"], failures["repeat"]["rescues"], failures["repeat"]["losses"]) == (0, 0, 0)
    assert (failures["cue"]["successes"], failures["cue"]["rescues"], failures["cue"]["losses"]) == (2, 2, 0)
    assert failures["cue"]["layout_interval95"] == [1., 1.] == failures["cue"]["repeat_layout_interval95"]
    assert (failures["push"]["rescues"], failures["push"]["losses"]) == (0, 0)
    assert (successes["raw"]["successes"], successes["cue"]["losses"]) == (2, 0)
    assert (successes["push"]["successes"], successes["push"]["losses"], successes["push"]["delta"]) == (0, 2, -1.)
    assert report["development"]["cue"]["total"] == 4 and report["development"]["cue"]["rescues"] == 2
    low, high = report["development"]["cue"]["layout_interval95"]
    assert 0 <= low < .5 < high <= 1
    assert (report["panda"]["raw"]["successes"], report["panda"]["cue"]["rescues"]) == (2, 2)
    # A lane fault is listed and leaves the denominator; it is not scored as a task failure.
    assert failures["broken"] == dict(successes=0, total=0, layouts={}, errors=[15, 37])
    cell = json.loads((out / "Kinova3-15.json").read_text())
    assert list(cell["branches"]) == ["raw", "repeat", "cue", "push"] and "outside 1 proposals" in cell["errors"]["broken"]
    assert [cell["branches"][name]["adapter"] for name in cell["branches"]] == list(cell["branches"])
    layout = failures["cue"]["layouts"]["15"]
    assert (layout["success"], layout["controls"], layout["first_divergence"], layout["changed_replans"]) == (True, 20, 0, 4)
    assert failures["repeat"]["layouts"]["15"]["first_divergence"] is None
    assert failures["raw"]["layouts"]["15"] == failures["repeat"]["layouts"]["15"]


def test_panel_rejects_unmatched_reset_or_incomplete_outcome(lanes):
    sim, policy = Plant(), Policy()
    support = collect_support(sim, case("Kinova3", 11))
    sim.corrupt = True
    with pytest.raises(ValueError, match="disagree on reset"):
        panel.run_start(sim, policy, support, case("Kinova3", 15), {})
    record, *_ = episode(Adapter())
    with pytest.raises(ValueError, match="opening proposal"):
        panel.verify(dict(record, untreated_opening_proposal_sha256="other"), record)
    with pytest.raises(ValueError, match="Incomplete native outcome"):
        panel.verify(dict(record, native_rewards=record["native_rewards"][:-1]), record)
