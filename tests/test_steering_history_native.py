"""Native-port reference, actual feedback and complete-episode recovery."""
import json
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import threading

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface
from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend, attempt_noise, case_key
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, file_sha256


class PhysicalReference:
    """Deterministic synthetic physics, with native-shaped observations/truth."""
    render_delta = 0

    def __init__(self, factory, cases, *, output, monitoring_root, journal, **kwargs):
        self.case, self.horizon = cases[0], cases[0].get("fixture_horizon", 100)
        self.output, self.monitoring_root = Path(output), Path(monitoring_root)
        self.output.mkdir(); self.monitoring_root.mkdir()
        self.key, self.journal = "reference", journal
        self.steps, self.success_steps, self.closed = 0, [], False
        self.dispatched_commands, self.completed_commands = [], []
        self.initial_evidence = dict(pairing=dict(matched=True, physical_state_exact=True,
            public_state_exact=True, images={key: dict(max_absolute_delta=self.render_delta)
                for key in ("main_images", "wrist_images", "extra_view_images")}))
        self.native = SimpleNamespace(current_raw_obs=dict(public_step=0))

    def _counts(self):
        return dict(controls=self.steps + 3, reset_controls=3)

    def observation(self):
        rgb = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
        return dict(states=np.ones((1, 25), np.float32), main_images=rgb.copy(),
                    wrist_images=rgb.copy(), extra_view_images=rgb[:, None].copy())

    def reset(self):
        token = self.journal.begin(128, phase="native_constructor_reset")
        self.journal.finish(token, 3)
        return self.observation()

    def step_chunk(self, commands):
        token = self.journal.begin(50, phase="policy_controls")
        self.dispatched_commands = commands.copy()
        self.completed_commands = commands.copy()
        self.steps += 50
        self.native.current_raw_obs = dict(public_step=self.steps)
        self.success_steps = [25]
        done = self.steps == self.horizon
        self.journal.finish(token, 50)
        if done:
            truth = dict(case=self.case, native_counts=self._counts(), simulator_success_once=True,
                         simulator_final_success=False, first_success_step=25)
            (self.monitoring_root / (self.key + ".json")).write_text(json.dumps(truth))
            videos = []
            for view in range(3):
                path = self.output / f"camera-{view}.fixture"
                path.write_bytes(b"synthetic, not an actual robot video")
                videos.append(dict(path=str(path), sha256=file_sha256(path)))
            (self.output / (self.key + ".json")).write_text(json.dumps(dict(completed=True,
                horizon=self.horizon, frame_clock=dict(frame_count=self.horizon + 1), videos=videos)))
        return self.observation(), done, {}

    def close(self):
        self.closed, self.native = True, None


def backend(root, *, spec, environment_type=PhysicalReference, maximum_rgb_delta=1, actual_attempts=5,
            fixed_case_order=False):
    from test_interface_ppo import batch_native
    native, _ = batch_native()
    native.model.layers = 18
    journal = ActionJournal(root / "accounting", rank=0, world=1, attempt="fixture", maximum_actions=None)
    result = NativeAttemptBackend(output=root / "episodes", canonical_root=root / "canonical",
        motor=FrozenRoboCasaInterface(native), spec=spec, native_factory=lambda _: None, journal=journal,
        writer_factory=lambda _: None, binding=dict(source="fixture", candidate="frozen-fixture",
            render_contract="diagnostic_only" if maximum_rgb_delta is None else "bounded"),
        environment_type=environment_type, maximum_rgb_delta=maximum_rgb_delta, actual_attempts=actual_attempts,
        fixed_case_order=fixed_case_order)
    return result, native.model, journal


@pytest.mark.parametrize("kind", ["velocity", "kv_all_layers"])
def test_native_port_full_attempts_truth_and_recovery_keep_actual_draws(tmp_path, kind):
    spec = (InterfaceSpec(kind=kind, steering_dim=32, fraction=.05) if kind == "velocity"
            else InterfaceSpec(kind=kind, steering_dim=1152, kv_layers=18, fraction=.04))
    policy = HistoryNoisePolicy(HistoryConfig(20, spec.steering_dim, width=8, heads=2,
                                            layers=1, context_length=12))
    case = dict(task="OpenDrawer", native_seed=17)
    original, model, journal = backend(tmp_path / "original", spec=spec)
    results = original.run_case(policy, case, horizon=100)
    assert len(original.references) == 5 and model.prefixes == model.draws == 10
    assert journal.charged() == 5 * 103  # Actual reset plus full physical suffix.
    assert all(row["success_once"] and not row["success_at_end"] for row in results)
    for previous, following in zip(results, results[1:]):
        assert previous["memory_exit"] == following["memory_entry"]
    recovered, model, journal = backend(tmp_path / "recovery", spec=spec)
    resumed = recovered.run_case(policy, case, horizon=100, prefix_references=original.references[:3])
    assert len(recovered.references) == 2 and model.prefixes == model.draws == 4
    assert journal.charged() == 2 * 103
    for old, new in zip(results, resumed, strict=True):
        assert old["memory_exit"] == new["memory_exit"]
        for left, right in zip(old["records"], new["records"], strict=True):
            np.testing.assert_array_equal(left["control"], right["control"])
            np.testing.assert_array_equal(left["native_noise"], right["native_noise"])
    # Hash rejection happens before another native environment or control.
    with Path(original.references[0]["path"]).open("ab") as stream:
        stream.write(b"changed")
    invalid, model, journal = backend(tmp_path / "invalid", spec=spec)
    with pytest.raises(ValueError, match="checksum"):
        invalid.run_case(policy, case, horizon=100, prefix_references=original.references[:1])
    assert journal.charged() == 0 and model.draws == 0


def test_unqualified_pixels_preserve_reset_cost_and_do_not_dispatch(tmp_path):
    class Unqualified(PhysicalReference):
        render_delta = 2
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    policy = HistoryNoisePolicy(HistoryConfig(20, 32, width=8, heads=2, layers=1, context_length=12))
    runner, _, journal = backend(tmp_path, spec=spec, environment_type=Unqualified)
    with pytest.raises(ValueError, match="render contract"):
        runner.run_case(policy, dict(task="OpenDrawer", native_seed=17), horizon=100)
    assert journal.charged() == 3
    assert len(list((tmp_path / "episodes").glob("*/attempt-1/initial.pt"))) == 1
    assert len(list((tmp_path / "episodes").glob("*/failure-*.pt"))) == 1
    assert not runner.references


def test_native_attempt_noise_is_case_stable_and_attempt_specific():
    case = dict(task="OpenDrawer", native_seed=17)
    first, meta = attempt_noise(case, 1, 0)
    second, repeated = attempt_noise(dict(native_seed=17, task="OpenDrawer"), 1, 0)
    np.testing.assert_array_equal(first, second)
    assert meta == repeated and first.shape == (50, 32) and first.dtype == np.float32
    assert not np.array_equal(first, attempt_noise(case, 2, 0)[0])
    assert not np.array_equal(first, attempt_noise(case, 1, 1)[0])


def test_sixth_query_uses_five_completed_episodes_and_resumes_exactly(tmp_path):
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    policy = HistoryNoisePolicy(HistoryConfig(20, 32, width=8, heads=2, layers=1, context_length=12))
    case = dict(task="OpenDrawer", native_seed=17)
    original, model, journal = backend(tmp_path / "original", spec=spec, actual_attempts=6)
    results = original.run_case(policy, case, horizon=100, attempts=6)
    assert len(results) == len(original.references) == 6
    assert results[5]["memory_entry"]["length"] == results[4]["memory_exit"]["length"]
    assert results[5]["memory_tokens"] == 12
    assert journal.charged() == 6 * 103 and model.draws == 12
    assert original.binding["native_attempt_protocol"]["actual_attempts"] == 6
    recovered, model, journal = backend(tmp_path / "recovered", spec=spec, actual_attempts=6)
    resumed = recovered.run_case(policy, case, horizon=100, attempts=6,
                                 prefix_references=original.references[:5])
    assert journal.charged() == 103 and len(recovered.references) == 1
    assert resumed[5]["memory_entry"] == results[5]["memory_entry"]
    for old, new in zip(results[5]["records"], resumed[5]["records"], strict=True):
        np.testing.assert_array_equal(old["control"], new["control"])
        np.testing.assert_array_equal(old["native_noise"], new["native_noise"])
    default, _, journal = backend(tmp_path / "default", spec=spec)
    with pytest.raises(ValueError, match="bound native attempt protocol"):
        default.run_case(policy, case, horizon=100, attempts=6)
    assert journal.charged() == 0


def test_fixed_native_rows_survive_six_query_episodes_and_unequal_cohorts_fail(tmp_path):
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    policy = HistoryNoisePolicy(HistoryConfig(20, 32, width=8, heads=2, layers=1, context_length=12))
    cases = [dict(case=dict(task="OpenDrawer", native_seed=seed), horizon=100) for seed in (17, 18)]
    runner, _, journal = backend(tmp_path / "fixed", spec=spec, actual_attempts=6, fixed_case_order=True)
    results = runner.run_bank(policy, cases, lanes=2, attempts=6)
    assert len(results) == 12 and runner.last_batch_sizes == [2] * 12
    assert journal.charged() == 12 * 103
    expected_owners = [case_key(item["case"]) for item in cases]
    import gzip
    import pickle
    for reference in runner.references:
        saved = torch.load(reference["path"], weights_only=False)
        for item in saved["chunks"]:
            with gzip.open(item["path"], "rb") as stream:
                row = pickle.load(stream)
            assert row["execution_provenance"]["case_owners"] == expected_owners
            assert row["execution_provenance"]["row"] == expected_owners.index(case_key(saved["case"]))
    unequal, _, journal = backend(tmp_path / "unequal", spec=spec, fixed_case_order=True)
    cases[1]["horizon"] = 50
    with pytest.raises(ValueError, match="equal-horizon wave"):
        unequal.run_bank(policy, cases, lanes=2)
    assert journal.charged() == 0


def test_explicit_diagnostic_render_contract_keeps_original_policy_pixels(tmp_path):
    class Diagnostic(PhysicalReference):
        render_delta = 2
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    policy = HistoryNoisePolicy(HistoryConfig(20, 32, width=8, heads=2, layers=1, context_length=12))
    runner, _, journal = backend(tmp_path, spec=spec, environment_type=Diagnostic, maximum_rgb_delta=None)
    results = runner.run_case(policy, dict(task="OpenDrawer", native_seed=17), horizon=100, attempts=1)
    assert journal.charged() == 103
    np.testing.assert_array_equal(results[0]["records"][0]["observation"]["main_images"],
                                 np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3))
    assert runner.binding["native_attempt_protocol"]["maximum_rgb_delta"] is None


@pytest.mark.parametrize("kind", ["velocity", "kv_all_layers"])
def test_batched_motor_preserves_serial_case_memory_feedback_and_recovery(tmp_path, kind):
    torch.manual_seed(712)
    spec = (InterfaceSpec(kind=kind, steering_dim=32, fraction=.05) if kind == "velocity"
            else InterfaceSpec(kind=kind, steering_dim=1152, kv_layers=18, fraction=.04))
    policy = HistoryNoisePolicy(HistoryConfig(20, spec.steering_dim, width=8, heads=2,
                                            layers=1, context_length=32))
    cases = [dict(case=dict(task="OpenDrawer", native_seed=17, fixture_horizon=horizon), horizon=horizon)
             for horizon in (100, 150, 50)]
    serial, _, _ = backend(tmp_path / "serial", spec=spec)
    expected = []
    for item in cases:
        expected.extend(serial.run_case(policy, item["case"], horizon=item["horizon"]))
    parallel, model, journal = backend(tmp_path / "parallel", spec=spec)
    actual = parallel.run_bank(deepcopy(policy), cases, lanes=2)
    assert len(actual) == len(parallel.references) == 15
    assert model.prefixes == model.draws <= 35
    assert any(size > 1 for size in parallel.last_batch_sizes)
    assert journal.charged() == sum(5 * (item["horizon"] + 3) for item in cases)
    assert not parallel.environments  # Closed cameras are not retained across the bank.
    references = {}
    for reference in parallel.references:
        saved = torch.load(reference["path"], weights_only=False)
        references.setdefault(case_key(saved["case"]), {})[saved["attempt"]] = reference
    for item in cases:
        key = case_key(item["case"])
        replayed = parallel.recover_prefix([references[key][i] for i in range(1, 6)],
                                           case=item["case"], horizon=item["horizon"], policy=policy)
        for result in replayed:
            original = next(row for row in expected if row["case"] == result["case"] and row["attempt"] == result["attempt"])
            assert original["memory_entry"] == result["memory_entry"]
            assert original["memory_exit"] == result["memory_exit"]
            for before, after in zip(original["records"], result["records"], strict=True):
                np.testing.assert_array_equal(before["control"], after["control"])
                np.testing.assert_array_equal(before["native_noise"], after["native_noise"])
                np.testing.assert_array_equal(after["executed_commands"], after["dispatched_commands"])
                assert after["execution_provenance"]["case_owners"][after["execution_provenance"]["row"]] == key
    recovery, model, journal = backend(tmp_path / "recovery", spec=spec)
    prefixes = {case_key(cases[0]["case"]): [references[case_key(cases[0]["case"])][i] for i in range(1, 4)],
                case_key(cases[1]["case"]): [references[case_key(cases[1]["case"])][i] for i in range(1, 6)]}
    recovered = recovery.run_bank(policy, cases, lanes=2, prefix_references=prefixes)
    assert len(recovered) == 15 and len(recovery.references) == 7
    assert journal.charged() == 2 * 103 + 5 * 53
    assert model.draws > 0 and all("records" not in result for result in recovered)


def test_batched_failure_preserves_completed_sibling_and_stops_new_cases(tmp_path):
    completed = threading.Event()

    class Failure(PhysicalReference):
        def step_chunk(self, commands):
            result = super().step_chunk(commands)
            if self.case["native_seed"] == 2:
                assert completed.wait(timeout=10)
                raise RuntimeError("failed owned physical lane")
            return result

    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    policy = HistoryNoisePolicy(HistoryConfig(20, 32, width=8, heads=2, layers=1, context_length=16))
    runner, _, journal = backend(tmp_path, spec=spec, environment_type=Failure)
    original = runner._completed

    def completed_attempt(result, controller):
        original(result, controller)
        completed.set()

    runner._completed = completed_attempt
    cases = [dict(case=dict(task="OpenDrawer", native_seed=i, fixture_horizon=h), horizon=h)
             for i, h in ((1, 50), (2, 100), (3, 50))]
    with pytest.raises(RuntimeError, match="failed owned physical lane"):
        runner.run_bank(policy, cases, lanes=2)
    saved = [torch.load(ref["path"], weights_only=False) for ref in runner.references]
    assert any(row["case"]["native_seed"] == 1 and row["attempt"] == 1 for row in saved)
    assert all(row["case"]["native_seed"] != 2 for row in saved)
    assert not (tmp_path / "episodes" / ("case-" + case_key(cases[2]["case"]))).exists()
    assert journal.charged() >= 106
    assert all(env.closed for env in runner.environments.values())
