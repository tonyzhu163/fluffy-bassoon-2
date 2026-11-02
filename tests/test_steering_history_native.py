"""Native-port reference, actual feedback and complete-episode recovery."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface
from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend, attempt_noise
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy, file_sha256


class PhysicalReference:
    """Deterministic synthetic physics, with native-shaped observations/truth."""
    render_delta = 0

    def __init__(self, factory, cases, *, output, monitoring_root, journal, **kwargs):
        self.case, self.horizon = cases[0], 100
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
        return dict(states=np.ones((1, 25), np.float32))

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


def backend(root, *, spec, environment_type=PhysicalReference):
    from test_interface_ppo import batch_native
    native, _ = batch_native()
    native.model.layers = 18
    journal = ActionJournal(root / "accounting", rank=0, world=1, attempt="fixture", maximum_actions=None)
    result = NativeAttemptBackend(output=root / "episodes", canonical_root=root / "canonical",
        motor=FrozenRoboCasaInterface(native), spec=spec, native_factory=lambda _: None, journal=journal,
        writer_factory=lambda _: None, binding=dict(source="fixture", candidate="frozen-fixture"),
        environment_type=environment_type)
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
