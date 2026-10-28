"""Shared public support streams reconstruct each reader without state transfer.

Run on allocated Linux CPU resources; no frozen motor or simulator is loaded.
"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.evaluation_stages import sha256
from fluffy_bassoon.meta_rl.pi05_fresh_probe import ProbeBoundary, run_live_attempt
from fluffy_bassoon.meta_rl.pi05_public_history import (
    PUBLIC_RECORD_KEYS, PublicHistoryRecorder, load_public_history,
    replay_public_history, save_public_history, validate_public_history,
)
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge, public_chunk_feedback


MOTOR, INTERFACE, SOURCE = "a" * 64, "b" * 64, "c" * 40
IDS = ("support-batch-0/lane-0", "support-batch-0/lane-1")


def reader(core):
    return StochasticPi05BridgePolicy(Pi05FeatureBridge(core=core, width=8, ticks=1,
        memory_length=8, residual_scale=1.)).eval()


@torch.no_grad()
def collect(collector):
    """Exercise the exact proposed public collector callbacks, not saved memory."""
    rng = torch.Generator().manual_seed(51)
    recorder = PublicHistoryRecorder(motor_sha256=MOTOR, interface_sha256=INTERFACE, history_ids=IDS)
    memory, previous, pending = None, torch.zeros(2, 7), torch.zeros(2)
    for attempt in range(3):
        for chunk in range(16):
            feedback = torch.cat([public_chunk_feedback(previous[lane:lane+1],
                previous_reward=float(pending[lane]), attempt_end=attempt > 0 and chunk == 0,
                attempt_index=attempt, chunk_index=chunk, attempts=4, chunks=16) for lane in range(2)])
            key = torch.randn(2, 3, 1, 256, generator=rng)
            value = torch.randn(2, 3, 1, 256, generator=rng)
            mask = torch.tensor([[True, True, False], [True, True, True]])
            record = dict(kv_key=key, kv_value=value, prefix_mask=mask, feedback=feedback)
            _, memory = collector.bridge.read_context(((key, value),), mask, feedback, memory)
            recorder.on_chunk(record)
            # Simulated executed requested actions and public terminal reward.
            previous = torch.randn(2, 7, generator=rng)
            pending = torch.tensor([1., 0.]) if chunk == 15 else torch.zeros(2)
        recorder.on_boundary(ProbeBoundary(memory, previous, pending, attempt + 1, IDS,
                                           attempt_end=True, noise_state=rng.get_state()))
    history = recorder.finish()
    return history, {key: value.clone() for key, value in memory.items()}


def validate(history, **kwargs):
    return validate_public_history(history, expected_motor_sha256=kwargs.get("motor", MOTOR),
                                   expected_interface_sha256=kwargs.get("interface", INTERFACE))


def replay(policy, history):
    return replay_public_history(policy, history, expected_motor_sha256=MOTOR,
                                  expected_interface_sha256=INTERFACE)


@pytest.mark.parametrize("core", ["persistent_ctm", "transformer_xl"])
def test_own_collector_memory_is_exactly_reconstructed_over_all_48_chunks(core):
    torch.manual_seed(19)
    policy = reader(core)
    history, online = collect(policy)
    before = deepcopy(policy.state_dict())
    receipt = validate(history)
    assert receipt["lanes"] == 2 and receipt["policy_chunks"] == 96
    assert receipt["simulator_steps"] == 480
    state = replay(policy, history)
    assert state.attempt_index == 3 and state.attempts == 4 and state.attempt_end
    assert state.history_ids == IDS and state.noise_state is None
    torch.testing.assert_close(state.memory, online, rtol=0, atol=0)
    assert all(not value.requires_grad for value in state.memory.values())
    torch.testing.assert_close(policy.state_dict(), before, rtol=0, atol=0)
    assert set(history["records"][0]) == set(PUBLIC_RECORD_KEYS)
    assert "memory" not in history["boundaries"][-1]
    assert "noise_state" not in history["boundaries"][-1]
    # Returned probe state must not alias the immutable public artifact.
    state.previous_command.zero_()
    state.pending_reward.zero_()
    assert history["boundaries"][-1]["pending_reward"].tolist() == [1., 0.]
    validate(history)


def test_cross_core_replay_uses_reader_state_schema_and_its_own_chronological_computation():
    torch.manual_seed(81)
    history, ctm_online = collect(reader("persistent_ctm"))
    xl = reader("transformer_xl")
    expected = None
    with torch.no_grad():
        for record in history["records"]:
            _, expected = xl.bridge.read_context(((record["kv_key"], record["kv_value"]),),
                record["prefix_mask"], record["feedback"], expected)
    state = replay(xl, history)
    assert set(state.memory) == {"memory", "length"} and set(state.memory) != set(ctm_online)
    torch.testing.assert_close(state.memory, expected, rtol=0, atol=0)
    assert state.memory["length"].tolist() == [8, 8]
    # The bare bridge is also a reader; its schema and transition path agree.
    torch.testing.assert_close(replay(xl.bridge, history).memory, expected, rtol=0, atol=0)


@pytest.mark.parametrize("private_key", ["theta_degrees", "labels", "target_actions", "scenes", "memory"])
def test_private_or_collector_fields_are_rejected_instead_of_silently_projected(private_key):
    recorder = PublicHistoryRecorder(motor_sha256=MOTOR, interface_sha256=INTERFACE, history_ids=IDS)
    record = dict(kv_key=torch.zeros(2, 1, 1, 256), kv_value=torch.zeros(2, 1, 1, 256),
                  prefix_mask=torch.ones(2, 1, dtype=torch.bool), feedback=torch.zeros(2, 11))
    record[private_key] = torch.zeros(2)
    with pytest.raises(ValueError, match="only KV"):
        recorder.on_chunk(record)


@pytest.mark.parametrize("change", ["clock", "boundary", "reward", "nan", "kv_shape", "extra_top", "hash"])
def test_complete_history_validation_rejects_corrupted_public_contract(change):
    history, _ = collect(reader("transformer_xl"))
    if change == "clock":
        history["records"][22]["feedback"][:, 9] = .75
    elif change == "boundary":
        history["boundaries"][0]["previous_command"].add_(1)
    elif change == "reward":
        history["records"][5]["feedback"][:, 7] = 1
    elif change == "nan":
        history["records"][0]["kv_key"].fill_(float("nan"))
    elif change == "kv_shape":
        history["records"][0]["kv_value"] = torch.zeros(2, 3, 2, 128)
    elif change == "extra_top":
        history["theta_degrees"] = [-30., 30.]
    else:
        history["content_sha256"] = "d" * 64
    with pytest.raises(ValueError):
        validate(history)


def test_callbacks_copy_inputs_enforce_boundaries_and_reject_incomplete_history():
    recorder = PublicHistoryRecorder(motor_sha256=MOTOR, interface_sha256=INTERFACE, history_ids=IDS)
    with pytest.raises(ValueError, match="48 chunks"):
        recorder.finish()
    record = dict(kv_key=torch.ones(2, 1, 1, 256), kv_value=torch.ones(2, 1, 1, 256),
        prefix_mask=torch.ones(2, 1, dtype=torch.bool), feedback=torch.zeros(2, 11))
    recorder.on_chunk(record)
    record["kv_key"].zero_()
    assert recorder._history["records"][0]["kv_key"].eq(1).all()
    for chunk in range(1, 16):
        record["feedback"][:, 10] = chunk / 16
        recorder.on_chunk(record)
    with pytest.raises(ValueError, match="boundary before"):
        recorder.on_chunk(record)


@pytest.mark.parametrize("bad", ["", "a" * 40, "A" * 64, "not-a-digest"])
def test_constructor_requires_declared_motor_and_interface_sha256(bad):
    with pytest.raises(ValueError):
        PublicHistoryRecorder(motor_sha256=bad, interface_sha256=INTERFACE, history_ids=IDS)
    with pytest.raises(ValueError):
        PublicHistoryRecorder(motor_sha256=MOTOR, interface_sha256=bad, history_ids=IDS)


def test_artifact_manifest_and_history_are_externally_pinned_and_never_overwritten(tmp_path):
    history, _ = collect(reader("transformer_xl"))
    path = tmp_path / "collector-base"
    receipt = save_public_history(path, history, source_commit=SOURCE, collector_id="base")
    loaded = load_public_history(path, receipt["manifest_sha256"],
        expected_motor_sha256=MOTOR, expected_interface_sha256=INTERFACE)
    assert loaded["content_sha256"] == history["content_sha256"]
    assert loaded["history_ids"] == history["history_ids"]
    for actual, expected in zip(loaded["records"], history["records"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert receipt["manifest"]["history_sha256"] == sha256(path / "history.pt")
    with pytest.raises(FileExistsError):
        save_public_history(path, history, source_commit=SOURCE, collector_id="base")
    with pytest.raises(ValueError, match="motor/interface"):
        validate(history, interface="d" * 64)
    with (path / "history.pt").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_public_history(path, receipt["manifest_sha256"],
            expected_motor_sha256=MOTOR, expected_interface_sha256=INTERFACE)


class FeedbackPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))
        self.feedbacks = []

    def sample_chunk(self, motor, observation, feedback, memory, **kwargs):
        self.feedbacks.append(feedback.clone())
        return torch.zeros(2, 8, 32), memory, dict(recurrence_updates=1, num_steps=4)


class ProbeEnv:
    device = "cpu"
    env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.zeros(2, dtype=torch.long)))

    def reset(self, **kwargs):
        return {"state": torch.zeros(2, 4)}, {}

    def chunk_step(self, command):
        return [{"state": torch.zeros(2, 4)}] * 5, None, None, None, [dict(success=torch.zeros(2))] * 5


def test_replayed_pending_terminal_reward_is_consumed_once_at_fresh_fourth_attempt():
    history, _ = collect(reader("transformer_xl"))
    state = replay(reader("persistent_ctm"), history)
    policy = FeedbackPolicy()
    motor = SimpleNamespace(env_obs_to_observation=lambda obs: SimpleNamespace(state=obs["state"]),
                            decode_actions=lambda actions, state: actions[:, :5, :7])
    result = run_live_attempt(policy, motor, ProbeEnv(),
        [dict(object=0, episode_id=100), dict(object=0, episode_id=101)], [-30., 30.], 1, state,
        generator=torch.Generator().manual_seed(9), preprocess=lambda obs: obs, prepare_actions=lambda actions: actions)
    first, second = policy.feedbacks[:2]
    torch.testing.assert_close(first[:, :7], history["boundaries"][-1]["previous_command"])
    assert first[:, 7].tolist() == [1., 0.] and first[:, 8].eq(1).all()
    assert first[:, 9].eq(.75).all() and first[:, 10].eq(0).all()
    assert second[:, 7:9].eq(0).all() and second[:, 9].eq(.75).all()
    assert second[:, 10].eq(1 / 16).all() and result["state"].attempt_index == 4
