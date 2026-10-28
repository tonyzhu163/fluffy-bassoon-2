"""Public recording cannot export private labels or confuse paired probe identities."""

from types import SimpleNamespace

import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_crossed_probe import paired_probe_validity, public_interface_sha256
from fluffy_bassoon.meta_rl.pi05_fresh_probe import ProbeBoundary, run_live_attempt


class RecordedPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))

    def sample_chunk(self, motor, observation, feedback, memory, **kwargs):
        key = torch.ones(2, 3, 1, 4)
        return torch.ones(2, 8, 7), {"state": torch.zeros(2, 1)}, dict(
            recurrence_updates=1, num_steps=4, kv_key=key, kv_value=key+1,
            prefix_mask=torch.ones(2, 3, dtype=torch.bool), feedback=feedback,
            target_actions=torch.ones(2, 8, 7), private_theta=torch.tensor([-30., 30.]),
            bridge_features=torch.full((2, 8), 42.))


class RecordedMotor:
    def env_obs_to_observation(self, obs):
        return SimpleNamespace(state=torch.zeros(2, 32))

    def decode_actions(self, actions, state):
        return actions


class RecordedEnv:
    device = "cpu"

    def __init__(self):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.tensor([0, 1])))

    def reset(self, **kwargs):
        return {}, {}

    def chunk_step(self, actions):
        return [{}]*5, None, None, None, [dict(success=torch.tensor([True, False]))]*5


def test_attempt_recording_exports_only_public_inputs_and_keeps_pending_outcome():
    public = []
    state = ProbeBoundary(None, torch.zeros(2, 7), torch.zeros(2), 0, ("a", "b"))
    result = run_live_attempt(RecordedPolicy(), RecordedMotor(), RecordedEnv(),
        [dict(object=0, episode_id=0), dict(object=1, episode_id=1)], [-30., 30.], 3, state,
        generator=torch.Generator().manual_seed(10), preprocess=lambda obs: obs,
        prepare_actions=lambda actions: actions[:, :5], on_public_chunk=public.append)
    assert len(public) == 16 and result["counts"]["native_prefix_queries"] == 32
    assert all(set(record) == {"kv_key", "kv_value", "prefix_mask", "feedback"} for record in public)
    assert all(value.device.type == "cpu" and not value.requires_grad for record in public for value in record.values())
    assert result["state"].attempt_index == 1
    torch.testing.assert_close(result["state"].pending_reward, torch.tensor([1., 0.]))
    assert all(record["feedback"][:, 7].eq(0).all() for record in public)


def test_paired_probe_validity_rejects_different_scene_or_noise_stream():
    same = [dict(first_prefix_sha256="a", final_noise_state_sha256="b") for _ in range(3)]
    assert paired_probe_validity(same)["first_prefix_identical"]
    assert paired_probe_validity(same)["native_noise_streams_identical"]
    other_scene = [*same, dict(first_prefix_sha256="c", final_noise_state_sha256="b")]
    assert not paired_probe_validity(other_scene)["first_prefix_identical"]
    other_noise = [*same, dict(first_prefix_sha256="a", final_noise_state_sha256="d")]
    assert not paired_probe_validity(other_noise)["native_noise_streams_identical"]
    assert not paired_probe_validity([])["first_prefix_identical"]


def test_public_stream_identity_allows_different_readers_but_rejects_changed_motor():
    contract = dict(task="put-on-plate", motor_sha256="a"*64, norm_stats_sha256="b"*64,
        rlinf_sources={"motor": "c"*64}, feedback="public", reward="success_once",
        attempts=4, episode_steps=80, action_chunk=5, denoise_steps=4, core="persistent_ctm")
    assert public_interface_sha256(contract) == public_interface_sha256(dict(contract, core="transformer_xl"))
    assert public_interface_sha256(contract) != public_interface_sha256(dict(contract, motor_sha256="d"*64))
    assert public_interface_sha256(contract) != public_interface_sha256(dict(contract, attempts=8))
