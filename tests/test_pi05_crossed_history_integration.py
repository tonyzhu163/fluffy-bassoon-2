"""Actual public hook → immutable history → two reader cores → paired probes.

This synthetic CPU integration test belongs in allocated Linux qualification.
It uses the native 512-wide final-layer KV contract without loading RLinf.
"""

from types import SimpleNamespace

import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_crossed_probe import paired_probe_validity
from fluffy_bassoon.meta_rl.pi05_fresh_probe import ProbeBoundary, clone_boundary, run_live_attempt
from fluffy_bassoon.meta_rl.pi05_public_history import (
    PublicHistoryRecorder, load_public_history, replay_public_history, save_public_history,
)
from fluffy_bassoon.rl.pi05_bridge_policy import StochasticPi05BridgePolicy
from fluffy_bassoon.rl.pi05_feature_bridge import Pi05FeatureBridge


MOTOR, INTERFACE = "a" * 64, "b" * 64


class NativeShapeMotor(nn.Module):
    num_steps, action_horizon, action_dim = 4, 8, 32

    def env_obs_to_observation(self, obs):
        return SimpleNamespace(state=obs["state"])

    def build_prefix_cache(self, observation):
        key = observation.state[:, None, None].expand(-1, 3, 1, -1)
        return None, torch.ones(key.shape[:2], dtype=torch.bool), ((key, key + .1),)

    def run_suffix(self, observation, noisy, timestep, cache, mask):
        return noisy.repeat(1, 1, 32)

    def velocity_from_suffix(self, suffix):
        return suffix[..., :32] * .1

    def decode_actions(self, actions, state):
        return actions[:, :5, :7]


class NativeShapeEnv:
    device = "cpu"

    def __init__(self):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.tensor([0, 1])))

    def reset(self, *, seed, options):
        # Only the returned public observation reaches the frozen feature motor.
        self.state = options["episode_id"].float()[:, None].expand(-1, 256).clone() / 100
        return {"state": self.state.clone()}, {}

    def chunk_step(self, command):
        self.state = self.state + command.mean((1, 2))[:, None] * .01
        return ([{"state": self.state.clone()}] * 5, None, None, None,
                [dict(success=torch.tensor([True, False]))] * 5)


def policy(kind):
    return StochasticPi05BridgePolicy(Pi05FeatureBridge(core=kind, width=8, ticks=1,
        memory_length=8, residual_scale=1.)).eval()


def run(policy, motor, env, scenes, state, seed, callback=None):
    return run_live_attempt(policy, motor, env, scenes, [-30., 30.], 17, state,
        generator=torch.Generator().manual_seed(seed), preprocess=lambda obs: obs,
        prepare_actions=lambda decoded: decoded, on_public_chunk=callback)


def test_live_hook_saved_stream_cross_core_replay_and_fresh_probe_pairing(tmp_path):
    torch.manual_seed(94)
    collector, xl = policy("persistent_ctm"), policy("transformer_xl")
    # Nonzero collector residual means exercise an actually modified behavior
    # stream; neither the alternate reader nor the recorder receives its memory.
    with torch.no_grad():
        collector.bridge.residual_output.weight.normal_(std=.01)
    motor, env = NativeShapeMotor().eval().requires_grad_(False), NativeShapeEnv()
    ids = ("shared-support/lane-0", "shared-support/lane-1")
    recorder = PublicHistoryRecorder(motor_sha256=MOTOR, interface_sha256=INTERFACE, history_ids=ids)
    state = ProbeBoundary(None, torch.zeros(2, 7), torch.zeros(2), 0, ids)
    for attempt in range(3):
        scenes = [dict(object=lane, episode_id=10 * (attempt + 1) + lane) for lane in range(2)]
        result = run(collector, motor, env, scenes, state, 100 + attempt, recorder.on_chunk)
        state = result["state"]
        recorder.on_boundary(state)
    history = recorder.finish()
    assert len(history["records"]) == 48
    assert all(record["kv_key"].shape == (2, 3, 1, 256) for record in history["records"])
    receipt = save_public_history(tmp_path / "stream", history, source_commit="c" * 40,
        collector_id="learned-ctm", collector_checkpoint_sha256="d" * 64)
    loaded = load_public_history(tmp_path / "stream", receipt["manifest_sha256"],
        expected_motor_sha256=MOTOR, expected_interface_sha256=INTERFACE)
    ctm_boundary = replay_public_history(collector, loaded,
        expected_motor_sha256=MOTOR, expected_interface_sha256=INTERFACE)
    xl_boundary = replay_public_history(xl, loaded,
        expected_motor_sha256=MOTOR, expected_interface_sha256=INTERFACE)
    torch.testing.assert_close(ctm_boundary.memory, state.memory, rtol=0, atol=0)
    assert set(xl_boundary.memory) == {"memory", "length"}
    assert set(ctm_boundary.memory) != set(xl_boundary.memory)
    assert xl_boundary.memory["length"].tolist() == [8, 8]
    branches = []
    scenes = [dict(object=lane, episode_id=100 + lane) for lane in range(2)]
    for reader, boundary in ((collector, ctm_boundary), (xl, xl_boundary)):
        for reset in (False, True):
            result = run(reader, motor, env, scenes, clone_boundary(boundary, reset_memory=reset), 900)
            first = torch.tensor(result["first_feedback"])
            torch.testing.assert_close(first[:, :7], state.previous_command)
            assert first[:, 7].tolist() == [1., 0.]
            assert first[:, 8].eq(1).all() and first[:, 9].eq(.75).all()
            assert result["state"].attempt_index == 4 and result["counts"]["simulator_steps"] == 160
            branches.append(result)
    pairing = paired_probe_validity(branches)
    assert pairing["branches"] == 4
    assert pairing["first_prefix_identical"] and pairing["native_noise_streams_identical"]
