"""Numerical sampler/feature checks; execute only on allocated Linux."""
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.robocasa_residual import FrozenRoboCasaMotor


class NativeShapeModel(torch.nn.Module):
    action_horizon, action_dim, action_env_dim, num_steps = 50, 32, 12, 10
    device = torch.device('cpu')

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls = 0

    def build_prefix_cache(self, observation):
        batch = len(observation['states'])
        return (torch.tensor([[[1., 3.], [3., 1.], [float('nan'), float('nan')]]]).expand(batch, -1, -1),
                torch.tensor([[True, True, False]]).expand(batch, -1), ())

    def predict_action_batch(self, observation, mode, noise=None, rng=None):
        assert mode == 'eval' and not torch.is_grad_enabled()
        assert 'success' not in observation
        self.calls += 1
        self.build_prefix_cache(observation)
        if noise is None:
            noise = torch.randn((len(observation['states']), 50, 32), generator=rng)
        return noise[..., :12], {}


def motor():
    stats = {key: dict(mean=[0.] * 32, std=[2.] * 32) for key in ('state', 'actions')}
    return FrozenRoboCasaMotor(NativeShapeModel(), stats)


def test_batch_retains_each_serial_noise_draw_and_features():
    m = motor()
    observation = dict(states=np.arange(100, dtype=np.float32).reshape(4, 25), success=True)
    chunks, features, seeds = m.infer_batch(observation, noise_seeds=[11, 22, 33, 44])
    assert m.model.calls == 1
    for lane, seed in enumerate(seeds):
        chunk, feature, _ = m.infer(dict(states=observation['states'][lane:lane+1]), noise_seed=seed)
        np.testing.assert_array_equal(chunks[lane], chunk)
        np.testing.assert_array_equal(features[lane], feature)


def test_batch_lane_reordering_preserves_results_without_global_rng_consumption():
    m = motor()
    observation = dict(states=np.zeros((2, 25), np.float32))
    state = torch.get_rng_state().clone()
    first = m.infer_batch(observation, noise_seeds=[1, 2])
    second = m.infer_batch(observation, noise_seeds=[2, 1])
    np.testing.assert_array_equal(first[0], second[0][::-1])
    assert torch.equal(state, torch.get_rng_state())


def test_batch_rejects_empty_or_mismatched_lane_seed_list_and_restores_hook():
    m = motor()
    original = m.model.build_prefix_cache
    with pytest.raises(ValueError):
        m.infer_batch(dict(states=np.zeros((1, 25), np.float32)), noise_seeds=[])
    with pytest.raises(ValueError):
        m.infer_batch(dict(states=np.zeros((2, 25), np.float32)), noise_seeds=[1])
    assert m.model.build_prefix_cache == original
