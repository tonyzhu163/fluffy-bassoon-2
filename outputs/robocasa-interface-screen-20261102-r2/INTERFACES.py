"""Finite, explicit interventions on the native frozen RoboCasa motor."""
from contextlib import contextmanager
import math

import numpy as np
import torch


def projection(inputs, outputs, seed):
    rng = np.random.default_rng(seed)
    value = rng.choice((-1., 1.), (inputs, outputs)).astype(np.float32)
    return torch.from_numpy(value / math.sqrt(inputs))


def kv_delta(cache, code, fraction):
    # Final-layer values only: leave keys/positions/masks and other layers intact.
    key, value = cache[-1]
    if not torch.count_nonzero(code):
        return cache
    assert value.shape[0] == code.shape[0]
    matrix = projection(code.shape[1], value.shape[-1], 921003).to(value.device)
    direction = (torch.tanh(code.float()) @ matrix).to(value.dtype)
    direction = direction.reshape(value.shape[0], *([1] * (value.ndim - 2)), value.shape[-1])
    rms = value.float().square().mean(tuple(range(1, value.ndim)), keepdim=True).sqrt().to(value.dtype)
    modified = value + fraction * rms * direction
    return (*cache[:-1], (key, modified))


def velocity_delta(velocity, code, fraction):
    if not torch.count_nonzero(code):
        return velocity
    assert velocity.shape[-1] == code.shape[-1] == 32
    # A low-dimensional code repeated across H50 at every native Euler step.
    return velocity + fraction * torch.tanh(code.float())[:, None, :].to(velocity.dtype)


class DirectHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(2064, 256), torch.nn.Tanh(),
                                           torch.nn.Linear(256, 600))
        self.register_buffer('mean', torch.zeros(2064))
        self.register_buffer('std', torch.ones(2064))

    @staticmethod
    def inputs(features):
        # Remove all first/mean base-action features (indices16..39).
        return torch.cat((features[..., :16], features[..., 40:]), -1)

    def forward(self, features):
        value = self.inputs(features)
        assert value.shape[-1] == 2064
        return self.network((value - self.mean) / self.std).reshape(-1, 50, 12)


class InterfaceMotor:
    def __init__(self, native, *, arm, controls, fraction=0., direct=None, critic=None):
        self.native, self.arm, self.controls, self.fraction = native, arm, controls, fraction
        self.direct, self.critic = direct, critic
        self.inference_draws = 0
        self.selector_choices = []
        self.selector_draws = {}

    @contextmanager
    def intervention(self, code):
        model = self.native.model
        prefix, velocity, predict = model.build_prefix_cache, model.velocity_from_suffix, model.predict_action_batch
        if self.arm == 'conditioning':
            def altered_prefix(*args, **kwargs):
                hidden, mask, cache = prefix(*args, **kwargs)
                return hidden, mask, kv_delta(cache, code, self.fraction)
            model.build_prefix_cache = altered_prefix
        elif self.arm == 'velocity':
            model.velocity_from_suffix = lambda suffix: velocity_delta(velocity(suffix), code, self.fraction)
        elif self.arm == 'noise':
            def altered_predict(observation, **kwargs):
                kwargs.pop('rng', None)
                kwargs['noise'] = torch.tanh(code)[:, None, :].expand(-1, 50, -1).contiguous()
                return predict(observation, **kwargs)
            model.predict_action_batch = altered_predict
        try:
            yield
        finally:
            model.build_prefix_cache, model.velocity_from_suffix, model.predict_action_batch = prefix, velocity, predict

    @torch.no_grad()
    def infer_batch(self, observation, *, noise_seeds):
        seeds = list(map(int, noise_seeds))
        code = torch.stack([self.controls[seed] for seed in seeds]).to(self.native.model.device)
        with self.intervention(code):
            chunks, features, _ = self.native.infer_batch(observation, noise_seeds=seeds)
        self.inference_draws += len(seeds)
        if self.arm == 'direct':
            stats = self.native.stats['actions']
            predicted = self.direct(torch.as_tensor(features, device=code.device)).cpu().numpy()
            mean, std = np.asarray(stats['mean'])[:12], np.asarray(stats['std'])[:12]
            # Same per-chunk six-dimensional random correction as action0.25,
            # but around the direct head rather than around the native expert.
            chunks = (predicted * (std + 1e-6) + mean).astype(np.float32)
        elif self.arm == 'select':
            banks, feature_banks = [chunks], [features]
            candidate_seeds = [seeds]
            for candidate in range(1, 8):
                more = [int(np.random.default_rng(seed + candidate * 1000003).integers(0, 2**63 - 1)) for seed in seeds]
                sample, feat, _ = self.native.infer_batch(observation, noise_seeds=more)
                banks.append(sample); feature_banks.append(feat)
                candidate_seeds.append(more)
                self.inference_draws += len(seeds)
            # A realizable fixed heuristic, not an oracle: the saved source
            # PPO value network scores each candidate's public frozen features.
            values = []
            for feat in feature_banks:
                tensor = torch.as_tensor(feat, device=code.device)
                normalized = (tensor - self.critic.feature_mean) / self.critic.feature_std
                values.append(self.critic.critic(normalized).squeeze(-1))
            selected = torch.stack(values).argmax(0).cpu().numpy()
            chunks = np.stack([banks[int(k)][i] for i, k in enumerate(selected)])
            features = np.stack([feature_banks[int(k)][i] for i, k in enumerate(selected)])
            self.selector_choices.extend(map(int, selected))
            for i, seed in enumerate(seeds):
                self.selector_draws[seed] = dict(candidate_seeds=[bank[i] for bank in candidate_seeds],
                    selected_candidate=int(selected[i]), actual_motor_seed=candidate_seeds[int(selected[i])][i])
        assert np.isfinite(chunks).all() and np.isfinite(features).all()
        return chunks, features, seeds
