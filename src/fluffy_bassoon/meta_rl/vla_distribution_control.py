"""Frozen mean/std interventions using exactly one original per-step random draw."""
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
import torch
from .vla_history import FrozenHistoryController
from .vla_history import file_sha256


class DistributionController(FrozenHistoryController):
    def __init__(self, policy, *, mean='original', std='learned', record_output=None):
        if mean not in ('original', 'zero') or std not in ('learned', 'unit'):
            raise ValueError('Unknown frozen distribution intervention')
        self.mean_operation, self.std_operation = mean, std
        super().__init__(policy, mode='full')
        self.last_distribution, self.draw_count = None, 0
        self.record_output = Path(record_output) if record_output is not None else None
        if self.record_output is not None:
            self.record_output.mkdir(parents=True, exist_ok=False)

    @torch.no_grad()
    def act(self, observation):
        if self.pending:
            raise RuntimeError('Observe actual feedback before the next distribution sample')
        observation = np.asarray(observation, np.float32)
        if observation.shape != (self.policy.config.observation_dim,) or not np.isfinite(observation).all():
            raise ValueError('Require one finite common public feature vector')
        token = np.concatenate((observation, self.previous_noise,
            np.asarray([self.previous_reward, self.previous_done, self.episode_start], np.float32)))
        weight = next(self.policy.parameters())
        mu, _, self.state = self.policy.step(torch.as_tensor(token[None], device=weight.device,
            dtype=weight.dtype), self.state, deterministic=True, generator=self.generator)
        learned_std = self.policy.log_std.clamp(-5, 2).exp().expand_as(mu)
        epsilon = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=self.generator)
        mean = mu if self.mean_operation == 'original' else torch.zeros_like(mu)
        std = learned_std if self.std_operation == 'learned' else torch.ones_like(learned_std)
        proposal = mean + std * epsilon
        def array(value):
            return value[0].detach().cpu().numpy().copy()
        proposed = array(proposal)
        self.draw_count += 1
        self.last_distribution = dict(format='frozen_steering_distribution_draw_v1', draw_index=self.draw_count,
            mean_operation=self.mean_operation, std_operation=self.std_operation,
            base_mean=array(mu), learned_std=array(learned_std), applied_mean=array(mean), applied_std=array(std),
            epsilon=array(epsilon), proposed_noise=proposed.copy(),
            expected_clipped_noise=np.clip(proposed, -1., 1.).astype(np.float32), random_draws=1, parameter_updates=0)
        if self.record_output is not None:
            path = self.record_output / f'draw-{self.draw_count:05d}.pkl'
            with path.open('xb') as stream:
                pickle.dump(dict(distribution=self.last_distribution, public_features=observation.copy(),
                    previous_noise=self.previous_noise.copy(), previous_reward=self.previous_reward,
                    previous_done=self.previous_done, episode_start=self.episode_start), stream, protocol=5)
            self.last_distribution = dict(self.last_distribution, before_rpc_path=str(path), before_rpc_sha256=file_sha256(path))
        self.pending = True
        return proposed

    def observe(self, executed_noise, reward, done):
        actual = np.asarray(executed_noise, np.float32)
        if not np.array_equal(actual, self.last_distribution['expected_clipped_noise']):
            raise ValueError('Actual executed coordinate differs from recorded distribution/clipping')
        super().observe(actual, reward, done)
        self.last_distribution = dict(self.last_distribution, actual_executed_noise=actual.copy(),
            actual_reward=float(reward), actual_done=bool(done))

    def snapshot(self):
        if self.pending:
            raise ValueError('Distribution recovery requires complete actual feedback')
        def cpu(value):
            if isinstance(value, dict): return {key: cpu(item) for key, item in value.items()}
            return value.detach().cpu().clone() if torch.is_tensor(value) else deepcopy(value)
        return dict(format='frozen_distribution_controller_context_v1', mean_operation=self.mean_operation,
            std_operation=self.std_operation, state=cpu(self.state), previous_noise=self.previous_noise.copy(),
            previous_reward=self.previous_reward, previous_done=self.previous_done, episode_start=self.episode_start,
            pending=False, generator=self.generator.get_state() if self.generator is not None else None,
            draw_count=self.draw_count, last_distribution=deepcopy(self.last_distribution))

    def restore(self, saved):
        if (saved.get('format') != 'frozen_distribution_controller_context_v1'
                or saved.get('mean_operation') != self.mean_operation or saved.get('std_operation') != self.std_operation
                or saved.get('pending') is not False):
            raise ValueError('Frozen distribution intervention/recovery boundary differs')
        device = next(self.policy.parameters()).device
        def target(value):
            if isinstance(value, dict): return {key: target(item) for key, item in value.items()}
            return value.to(device).clone() if torch.is_tensor(value) else deepcopy(value)
        self.state = target(saved['state'])
        for name in ('previous_noise', 'previous_reward', 'previous_done', 'episode_start', 'pending', 'draw_count', 'last_distribution'):
            setattr(self, name, deepcopy(saved[name]))
        self.generator = torch.Generator(device=device).set_state(saved['generator']) if saved['generator'] is not None else None


def qualify_original(policy, observation, *, seed=173):
    """Action/cache-feedback sampler equivalence only; zero robot interactions."""
    device = next(policy.parameters()).device
    original = FrozenHistoryController(policy, mode='full', generator=torch.Generator(device=device).manual_seed(seed))
    instrumented = DistributionController(policy)
    instrumented.generator = torch.Generator(device=device).manual_seed(seed)
    for step in range(4):
        a, b = original.act(observation), instrumented.act(observation)
        if not np.array_equal(a, b) or not torch.equal(original.generator.get_state(), instrumented.generator.get_state()):
            raise RuntimeError('Original distribution adapter does not reproduce qualified stochastic actions/RNG bitwise')
        for controller, proposal in ((original, a), (instrumented, b)):
            controller.observe(np.clip(proposal, -1., 1.).astype(np.float32), -1., step == 3)
    return dict(bitwise_actions=True, bitwise_sampler_state=True, verification_steps=4,
        robot_interactions=0, feedback='Prespecified synthetic qualification feedback only; never source/target training data')
