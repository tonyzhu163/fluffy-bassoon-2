"""Resumable fitting and actor exports for decoded physical chunk histories.

Callers supply admitted complete trials and fixed fitting statistics. Validation
never fits statistics, and operational checkpoint boundaries are whole updates.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random

import numpy as np
import torch

from .chunk_history import ChunkHistoryLayout, MODES
from .chunk_policy import ChunkModelConfig, ChunkResidualPolicy
from .chunk_value import ChunkHistoryLearner, ChunkObjectiveConfig, chunk_batch, chunk_learning_step
from .vla_history_train import _rng_state, _restore_rng
from .warmstart import atomic_save, checked_load, validate_weights


FIELDS = ('features', 'bases', 'issued', 'delivered_masks', 'rewards',
          'terminated', 'truncated', 'physics_seconds', 'episode_starts')
RESUME_FORMAT = 'decoded_chunk_student_resume_v1'
POLICY_FORMAT = 'decoded_chunk_student_policy_v1'


@dataclass(frozen=True)
class ChunkFitConfig:
    method: str = 'dit'
    mode: str = 'full'
    seed: int = 0
    batch_size: int = 8
    max_episodes: int = 9
    value_updates: int = 1000
    actor_updates: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.

    def __post_init__(self):
        if (self.method not in ('ad', 'dit', 'ic_iql', 'static_behavior_cloning')
                or self.mode not in MODES or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or any(type(v) is not int or v < 1 for v in
                       (self.batch_size, self.max_episodes, self.actor_updates))
                or type(self.value_updates) is not int or self.value_updates < 0
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0):
            raise ValueError('Invalid decoded-student fitting configuration')
        if self.method in ('ad', 'static_behavior_cloning') and self.value_updates != 0:
            raise ValueError('Imitation has no value-fitting stage')
        if self.method in ('dit', 'ic_iql') and self.value_updates < 1:
            raise ValueError('Value-weighted fitting requires a value stage')
        if self.method == 'static_behavior_cloning' and self.mode != 'static':
            raise ValueError('The context-free imitation control uses static mode')


def sample_chunk_windows(trials, sampler, *, batch_size, max_episodes, context_length):
    """Select complete contiguous episodes before building causal tokens."""
    if (not trials or any(type(v) is not int or v < 1 for v in
                          (batch_size, max_episodes, context_length))):
        raise ValueError('Require nonempty physical trials and positive window limits')
    windows = []
    for _ in range(batch_size):
        trial = trials[int(sampler.integers(len(trials)))]
        starts = np.flatnonzero(trial['episode_starts'])
        if not len(starts) or starts[0] != 0:
            raise ValueError('Require an explicit complete first physical episode')
        first = int(sampler.integers(len(starts)))
        ends = np.r_[starts[1:], len(trial['features'])]
        maximum = min(max_episodes, len(starts)-first)
        admissible = [count for count in range(1, maximum+1)
                      if ends[first+count-1]-starts[first] <= context_length]
        if not admissible:
            raise ValueError('A physical episode exceeds the fixed context capacity')
        count = int(sampler.integers(1, admissible[-1]+1))
        begin, end = starts[first], ends[first+count-1]
        windows.append({key: np.asarray(trial[key])[begin:end].copy() for key in FIELDS})
    return windows


def _statistics(values, layout):
    expected = dict(feature_mean=(layout.feature_dim,), feature_std=(layout.feature_dim,),
                    residual_radius=layout.command_shape)
    result = {}
    for key, shape in expected.items():
        value = torch.as_tensor(values[key]).detach().cpu().clone()
        if value.dtype != torch.float32 or tuple(value.shape) != shape or not torch.isfinite(value).all():
            raise ValueError('Require fixed finite float32 decoded fitting statistics')
        if key != 'feature_mean' and not (value > 0).all():
            raise ValueError('Fitting scales and residual radii must be positive')
        result[key] = value
    return result


def _statistics_hash(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


class ChunkTrainer:
    """Value fitting followed by frozen-value actor extraction; complete resume."""
    def __init__(self, model_config, fit_config, statistics, data_contract, *, objective_config=None, device='cpu'):
        if not isinstance(model_config, ChunkModelConfig) or not isinstance(fit_config, ChunkFitConfig):
            raise ValueError('Require explicit decoded model and fitting configurations')
        data = json.loads(json.dumps(data_contract, allow_nan=False))
        if fit_config.method not in data.get('eligible_methods', []):
            raise ValueError('The supplied physical data is not admitted for this method')
        if fit_config.method == 'ad' and (not data.get('chronological_learning_histories')
                or len(set(data.get('teacher_policy_versions', []))) < 2):
            raise ValueError('AD requires qualified chronological multi-version histories')
        objective = 'ad' if fit_config.method == 'static_behavior_cloning' else fit_config.method
        objective_config = objective_config or ChunkObjectiveConfig(objective)
        if objective_config.objective != objective or objective_config.discount != data.get('discount'):
            raise ValueError('Preserve the declared objective and physical return clock')
        self.statistics = _statistics(statistics, model_config.layout)
        self.fit_config, self.objective_config = fit_config, objective_config
        self.contract = dict(model=asdict(model_config), fitting=asdict(fit_config),
            objective=asdict(objective_config), statistics_sha256=_statistics_hash(self.statistics), data=data)
        random.seed(fit_config.seed);np.random.seed(fit_config.seed);torch.manual_seed(fit_config.seed)
        self.learner = ChunkHistoryLearner(model_config, self.statistics['residual_radius'], objective_config).to(device)
        with torch.no_grad():
            for model in (self.learner.actor, self.learner.q, self.learner.v, self.learner.target_q):
                model.feature_mean.copy_(self.statistics['feature_mean'])
                model.feature_std.copy_(self.statistics['feature_std'])
        self.optimizers = dict(value=torch.optim.AdamW([*self.learner.q.parameters(), *self.learner.v.parameters()],
            lr=fit_config.learning_rate, weight_decay=fit_config.weight_decay),
            actor=torch.optim.AdamW(self.learner.actor.parameters(), lr=fit_config.learning_rate,
                                   weight_decay=fit_config.weight_decay))
        # Actor windows match across objectives even when value fitting consumes
        # a different number of draws. Both streams survive operational resume.
        self.samplers = {phase:np.random.default_rng(np.random.SeedSequence([fit_config.seed, index]))
                         for index, phase in enumerate(('actor', 'value'))}
        self.updates = dict(value=0, actor=0)
        self.valid_positions = dict(value=0, actor=0)
        self.failed = False
        self._statistics_signature = self._statistic_versions()

    def _statistic_versions(self):
        values = dict(self.statistics)
        for name in ('actor', 'q', 'v', 'target_q'):
            model = getattr(self.learner, name)
            values[name+'.mean'], values[name+'.std'] = model.feature_mean, model.feature_std
        values['actor.radius'] = self.learner.actor.residual_radius
        return {key:(id(value), value._version, value.dtype, value.device) for key,value in values.items()}

    def _check_statistics(self):
        if self._statistic_versions() != self._statistics_signature:
            raise RuntimeError('Fixed fitting statistics changed')

    @property
    def phase(self):
        if self.updates['value'] < self.fit_config.value_updates:
            return 'value'
        return 'actor' if self.updates['actor'] < self.fit_config.actor_updates else 'complete'

    def _batch(self, trials, sampler):
        windows = sample_chunk_windows(trials, sampler, batch_size=self.fit_config.batch_size,
            max_episodes=self.fit_config.max_episodes, context_length=self.learner.actor.config.context_length)
        return chunk_batch(windows, self.learner.actor, mode=self.fit_config.mode,
                           discount=self.objective_config.discount)

    def step(self, trials, *, execution_batch_size=None):
        if self.failed or self.phase == 'complete':
            raise RuntimeError('Resume a complete prior boundary after failure; do not exceed the fitting budget')
        phase = self.phase
        try:
            self._check_statistics()
            self.learner.train()
            batch = self._batch(trials, self.samplers[phase])
            metrics = chunk_learning_step(self.learner, self.optimizers[phase], batch,
                phase=phase, mode=self.fit_config.mode, execution_batch_size=execution_batch_size)
            self.updates[phase] += 1
            self.valid_positions[phase] += int(batch['valid'].sum())
            return dict(phase=phase, updates=dict(self.updates), valid_positions=dict(self.valid_positions), **metrics)
        except BaseException:
            self.failed = True
            raise

    @torch.no_grad()
    def validate(self, trials, *, phase, batches=4):
        if phase not in ('value', 'actor') or type(batches) is not int or batches < 1:
            raise ValueError('Require an explicit validation phase and positive window count')
        self._check_statistics()
        self.learner.eval()
        sampler = np.random.default_rng(734001)
        function = self.learner.value_loss if phase == 'value' else self.learner.actor_loss
        values = [float(function(self._batch(trials, sampler), mode=self.fit_config.mode)[0]) for _ in range(batches)]
        if not all(math.isfinite(value) for value in values):
            raise FloatingPointError('Nonfinite decoded validation objective')
        return float(np.mean(values))

    def save(self, path):
        if self.failed:
            raise RuntimeError('Never replace a resumable boundary with a failed update')
        self._check_statistics()
        return atomic_save(path, dict(format=RESUME_FORMAT, contract=deepcopy(self.contract),
            statistics=self.statistics, learner=self.learner.state_dict(),
            optimizers={key:value.state_dict() for key,value in self.optimizers.items()},
            updates=dict(self.updates), valid_positions=dict(self.valid_positions),
            rng=_rng_state(self.samplers['actor']), value_sampler=self.samplers['value'].bit_generator.state))

    @classmethod
    def restore(cls, path, expected_sha256, *, expected_contract, device='cpu'):
        saved = checked_load(path, expected_sha256, weights_only=False)
        if saved.get('format') != RESUME_FORMAT or saved.get('contract') != expected_contract:
            raise ValueError('Decoded fitting resume contract differs')
        if not _finite_tree((saved['learner'], saved['optimizers'])):
            raise ValueError('Nonfinite decoded learner or optimizer checkpoint')
        contract = saved['contract'];model = dict(contract['model'])
        model['layout'] = ChunkHistoryLayout(**model['layout'])
        result = cls(ChunkModelConfig(**model), ChunkFitConfig(**contract['fitting']),
            saved['statistics'], contract['data'], objective_config=ChunkObjectiveConfig(**contract['objective']), device=device)
        if result.contract != contract:
            raise ValueError('Restored fitting statistics differ')
        validate_weights(result.learner, saved['learner'])
        result.learner.load_state_dict(saved['learner'], strict=True)
        for name in ('actor', 'q', 'v', 'target_q'):
            model = getattr(result.learner, name)
            for key in ('feature_mean', 'feature_std'):
                if not torch.equal(getattr(model, key).cpu(), result.statistics[key]):
                    raise ValueError('Restored model normalization differs from fixed fitting statistics')
        if not torch.equal(result.learner.actor.residual_radius.cpu(), result.statistics['residual_radius']):
            raise ValueError('Restored actor bounds differ from fixed fitting statistics')
        result._statistics_signature = result._statistic_versions()
        for key, optimizer in result.optimizers.items():
            optimizer.load_state_dict(saved['optimizers'][key])
        result.updates, result.valid_positions = saved['updates'], saved['valid_positions']
        if (set(result.updates) != {'value', 'actor'} or set(result.valid_positions) != {'value', 'actor'}
                or any(type(v) is not int or v < 0 for v in (*result.updates.values(), *result.valid_positions.values()))
                or result.updates['value'] > result.fit_config.value_updates
                or result.updates['actor'] > result.fit_config.actor_updates
                or result.updates['actor'] and result.updates['value'] != result.fit_config.value_updates):
            raise ValueError('Invalid complete-update resume cursor')
        _restore_rng(saved['rng'], result.samplers['actor'])
        result.samplers['value'].bit_generator.state = saved['value_sampler']
        return result

    def export_policy(self, path):
        if self.failed:
            raise RuntimeError('Cannot export a failed learner')
        self._check_statistics()
        return atomic_save(path, dict(format=POLICY_FORMAT, contract=deepcopy(self.contract),
            updates=dict(self.updates), policy={key:value.detach().cpu().clone()
                                               for key,value in self.learner.actor.state_dict().items()}))


def load_chunk_policy(path, expected_sha256, *, expected_contract, device='cpu'):
    saved = checked_load(path, expected_sha256, weights_only=False)
    if saved.get('format') != POLICY_FORMAT or saved.get('contract') != expected_contract:
        raise ValueError('Decoded policy export contract differs')
    config = dict(saved['contract']['model'])
    config['layout'] = ChunkHistoryLayout(**config['layout'])
    policy = ChunkResidualPolicy(ChunkModelConfig(**config), saved['policy']['residual_radius']).to(device)
    validate_weights(policy, saved['policy'])
    policy.load_state_dict(saved['policy'], strict=True)
    return policy.eval().requires_grad_(False), saved
