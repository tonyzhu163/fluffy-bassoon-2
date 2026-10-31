"""Native3090-token supervised noise policy and frozen RoboTwin controller."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .gtrxl import GatedTransformerXLCore
from .robotwin_history_corpus import FORMAT as DATA_FORMAT, native_tokens
from .vla_history import HistoryNoisePolicy, file_sha256
from .vla_noise_ppo import policy_inputs
from .warmstart import atomic_save, checked_load, validate_weights


MODEL_FORMAT = 'robotwin_native_history_policy_v1'
MODES = ('full', 'static')


@dataclass(frozen=True)
class NativeHistoryConfig:
    observation_dim: int = 2062
    noise_dim: int = 1024
    width: int = 128
    layers: int = 2
    heads: int = 4
    context_length: int = 300
    initial_std: float = .5
    gate_bias: float = 2.
    action_limit: float = 1.5
    train_episodes: int = 4

    def __post_init__(self):
        if (self.observation_dim != 2062 or self.noise_dim != 1024 or self.action_limit != 1.5
                or self.train_episodes != 4 or any(type(x) is not int or x < 1 for x in
                    (self.width,self.layers,self.heads,self.context_length))
                or self.width % self.heads or self.width % 2
                or not math.isfinite(self.initial_std) or self.initial_std <= 0
                or not math.isfinite(self.gate_bias)):
            raise ValueError('Invalid native2062/1024/bound1.5 history policy contract')


class NativeHistoryPolicy(HistoryNoisePolicy):
    """Reuse qualified GTrXL and Gaussian physical-target NLL, with native clocks.

    Proposals are physical coordinates. The frozen deployment adapter applies
    the explicit bound1.5 clip, as a declared supervised policy family.
    """
    def __init__(self, config=None):
        nn.Module.__init__(self)
        self.config = config or NativeHistoryConfig()
        self.core = GatedTransformerXLCore(3090,width=self.config.width,layers=self.config.layers,
            heads=self.config.heads,memory_length=self.config.context_length,gate_bias=self.config.gate_bias)
        self.actor = nn.Linear(self.config.width,1024)
        self.log_std = nn.Parameter(torch.full((1024,),math.log(self.config.initial_std)))
        self.register_buffer('observation_mean',torch.zeros(2062))
        self.register_buffer('observation_std',torch.ones(2062))


def load_dataset(index_path, expected_sha256, objective):
    index_path = Path(index_path)
    if file_sha256(index_path) != expected_sha256:
        raise ValueError('Native dataset index checksum differs')
    metadata = json.loads(index_path.read_text())
    if (metadata.get('format') != DATA_FORMAT or metadata.get('completed') is not True
            or metadata.get('partition') not in ('train','validation') or objective not in metadata.get('datasets',{})):
        raise ValueError('Require native dataset with explicit source partition/case provenance')
    receipt = metadata['datasets'][objective]
    path = Path(receipt['path'])
    if not path.is_absolute(): path = index_path.parent/path
    if file_sha256(path) != receipt['sha256']:
        raise ValueError('Native dataset array checksum differs')
    with np.load(path,allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in archive.files}
    lengths = arrays.pop('lengths')
    total = len(arrays['observations'])
    if (lengths.dtype != np.int64 or lengths.ndim != 1 or (lengths < 1).any()
            or int(lengths.sum()) != total or len(lengths) != len(metadata['task_keys'])
            or any(len(value) != total for value in arrays.values())
            or not metadata.get('case_ids') or len(metadata['case_ids']) != len(metadata['source_trials'])):
        raise ValueError('Native history/case lengths differ')
    for name in ('episode_starts','source_trial_starts','terminated','truncated','native_terminated','native_truncated','budget_truncated'):
        if arrays[name].dtype!=np.bool_ or arrays[name].shape!=(total,):
            raise ValueError('Native boundary/mask fields must be boolean vectors')
    if arrays['rewards'].dtype!=np.float32 or not np.isfinite(arrays['rewards']).all():
        raise ValueError('Native feedback must be finite float32')
    for name,width in (('observations',2062),('next_observations',2062),('history_inputs',3090),
                       ('executed_noise',1024),('target_noise',1024)):
        value=arrays[name]
        if value.shape != (total,width) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid native public feature/input/label array')
    if (np.abs(arrays['executed_noise']) > 1.5+1e-6).any() or (np.abs(arrays['target_noise']) > 1.5+1e-6).any():
        raise ValueError('Native data exceeds physical action bound')
    if objective != 'dpt' and not np.array_equal(arrays['target_noise'],arrays['executed_noise']):
        raise ValueError('AD/BC labels differ from actual executed physical noise')
    trials=[]; offset=0
    for length in lengths:
        trial={key:value[offset:offset+length] for key,value in arrays.items()}; offset+=length
        rows=[dict(features=trial['observations'][i],noise=trial['executed_noise'][i],reward=float(trial['rewards'][i]),
            episode_start=bool(trial['episode_starts'][i]),episode_index=int(trial['episode_indices'][i]),
            primitive_step=int(trial['primitive_steps'][i]),episode_limit=int(trial['episode_limits'][i]),
            trial_start=bool(trial['source_trial_starts'][i])) for i in range(length)]
        if (not rows[0]['episode_start'] or not (trial['terminated'][-1] or trial['truncated'][-1])
                or not np.allclose(native_tokens(rows,carry_across_trials=True),trial['history_inputs'],rtol=0,atol=1e-6)):
            raise ValueError('Native inputs are not causal actual history')
        trials.append(trial)
    if file_sha256(path) != receipt['sha256'] or file_sha256(index_path) != expected_sha256:
        raise ValueError('Native dataset changed while loading')
    metadata['selected_objective']=objective
    metadata['index_sha256']=expected_sha256
    return trials,metadata


def validate_partitions(train, validation):
    if (train['partition'] != 'train' or validation['partition'] != 'validation'
            or train['interface'] != validation['interface']
            or set(train['case_ids']) & set(validation['case_ids'])):
        raise ValueError('Require compatible disjoint whole source-training/validation cases')
    a,b=train['selected_objective'],validation['selected_objective']
    if not ((a=='ad' and b=='behavior_cloning') or (a==b=='dpt')):
        raise ValueError('AD validates against explicit BC; DPT validates against DPT labels')
    if a=='dpt' and train['teacher_recovery_sha256'] != validation['teacher_recovery_sha256']:
        raise ValueError('Paired DPT training/validation require the same pinned source teacher')


def sample_batch(trials, policy, sampler, *, batch_size, max_window, mode):
    if mode not in MODES or batch_size < 1 or max_window < 1:
        raise ValueError('Invalid native batch/context/mode')
    windows=[]
    for index in sampler.integers(len(trials),size=batch_size):
        trial=trials[int(index)]
        end=int(sampler.integers(1,len(trial['observations'])+1))
        length=int(sampler.integers(1,min(end,max_window)+1)); start=end-length
        tokens=trial['history_inputs'][start:end].copy()
        if mode=='static': tokens[:,2062:-2]=0. # Keep current episode/primitive clocks.
        windows.append((tokens,trial['target_noise'][start:end],trial['episode_starts'][start:end]))
    device=next(policy.parameters()).device; longest=max(len(window[0]) for window in windows)
    tokens=torch.zeros(longest,batch_size,3090,device=device)
    targets=torch.zeros(longest,batch_size,1024,device=device)
    valid=torch.zeros(longest,batch_size,dtype=torch.bool,device=device); starts=torch.zeros_like(valid)
    for lane,(observations,labels,boundaries) in enumerate(windows):
        length=len(observations)
        tokens[:length,lane]=torch.as_tensor(observations,device=device)
        targets[:length,lane]=torch.as_tensor(labels,device=device)
        valid[:length,lane]=True; starts[:length,lane]=torch.as_tensor(boundaries,device=device)
    return tokens,targets,valid,starts


def export_policy(path, policy, manifest):
    if not manifest or any(not value.isfinite().all() for value in policy.state_dict().values()):
        raise ValueError('Require finite native policy and complete provenance')
    return atomic_save(path,dict(format=MODEL_FORMAT,config=asdict(policy.config),manifest=manifest,
        policy={key:value.detach().cpu().clone() for key,value in policy.state_dict().items()}))


def load_policy(path, expected_sha256, *, device='cpu'):
    saved=checked_load(path,expected_sha256)
    if saved.get('format') != MODEL_FORMAT or saved.get('manifest',{}).get('mode') not in MODES:
        raise ValueError('Require the native history model format and reset contract')
    with torch.random.fork_rng(devices=[]): policy=NativeHistoryPolicy(NativeHistoryConfig(**saved['config']))
    validate_weights(policy,saved['policy']); policy.load_state_dict(saved['policy'],strict=True)
    if (policy.observation_std <= 0).any(): raise ValueError('Invalid source feature normalization')
    return policy.to(device).eval().requires_grad_(False),saved['manifest']


class FrozenNativeHistoryController:
    """Explicit physical-episode versus new-trial lifecycle, no deployment SGD."""
    def __init__(self, policy, manifest, *, interface, seed=0, erase_each_episode=False, reward_blind=False):
        if manifest.get('mode') not in MODES or interface != manifest.get('interface'):
            raise ValueError('Native motor/features/noise/reset interface differs from training')
        self.policy=policy.eval().requires_grad_(False); self.manifest=manifest
        self.mode=manifest['mode']; self.erase_each_episode=bool(erase_each_episode); self.reward_blind=bool(reward_blind)
        self.generator=torch.Generator(device=next(policy.parameters()).device).manual_seed(seed)
        self.reset_trial()

    def reset_trial(self, *, seed=None):
        if getattr(self,'pending',False):
            raise RuntimeError('Observe the actual pending outcome before starting another trial')
        if seed is not None: self.generator.manual_seed(seed)
        self.state=None; self.previous_noise=np.zeros(1024,np.float32); self.previous_reward=0.
        self.episode_index=-1; self.episode_limit=None; self.primitive_step=0; self.pending=False; self.episode_open=False

    def reseed_action_rng(self, seed):
        """Pair query action streams without erasing support state or clocks."""
        if self.pending or self.episode_open or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Reseed action RNG only between completed episodes with a declared seed')
        self.generator.manual_seed(seed)

    def begin_episode(self, *, episode_index, episode_limit):
        allowed=set((self.manifest['interface'].get('task_episode_limits') or {}).values()) or {self.manifest['interface']['episode_limit']}
        if (self.pending or self.episode_open or type(episode_index) is not int
                or episode_index != self.episode_index+1 or episode_limit not in allowed
                or (self.episode_limit is not None and episode_limit != self.episode_limit)):
            raise ValueError('End the previous episode; preserve sequential episode clocks and native horizon')
        self.episode_index=episode_index; self.episode_limit=int(episode_limit); self.primitive_step=0; self.episode_open=True
        if self.mode=='static' or self.erase_each_episode:
            self.state=None; self.previous_noise.fill(0.); self.previous_reward=0.

    @torch.no_grad()
    def act(self, features, *, deterministic=False):
        features=np.asarray(features,np.float32)
        if self.pending or not self.episode_open or features.shape != (2062,) or not np.isfinite(features).all():
            raise ValueError('Begin an episode and observe its previous actual outcome before acting')
        clear=self.mode=='static'
        token=policy_inputs(features,np.zeros(1024,np.float32) if clear else self.previous_noise,
            0. if clear or self.reward_blind else self.previous_reward,
            not clear and not self.erase_each_episode and self.episode_index>0 and self.primitive_step==0,
            self.episode_index,self.primitive_step,train_episodes=4,episode_limit=self.episode_limit)
        reference=next(self.policy.parameters())
        proposal,_,self.state=self.policy.step(torch.as_tensor(token[None],device=reference.device),
            None if clear else self.state,deterministic=deterministic,generator=self.generator)
        if not torch.isfinite(proposal).all(): raise FloatingPointError('Nonfinite native steering proposal')
        self.last_noise=np.clip(proposal[0].cpu().numpy(),-1.5,1.5).astype(np.float32)
        self.pending=True
        return self.last_noise.reshape(32,32).copy()

    def observe(self, noise, reward, done, *, executed_steps):
        actual=np.asarray(noise,np.float32).reshape(-1)
        if (not self.pending or actual.shape != (1024,) or not np.isfinite(actual).all()
                or not np.allclose(actual,self.last_noise,rtol=0,atol=1e-6)
                or not math.isfinite(float(reward)) or type(done) not in (bool,np.bool_)
                or type(executed_steps) is not int or not 0 <= executed_steps <= min(32,self.episode_limit-self.primitive_step)
                or (executed_steps==0 and not done)
                or (not done and self.primitive_step+executed_steps==self.episode_limit)):
            raise ValueError('Require actual bounded noise, reward, terminal flag and valid native duration')
        self.previous_noise=actual.copy(); self.previous_reward=float(reward)
        self.primitive_step+=executed_steps; self.pending=False; self.episode_open=not done
