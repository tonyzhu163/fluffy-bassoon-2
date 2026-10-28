"""Atomic learner snapshots taken only between complete meta-trials."""

import os
import re
from pathlib import Path

import torch

from fluffy_bassoon.training.training_state import restore_rng, rng_state
from fluffy_bassoon.evaluation_stages import sha256


def save_checkpoint(path, *, policy, optimizer, environment_rng, iteration, counters, contract):
    path = Path(path)
    temporary = path.with_suffix(".partial")
    payload = dict(version=1, boundary="between_trials", contract=contract,
                   iteration=iteration, counters=dict(counters), policy=policy.state_dict(),
                   optimizer=optimizer.state_dict(), environment_rng=environment_rng.bit_generator.state,
                   rng=rng_state())
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256(path)


def restore_checkpoint(path, *, policy, optimizer, environment_rng, contract):
    # Own training snapshots include Python and NumPy RNG state, not only tensors.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("version") != 1 or saved.get("boundary") != "between_trials"
            or saved.get("contract") != contract):
        raise ValueError("Checkpoint source or training contract differs")
    policy.load_state_dict(saved["policy"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    environment_rng.bit_generator.state = saved["environment_rng"]
    restore_rng(saved["rng"])
    return saved["iteration"], dict(saved["counters"])


@torch.no_grad()
def initialize_from_motor_checkpoint(path, expected_sha256, *, policy, contract):
    if not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-f]{64}',expected_sha256) or sha256(path)!=expected_sha256:
        raise ValueError('Motor checkpoint checksum differs from the declared input')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    parent=saved.get('contract',{})
    if (saved.get('version')!=1 or saved.get('boundary')!='between_trials'
            or type(saved.get('iteration')) is not int or saved['iteration']<1
            or not isinstance(parent,dict) or parent.get('mode')!='episode'
            or not isinstance(parent.get('source_commit'),str)
            or not re.fullmatch('[0-9a-f]{40}',parent['source_commit'])):
        raise ValueError('Require a complete episode-trained motor checkpoint')
    # shortcut: One motor stage; extend cost accounting before chaining warm-starts.
    if parent.get('initialization') is not None:
        raise ValueError('Chained motor initialization requires explicit pretraining cost accounting')
    keys=('task','attempts','horizon','observation_encoding','arms','core','core_config','dtype')
    if any(key not in parent or parent[key]!=contract[key] for key in keys):
        raise ValueError('Motor observation, action or core contract differs')
    counters=saved.get('counters',{})
    required={'environment_steps','rollout_forward_samples','update_forward_samples','optimizer_steps'}
    if (not isinstance(counters,dict) or not required.issubset(counters)
            or any(type(v) is not int or v<0 for v in counters.values())):
        raise ValueError('Require complete nonnegative motor pretraining counters')
    expected,actual=policy.state_dict(),saved.get('policy',{})
    if (not isinstance(actual,dict) or actual.keys()!=expected.keys()
            or any(not isinstance(actual[key],torch.Tensor) or actual[key].shape!=value.shape
                   or actual[key].dtype!=value.dtype or not actual[key].isfinite().all()
                   for key,value in expected.items()) or sha256(path)!=expected_sha256):
        raise ValueError('Motor checkpoint tensors or bytes differ from the declared schema')
    policy.load_state_dict(actual,strict=True)
    policy.critic.weight.zero_();policy.critic.bias.zero_()
    return dict(kind='episode_motor_warmstart',checkpoint_sha256=expected_sha256,
                source_commit=parent['source_commit'],source_iteration=saved['iteration'],
                pretraining_counters=dict(counters),
                transferred='core, actor and exploration scale',critic='zero initial value baseline',
                optimizer='fresh',rng='new phase seed; parent RNG not restored')
