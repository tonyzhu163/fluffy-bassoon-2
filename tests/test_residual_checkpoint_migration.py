"""A JSON migration manifest must preserve exact typed checkpoint recovery."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest
import torch

from test_robocasa_residual import bound, learner
from fluffy_bassoon.robocasa.residual import (
    collect_pending, install_terminal_labels, restore_checkpoint, save_checkpoint,
)
from fluffy_bassoon.meta_rl.residual_ppo import ppo_config_record, stack_records, update
from fluffy_bassoon.robocasa.ppo import rollout_migration_contract


@pytest.mark.parametrize('kind', ['eight_gpu_rollout_only', 'eight_gpu_evaluation_only'])
def test_json_roundtrip_keeps_typed_bound_full_recovery_and_next_adam(tmp_path, kind):
    policy, optimizer, motor, env, generators = learner()
    rows, episodes = collect_pending(policy, bound(), motor, env, episodes=2, generator=generators['action'])
    original = dict(source_commit='a' * 40, bound=asdict(bound()), motor_seed=42, arm='S')
    checkpoint = tmp_path / 'pending.pt'
    checksum = save_checkpoint(checkpoint, policy=policy, optimizer=optimizer, bound=bound(), motor=motor,
        environment=env, generators=generators, contract=original, iteration=0, stage='pending_labels',
        records=rows, episodes=episodes)
    current = dict(original, source_commit='b' * 40)
    manifest = json.loads(json.dumps(dict(kind=kind, original_contract=original,
        from_source_commit='a' * 40, to_source_commit='b' * 40, checkpoint_sha256=checksum)))
    resumed_contract = rollout_migration_contract(current, manifest, checkpoint_sha256=checksum, kind=kind)
    assert resumed_contract == original and isinstance(resumed_contract['bound']['indices'], tuple)
    p2, o2, m2, e2, g2 = learner()
    restored = restore_checkpoint(checkpoint, policy=p2, optimizer=o2, bound=bound(), motor=m2,
        environment=e2, generators=g2, contract=resumed_contract)
    labels = {episode['id']: index % 2 for index, episode in enumerate(episodes)}
    update(policy, optimizer, stack_records(install_terminal_labels(rows, episodes, labels, arm='S')),
           generator=generators['update'])
    update(p2, o2, stack_records(install_terminal_labels(restored['pending_records'], restored['pending_episodes'], labels, arm='S')),
           generator=g2['update'])
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, p2.state_dict()[key], rtol=0, atol=0)
    wrong = deepcopy(current)
    wrong['motor_seed'] += 1
    with pytest.raises(ValueError):
        rollout_migration_contract(wrong, manifest, checkpoint_sha256=checksum, kind=kind)
    wrong = deepcopy(current)
    wrong['bound']['scale'] = tuple(value * 2 for value in wrong['bound']['scale'])
    with pytest.raises(ValueError):
        rollout_migration_contract(wrong, manifest, checkpoint_sha256=checksum, kind=kind)
    with pytest.raises(ValueError):
        rollout_migration_contract(current, manifest, checkpoint_sha256=checksum,
                                  kind='eight_gpu_evaluation_only' if kind=='eight_gpu_rollout_only' else 'eight_gpu_rollout_only')


@pytest.mark.parametrize('target_kl', [None, .02])
def test_checkpoint_keeps_kl_setting_and_rejects_changed_threshold(tmp_path, target_kl):
    policy, optimizer, motor, env, generators = learner()
    policy.config = replace(policy.config, target_kl=target_kl)
    contract = dict(ppo=ppo_config_record(policy.config))
    path = tmp_path / 'boundary.pt'
    save_checkpoint(path, policy=policy, optimizer=optimizer, bound=bound(), motor=motor,
        environment=env, generators=generators, contract=contract, iteration=0, stage='boundary')
    saved = torch.load(path, weights_only=False)
    assert saved['config']['ppo'].get('target_kl') == target_kl
    if target_kl is None:
        assert 'target_kl' not in saved['config']['ppo']  # Pre-KL on-disk shape.
    p2, o2, m2, e2, g2 = learner()
    p2.config = replace(p2.config, target_kl=target_kl)
    restore_checkpoint(path, policy=p2, optimizer=o2, bound=bound(), motor=m2,
        environment=e2, generators=g2, contract=contract)
    p2.config = replace(p2.config, target_kl=.01)
    with pytest.raises(ValueError, match='Exact motor'):
        restore_checkpoint(path, policy=p2, optimizer=o2, bound=bound(), motor=m2,
            environment=e2, generators=g2, contract=contract)
