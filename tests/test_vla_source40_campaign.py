"""The shared action ceiling cannot be multiplied by duplicate source launches."""
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_campaign import (ALL_TASKS, VALIDATION_TASKS,
    load_campaign, reserve, reserve_ppo, reserve_diagnostic, FORMAT)
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash


def fixture(tmp_path, monkeypatch):
    plan = dict(schedule_sha256='s', partition_budgets=dict(train=dict(maximum_total_actions=9500),
        validation=dict(maximum_total_actions=12000)))
    monkeypatch.setattr('fluffy_bassoon.meta_rl.vla_source40_campaign.load_schedule', lambda _: plan)
    data = dict(format=FORMAT, maximum_total_actions=1000000,
        source_training_task_keys=ALL_TASKS, source_validation_task_keys=VALIDATION_TASKS,
        source_schedule_sha256='s', prior_qualification_actions=12944,
        sac_maximum_total_actions=490000, ppo_maximum_total_actions=490000,
        unassigned_reserve_actions=21056, sac_slots=[dict(task_key=t, partition=p, maximum_total_actions=c)
            for p, tasks, c in [('train', ALL_TASKS, 9500), ('validation', VALIDATION_TASKS, 12000)] for t in tasks])
    data['campaign_sha256'] = content_hash(data)
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(data))
    return path, tmp_path / 'ledger.json'


def test_cap_counts_prior_source_and_both_algorithms(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    data = load_campaign(path)
    assert sum(s['maximum_total_actions'] for s in data['sac_slots']) == 476000
    data['ppo_maximum_total_actions'] += 1
    data['campaign_sha256'] = content_hash({k: v for k, v in data.items() if k != 'campaign_sha256'})
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Aggregate'):
        load_campaign(path)


def test_no_duplicate_stream_or_identity_and_exact_resume_only(tmp_path, monkeypatch):
    path, ledger = fixture(tmp_path, monkeypatch)
    kwargs = dict(task='libero_spatial:00', partition='train', run='source-r1', commit='a' * 40)
    first = reserve(path, 'schedule', ledger, **kwargs)
    with pytest.raises(ValueError, match='Slot already'):
        reserve(path, 'schedule', ledger, **dict(kwargs, run='source-r2'))
    assert reserve(path, 'schedule', ledger, **kwargs, resume=True) == first
    with pytest.raises(ValueError, match='Slot already'):
        reserve(path, 'schedule', ledger, **dict(kwargs, commit='b' * 40), resume=True)
    with pytest.raises(ValueError, match='Run identity'):
        reserve(path, 'schedule', ledger, **dict(kwargs, task='libero_spatial:01'))
    with pytest.raises(ValueError, match='no matching'):
        reserve(path, 'schedule', ledger, **dict(kwargs, task='libero_spatial:02', partition='validation'))


def test_ppo_fresh_rollout_reservations_share_cap_and_cannot_duplicate(tmp_path, monkeypatch):
    path, ledger = fixture(tmp_path, monkeypatch)
    kwargs = dict(run='ppo-r1', commit='a' * 40, action_cap=9500)
    record = reserve_ppo(path, ledger, **kwargs)
    with pytest.raises(ValueError, match='already reserved'):
        reserve_ppo(path, ledger, **kwargs)
    assert reserve_ppo(path, ledger, **kwargs, resume=True) == record
    with pytest.raises(ValueError, match='exceed'):
        reserve_ppo(path, ledger, run='ppo-r2', commit='b' * 40, action_cap=490000)


def test_target_diagnostic_escrow_preserves_source_slots_and_total_cap(tmp_path, monkeypatch):
    path, ledger = fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    kwargs = dict(run='subset-eval-r1', commit='a' * 40, action_cap=20000, prior_action_charge=10)
    record = reserve_diagnostic(path, ledger, **kwargs)
    assert record['source_pretraining_data'] is False
    assert record['pool'] == 'unassigned_reserve_actions'
    assert reserve_diagnostic(path, ledger, **kwargs, resume=True) == record
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='already reserved'):
        reserve_diagnostic(path, ledger, **kwargs)
    with pytest.raises(ValueError, match='already reserved'):
        reserve_diagnostic(path, ledger, **dict(kwargs, action_cap=19000), resume=True)
    with pytest.raises(ValueError, match='exceed'):
        reserve_diagnostic(path, ledger, run='other-r1', commit='b' * 40, action_cap=1047, prior_action_charge=10)
    with pytest.raises(ValueError, match='remain fixed'):
        reserve_diagnostic(path, ledger, **dict(kwargs, prior_action_charge=0), resume=True)
    reserve_diagnostic(path, ledger, run='other-r1', commit='b' * 40, action_cap=1046, prior_action_charge=10)


def test_diagnostic_rejects_unreserved_resume_and_wrong_pool(tmp_path, monkeypatch):
    path, ledger = fixture(tmp_path, monkeypatch)
    kwargs = dict(run='subset-eval-r1', commit='a' * 40, action_cap=20000)
    with pytest.raises(ValueError, match='no reservation'):
        reserve_diagnostic(path, ledger, **kwargs, resume=True)
    reserve_ppo(path, ledger, run='ppo-r1', commit='a' * 40, action_cap=9500)
    with pytest.raises(ValueError, match='another campaign or budget pool'):
        reserve_diagnostic(path, ledger, **kwargs)
