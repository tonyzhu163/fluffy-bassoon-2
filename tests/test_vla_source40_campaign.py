"""The shared action ceiling cannot be multiplied by duplicate source launches."""
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_campaign import (ALL_TASKS, VALIDATION_TASKS,
    load_campaign, reserve, FORMAT)
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
