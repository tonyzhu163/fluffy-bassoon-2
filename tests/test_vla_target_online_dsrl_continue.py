"""Lossless development continuation semantics; allocated Linux tests only."""
from copy import deepcopy
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pytest

import fluffy_bassoon.meta_rl.vla_target_online_dsrl_continue as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent_fixture(tmp_path, task=0):
    root = tmp_path / 'terminal-six'
    case = root / 'experiment' / f'case-{task:02d}'
    case.mkdir(parents=True)
    (root / 'launcher.lock').touch()
    (root / 'exit-code.txt').write_text('0\n')
    old_contract = dict(case=module.onset.case(task), recipe=module.six.RECIPE,
        source_commit=module.PARENT_SOURCE, maximum_total_actions=9500)
    payload = dict(agent=b'unchanged-actor-critic-target-temperature-and-optimizer',
        agent_rng=np.array([3, 4], np.uint32), replay=[dict(actions=np.arange(32, dtype=np.float32))],
        updates=3200, sampler=np.random.default_rng(0).bit_generator.state,
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), contract=old_contract,
        boundary='complete physical episode', progress=dict(training_episodes=16, next_milestone=6,
            motor_rng=dict(format='jax_motor_key_v1', key_data=np.array([5, 6], np.uint32))))
    with (case / 'target-recovery.pkl').open('wb') as stream:
        pickle.dump(payload, stream, protocol=5)
    write(case / 'contract.json', old_contract)
    write(case / 'index.json', dict(completed=True, case=module.onset.case(task),
        contract_sha256=module.content_hash(old_contract), training_episodes=16, training_chunks=640,
        target_optimizer_updates=3200, query_receipts=[dict(training_episode_budget=x) for x in module.EPISODES[:6]],
        maximum_total_actions=9500, total_action_charge=9030, target_only=True, source_assembly_admission=False,
        latest_recovery_sha256=module.onset.sha256(case / 'target-recovery.pkl')))
    write(root / 'experiment/index.json', dict(format=module.six.FORMAT, completed=True,
        total_action_charge=35558, contract=dict(source_commit=module.PARENT_SOURCE)))
    write(root / 'worker-time.json', dict(worker_seconds=7200, contract=dict(source_commit=module.PARENT_SOURCE,
        run_id=root.name), attempts=[dict(status='finished', exit_code=0, charged_seconds=1851,reserved_seconds=7200)]))
    journal = case / 'physical-journal'
    event = 0
    for role, count in (('training',16),('query',6),('qualification',1)):
        for _ in range(count):
            for kind, steps in [('reset',10)] + ([] if role=='qualification' else [('step',400)]):
                event += 1
                write(journal / f'{event}.json', dict(role=role,kind=kind,returned=True,
                    actual_steps=steps,maximum_steps=steps))
    (case/'raw').mkdir()
    (case/'raw/parent-episode.pkl.gz').write_bytes(b'original raw public observations and actions')
    return root, payload


def test_extended_schedule_keeps_prefix_cursor_and_declares_unique_costs(tmp_path):
    root, _ = parent_fixture(tmp_path)
    parent = module.terminal_parent(root,0)
    plan = module.contract('a'*40,0,parent,6000)
    assert tuple(module.EPISODES[:6]) == tuple(x[0] for x in module.six.MILESTONES)
    assert module.EPISODES[6:] == (40,80,160,240)
    assert plan['maximum_new_actions']==100000
    assert plan['maximum_cumulative_case_actions']==109030
    assert plan['parent']['parent_campaign_action_charge']==35558
    assert parent['parent_worker_charge_seconds']==1851
    assert 'Outcome-based development' in plan['task_selection']
    assert 224*410+4*410+10==93490 < 100000
    with pytest.raises(ValueError):
        module.contract('a'*40,0,parent,4800)


def test_migration_preserves_all_raw_and_learner_rng_leaves(tmp_path):
    root, original = parent_fixture(tmp_path)
    parent = module.terminal_parent(root,0)
    plan = module.contract('a'*40,0,parent,6000)
    output=tmp_path/'new/experiment'
    output.mkdir(parents=True)
    case, receipt=module.migrate(parent,output,plan)
    with (case/'target-recovery.pkl').open('rb') as stream:
        saved=pickle.load(stream)
    assert module.state_fingerprint(saved)==module.state_fingerprint(original)
    assert saved['contract']!=original['contract']
    assert saved['progress']['next_milestone']==6
    assert (case/'parent-target-recovery.pkl').read_bytes()==(Path(parent['source_case'])/'target-recovery.pkl').read_bytes()
    assert module.onset.charged(case)==9030
    assert receipt['numerical_state_unchanged']
    again,_=module.migrate(parent,output,plan)
    assert again==case
    (case/'raw/parent-episode.pkl.gz').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='evidence changed'):
        module.migrate(parent,output,plan)


def test_parent_must_be_terminal_complete_exact_source_and_case(tmp_path):
    root,_=parent_fixture(tmp_path,40)
    assert module.terminal_parent(root,40)['task']==40
    with pytest.raises(ValueError):
        module.terminal_parent(root,10)
    p=root/'experiment/index.json'
    changed=module.read(p)
    changed['contract']['source_commit']='b'*40
    write(p,changed)
    with pytest.raises(ValueError,match='exact terminal'):
        module.terminal_parent(root,40)
