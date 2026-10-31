"""Allocated-Linux checks for immutable prefixes while source collection runs."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup_assembly as module
from fluffy_bassoon.meta_rl.vla_history import FIELDS, file_sha256, load_dataset, save_dataset
from fluffy_bassoon.meta_rl.vla_source40_corpus import PI05_RECEIPT
from fluffy_bassoon.meta_rl.vla_source40_schedule import case_plan, content_hash
from fluffy_bassoon.meta_rl.vla_source40_training import (
    FEATURE_KIND, INDEX_FORMAT, NOISE_CONTRACT, load_assembly, load_source)


def source(root, slot):
    task, split = slot.split('/')
    suite, number = task.split(':')
    history = root / 'experiment'
    history.mkdir(parents=True)
    cases = case_plan(task, split, 4, 4, 0)
    feature = dict(kind=FEATURE_KIND, checkpoint_receipt_sha256=PI05_RECEIPT,
        model_config='pi05_libero', observation_dim=4104)
    producer = dict(format='openpi_pi05_libero40_persistent_source_v1', source_only=True,
        source_data_from_targets=False, source_commit='a' * 40, partition=split,
        task=dict(task_key=task, suite=suite, task_id=int(number)), cases=cases,
        history_id=task + '/' + split, motor=dict(receipt_sha256=PI05_RECEIPT, weights_frozen=True),
        feature_contract=feature, noise_contract=NOISE_CONTRACT, collection_mode='physical_budget',
        maximum_total_actions=9500 if split == 'train' else 12000, horizon=400, chunk=10)
    actor = history / 'case-00001-source.pkl'
    actor.write_bytes(('immutable source actor for ' + slot).encode())
    values = dict(observations=np.full((8, 4104), .01, np.float32),
        executed_noise=np.arange(256, dtype=np.float32).reshape(8, 32) / 100,
        rewards=np.zeros(8, np.float32), dones=np.ones(8, bool), episode_starts=np.ones(8, bool),
        teacher_steps=np.arange(8, dtype=np.int64))
    publication = dict(status='paired_ad_dpt', case_count=2, through_case=1, episodes=8, transitions=8,
        source_checkpoint=actor.name, source_checkpoint_sha256=file_sha256(actor), datasets={})
    for objective in ('ad', 'dpt'):
        trial = copy.deepcopy(values)
        trial['target_noise'] = trial['executed_noise'].copy() + (0 if objective == 'ad' else .25)
        path = history / 'prefix-through-case-00001' / f'{objective}.npz'
        manifest = save_dataset(path, [trial], objective=objective, metadata=dict(
            source_contract=producer, task_key=task, global_task_split='source40', local_scene_split=split,
            feature_contract=feature, motor_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
            trial_ids=[producer['history_id']], source_case_ids=[case['case_id'] for case in cases[:2]],
            source_generation_episodes=8, source_generation_chunks=8, source_updates=10000,
            source_actor_checkpoint_sha256=file_sha256(actor)))
        publication['datasets'][objective] = dict(path=str(path.relative_to(history)), sha256=manifest['sha256'])
    receipts = []
    count = 9 if split == 'train' else 8  # Current training has a partial third case.
    journal = history / 'raw/decision-journal'
    journal.mkdir(parents=True)
    for index in range(count):
        case = cases[index // 4]
        receipts.append(dict(case_id=case['case_id'], episode_in_case=index % 4,
            initial_state=case['initial_state'], scene_seed=case['scene_seed'], suite=suite, task_id=int(number),
            initial_state_vector_sha256=content_hash(dict(task=task, state=case['initial_state'])),
            primitive_steps=4, chunks=1))
        module.write(journal / f'intent-{index}.json', dict(maximum_primitive_steps=10))
        module.write(journal / f'decision-{index}.json', dict(primitive_steps=4))
        module.write(journal / f'reset-{index}.json', dict(settling_steps=10, returned=True))
    current_contract = dict(producer, source_commit='b' * 40,
        maximum_total_actions=21800 if split == 'train' else 12000)
    module.write(history / 'contract.json', current_contract)
    index = dict(format=INDEX_FORMAT, completed=split == 'validation', collection_qualified=split == 'validation',
        source_only=True, task_key=task, partition=split, source_updates=12000, chunks=count,
        collection_mode='physical_budget', source_contract_sha256=content_hash(current_contract),
        feature_contract=feature, motor_receipt_sha256=PI05_RECEIPT, noise_contract=NOISE_CONTRACT,
        episode_receipts=receipts, publications=[publication])
    module.write(history / 'index.json', index)
    actions = count * 14
    return (dict(root=str(root.resolve()), maximum_total_actions=21800 if split == 'train' else actions),
        dict(root=str(root.resolve()), actions=actions))


def bank(tmp_path, monkeypatch):
    ledger, slots = dict(historical_extra_actions=sum(module.EXTRAS.values()), leaves={}), []
    for ordinal, slot in enumerate(module.expected_slots()):
        leaf, parent = source(tmp_path / f'source-{ordinal:02d}', slot)
        ledger['leaves'][slot] = leaf
        slots.append(dict(slot=slot, parent=parent))
    plan = dict(plan_sha256='c' * 64, slots=slots)
    plan_path, ledger_path = tmp_path / 'plan.json', tmp_path / 'ledger.json'
    module.write(plan_path, plan)
    module.write(ledger_path, ledger)
    monkeypatch.setattr(module, 'load_plan', lambda path: module.read(path))
    return plan_path, ledger_path


def captured(tmp_path, monkeypatch):
    plan, ledger = bank(tmp_path, monkeypatch)
    result = module.capture(plan, file_sha256(plan), ledger, tmp_path / 'snapshot')
    path = tmp_path / 'snapshot/admission.json'
    return plan, ledger, result, path, module.load_admission(path, file_sha256(path))


def test_live_prefix_freezes_one_snapshot_with_real_clock_and_full_cumulative_cost(tmp_path, monkeypatch):
    _, _, value, path, admission = captured(tmp_path, monkeypatch)
    assert len(value['source_indices']['train']) == 40 and len(value['source_indices']['validation']) == 8
    selected = value['source_indices']['train'][0]
    snapshot = module.read(selected['index_path'])
    assert snapshot['completed'] is False and snapshot['stopping_reason'] == 'immutable_complete_prefix'
    assert snapshot['chunks'] == 8 and len(snapshot['episode_receipts']) == 8
    assert snapshot['source_updates'] == 10000  # Selected actor, not the later12000-step live actor.
    assert snapshot['total_action_budget_charge'] == 126  # Includes the ninth unused live episode.
    assert snapshot['case_coverage']['completed_cases'] == 2
    assert snapshot['case_coverage']['candidate_bank_exhausted'] is False
    loaded = load_source(selected['index_path'], 'train', prefix_snapshot_admission=admission)
    np.testing.assert_array_equal(loaded['datasets']['ad']['teacher_steps'], np.arange(8))
    assert loaded['episodes'] == 8 and loaded['costs']['charged_total_actions'] == 126
    with pytest.raises(ValueError, match='completed qualified'):
        load_source(selected['index_path'], 'train')  # Legacy admission stays strict.
    assert value['cumulative_source_action_charge'] == 40 * 126 + 8 * 112 + sum(module.EXTRAS.values())
    assert module.load_admission(path, file_sha256(path)) == admission


def test_full_all40_fit_uses_same_milestone_trainer_while_live_sources_advance(tmp_path, monkeypatch):
    _, ledger, _, admission_path, admission = captured(tmp_path, monkeypatch)
    result = module.assemble_topup(admission_path, file_sha256(admission_path), tmp_path / 'assembly')
    manifest = tmp_path / 'assembly/topup-inputs.json'
    command = module.milestone_command(manifest, file_sha256(manifest), output=tmp_path / 'fit',
        source_commit='d' * 40, hostname='allocated-linux', resume=True)
    assert 'fluffy_bassoon.meta_rl.vla_source40_subset_training' in command and '--all40-source' in command
    assert '--qualification-subset' not in command and command[-1] == '--resume'
    assert command[command.index('--scales') + 1:command.index('--seed')] == ['25', '100', '500']
    live = Path(module.read(ledger)['leaves']['libero_spatial:00/train']['root']) / 'experiment/index.json'
    newer = module.read(live)
    newer['source_updates'] += 1000
    module.write(live, newer)
    assert load_assembly(tmp_path / 'assembly/assembly.json')['campaign'] == admission
    assert result['source_indices']['train'][0]['selected_chunks'] == 8
    from fluffy_bassoon.meta_rl.vla_source40_subset_training import frozen_baseline
    baseline = frozen_baseline(result['source_indices']['train'][0]['index_path'], prefix_snapshot_admission=admission)
    assert baseline['source_updates'] == 10000


def test_corrupt_latest_publication_cannot_be_replaced_by_an_older_or_parent_prefix(tmp_path, monkeypatch):
    plan, ledger = bank(tmp_path, monkeypatch)
    root = Path(module.read(ledger)['leaves']['libero_spatial:00/train']['root'])
    with (root / 'experiment/prefix-through-case-00001/ad.npz').open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum differs'):
        module.capture(plan, file_sha256(plan), ledger, tmp_path / 'bad')


def test_duplicate_or_target_slot_cannot_disappear_from_full_coverage(tmp_path, monkeypatch):
    plan, ledger = bank(tmp_path, monkeypatch)
    value = module.read(ledger)
    value['leaves']['libero_90:00/train'] = value['leaves'].pop('libero_spatial:00/train')
    module.write(ledger, value)
    with pytest.raises(ValueError, match='canonical all40'):
        module.capture(plan, file_sha256(plan), ledger, tmp_path / 'bad')


def test_snapshot_cost_receipts_cannot_be_changed_after_admission(tmp_path, monkeypatch):
    _, _, result, _, admission = captured(tmp_path, monkeypatch)
    row = result['source_indices']['train'][0]
    journal = Path(row['index_path']).parent / 'physical-cost-journal'
    module.write(journal / 'decision-0.json', dict(primitive_steps=1))
    with pytest.raises(ValueError, match='physical receipts'):
        load_source(row['index_path'], 'train', prefix_snapshot_admission=admission)
