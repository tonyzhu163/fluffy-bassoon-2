"""Canonical48-slot input resolution without constructing a model."""
import json
from pathlib import Path

import pytest

from fluffy_bassoon.meta_rl.vla_history import file_sha256
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash
from fluffy_bassoon.meta_rl.vla_source40_training import ALL_TASKS, BUDGET_VALIDATION_TASKS, CAMPAIGN_FORMAT
import fluffy_bassoon.meta_rl.vla_source40_offline_pipeline as module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def inputs(root):
    slots = [dict(task_key=task, partition=partition, maximum_total_actions=cap)
        for partition, tasks, cap in (('train', ALL_TASKS, 9500), ('validation', BUDGET_VALIDATION_TASKS, 12000))
        for task in sorted(tasks)]
    campaign = dict(format=CAMPAIGN_FORMAT, maximum_total_actions=1_000_000,
        source_training_task_keys=sorted(ALL_TASKS), source_validation_task_keys=sorted(BUDGET_VALIDATION_TASKS),
        source_schedule_sha256='s' * 64, prior_qualification_actions=12944, sac_slots=slots)
    campaign['campaign_sha256'] = content_hash(campaign)
    campaign_path, ledger_path = root / 'campaign.json', root / 'ledger.json'
    write(campaign_path, campaign)
    reservations = {}
    for ordinal, slot in enumerate(slots):
        run = f'actual-reserved-stream-{ordinal:02d}'
        record = dict(slot, run_id=run, source_commit='a' * 40,
            campaign_sha256=campaign['campaign_sha256'], source_schedule_sha256=campaign['source_schedule_sha256'])
        reservations[slot['task_key'] + '/' + slot['partition']] = record
        contract = dict(source_commit=record['source_commit'], schedule_sha256=record['source_schedule_sha256'],
            task=dict(task_key=slot['task_key']), partition=slot['partition'], maximum_total_actions=slot['maximum_total_actions'])
        directory = root / 'outputs' / run
        write(directory / 'experiment/contract.json', contract)
        write(directory / 'experiment/index.json', dict(completed=True, collection_qualified=True,
            source_updates=40, source_contract_sha256=content_hash(contract)))
        (directory / 'exit-code.txt').write_text('0\n')
    write(ledger_path, dict(campaign_sha256=campaign['campaign_sha256'], reservations=reservations))
    return campaign_path, ledger_path


def resolve(root, campaign, ledger):
    return module.canonical_inputs(root, campaign, file_sha256(campaign), ledger, file_sha256(ledger))


def test_canonical_inputs_keep40_training_goals_and_only_fixed8_validation_goals(tmp_path):
    campaign, ledger = inputs(tmp_path)
    before = ledger.read_bytes()
    _, groups, prior = resolve(tmp_path, campaign, ledger)
    assert len(groups['train']) == 40 and len(groups['validation']) == 8
    assert {row['task_key'] for row in groups['train']} == ALL_TASKS
    assert {row['task_key'] for row in groups['validation']} == BUDGET_VALIDATION_TASKS
    assert len({row['run_id'] for rows in groups.values() for row in rows}) == 48
    assert prior == 12944 and ledger.read_bytes() == before
    assert all(row['index_sha256'] == file_sha256(row['index_path']) for rows in groups.values() for row in rows)


def test_missing_or_raw_only_slots_cannot_be_filtered_into_a_full_corpus(tmp_path):
    campaign, ledger = inputs(tmp_path)
    data = json.loads(ledger.read_text())
    row = next(iter(data['reservations'].values()))
    path = tmp_path / 'outputs' / row['run_id'] / 'experiment/index.json'
    value = json.loads(path.read_text())
    value.update(completed=True, collection_qualified=False, source_updates=0)
    write(path, value)
    with pytest.raises(ValueError, match='All48 source histories.*raw-only'):
        resolve(tmp_path, campaign, ledger)
    data['reservations'].pop(next(iter(data['reservations'])))
    write(ledger, data)
    with pytest.raises(ValueError, match='every canonical all40'):
        resolve(tmp_path, campaign, ledger)


def test_reservation_source_and_cap_must_match_original_terminal_contract(tmp_path):
    campaign, ledger = inputs(tmp_path)
    data = json.loads(ledger.read_text())
    row = next(iter(data['reservations'].values()))
    row['source_commit'] = 'b' * 40
    write(ledger, data)
    with pytest.raises(ValueError, match='exact canonical reservation'):
        resolve(tmp_path, campaign, ledger)
    row['source_commit'], row['maximum_total_actions'] = 'a' * 40, 9501
    write(ledger, data)
    with pytest.raises(ValueError, match='changed identity, cap'):
        resolve(tmp_path, campaign, ledger)


def test_successful_exit_is_required_and_ledger_pin_cannot_drift(tmp_path):
    campaign, ledger = inputs(tmp_path)
    data = json.loads(ledger.read_text())
    row = next(iter(data['reservations'].values()))
    (tmp_path / 'outputs' / row['run_id'] / 'exit-code.txt').write_text('1\n')
    with pytest.raises(ValueError, match='terminal exit'):
        resolve(tmp_path, campaign, ledger)
    with pytest.raises(ValueError, match='source-ledger file SHA256'):
        module.canonical_inputs(tmp_path, campaign, file_sha256(campaign), ledger, '0' * 64)


def test_milestone_command_reuses_same_trainer_and_unchanged_optimizer_grid(tmp_path, monkeypatch):
    campaign, ledger = inputs(tmp_path)
    _, groups, _ = resolve(tmp_path, campaign, ledger)
    assembly_path = tmp_path / 'assembly.json'
    write(assembly_path, dict(immutable='full source assembly fixture'))
    selected = {partition: [dict(index_path=row['index_path'], index_sha256=row['index_sha256']) for row in rows]
        for partition, rows in groups.items()}
    assembly = dict(assembly_sha256='assembly-content', selected_sources=selected)
    monkeypatch.setattr(module, 'load_assembly', lambda *args, **kwargs: assembly)
    manifest = dict(format=module.FORMAT, completed=True, small_pilot=False,
        assembly_path=str(assembly_path), assembly_file_sha256=file_sha256(assembly_path),
        assembly_sha256='assembly-content', source_indices=groups)
    manifest['content_sha256'] = content_hash(manifest)
    path = tmp_path / 'all40-inputs.json'
    write(path, manifest)
    command = module.milestone_command(path, file_sha256(path), output=tmp_path / 'students',
        source_commit='a' * 40, hostname='allocated-host', resume=True)
    assert 'fluffy_bassoon.meta_rl.vla_source40_subset_training' in command and '--all40-source' in command
    assert '--qualification-subset' not in command and command[-1] == '--resume'
    train_start, val_start, assembly_start = (command.index(flag) for flag in ('--train-index', '--validation-index', '--assembly'))
    assert len(command[train_start + 1:val_start]) == 40
    assert len(command[val_start + 1:assembly_start]) == 8
    assert command[command.index('--scales') + 1:command.index('--seed')] == ['25', '100', '500']
    assert command[command.index('--max-controller-seconds') + 1] == '600'
    assert command[command.index('--max-total-seconds') + 1] == '2700'
