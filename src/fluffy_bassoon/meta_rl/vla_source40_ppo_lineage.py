"""Bind frozen PPO deployment to successful terminal R3 repair lineages.

Reads terminal metadata and immutable migration proofs only. It never searches
for an older checkpoint, changes a reservation, or treats failed partial runs
as final models. Full checkpoint admission remains in frozen_validation.
"""
import hashlib
import json
from pathlib import Path

from .vla_source40_schedule import content_hash


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def clock_charge(clock):
    rows = clock.get('attempts', [])
    if (clock.get('worker_seconds') != 21600 or not rows
            or any(row.get('status') != 'finished' or type(row.get('charged_seconds')) is not int
                   or row['charged_seconds'] < 0 for row in rows)):
        raise ValueError('Source clock must contain only audited terminal attempts under21600 seconds')
    value = sum(row['charged_seconds'] for row in rows)
    if value > 21600:
        raise ValueError('Cumulative source GPU charge exceeds its original six-hour ceiling')
    return value


def terminal_source(root, mode, ledger_path):
    root, ledger_path = Path(root).resolve(), Path(ledger_path).resolve()
    exit_path, index_path = root / 'exit-code.txt', root / 'experiment/index.json'
    if not exit_path.is_file() or exit_path.read_text().strip() != '0' or not index_path.is_file():
        raise ValueError('Require the explicitly supplied successful terminal R3 source; no failed/partial fallback')
    clock_path, recovery = root / 'worker-time.json', root / 'experiment/learner/latest.pt'
    clock, index = read(clock_path), read(index_path)
    charge = clock_charge(clock)
    outer = read(root / 'experiment/source-contract.json')
    if (clock['attempts'][-1].get('exit_code') != 0 or clock.get('contract', {}).get('run_id') != root.name
            or clock['contract'].get('source_commit') != outer.get('source_commit')
            or clock['contract'].get('memory_mode') != mode or outer.get('memory_mode') != mode
            or clock['contract'].get('maximum_total_actions') != 172000 or outer.get('maximum_total_actions') != 172000
            or index.get('source_contract') != outer or index.get('actual_tasks_with_committed_episodes') != 40
            or index.get('committed_task_exposure', {}).get('remaining') != 0):
        raise ValueError('Terminal source index, mode, task coverage or original172k worker contract differs')
    cost = index.get('physical_cost', {})
    if type(cost.get('total_action_charge')) is not int or not 0 < cost['total_action_charge'] <= 172000:
        raise ValueError('Terminal source must retain all physical charges under its original172k ceiling')
    ledger, chain, seen = read(ledger_path), [], set()
    current, current_clock = root, clock
    original_id = None
    while (current / 'migration.json').is_file():
        path = current / 'migration.json'
        value = read(path)
        if current in seen:
            raise ValueError('Source migration ancestry contains a cycle')
        seen.add(current)
        original_id = original_id or value['original_reservation_id']
        reservation = ledger.get('reservations', {}).get(original_id)
        parent = Path(value['source_root']).resolve()
        parent_clock_path = parent / 'worker-time.json'
        parent_clock = read(parent_clock_path)
        carried = clock_charge(parent_clock)
        if (value.get('format') != 'source40_ppo_explicit_recovery_migration_v1'
                or Path(value['destination_root']).resolve() != current
                or value['original_reservation_id'] != original_id or value.get('original_reservation') != reservation
                or value.get('reservation_sha256') != content_hash(reservation)
                or value.get('maximum_total_actions') != 172000 or value.get('worker_seconds') != 21600
                or value.get('numerical_state_unchanged') is not True
                or value.get('state_sha256_before') != value.get('state_sha256_after')
                or value.get('carried_time_seconds') != carried
                or current_clock['attempts'][:len(parent_clock['attempts'])] != parent_clock['attempts']
                or current_clock.get('migration', {}).get('original_time_sha256') != sha(parent_clock_path)
                or sha(value['initial_checkpoint_path']) != value['initial_checkpoint_sha256']
                or sha(parent / 'experiment/learner/latest.pt') != value['original_checkpoint_sha256']
                or sha(current / 'prior-artifacts.json') != value['raw_evidence_index_sha256']
                or cost['total_action_charge'] < value['carried_physical_cost']['total_action_charge']):
            raise ValueError('Source migration lost original reservation, state, raw evidence, charges or cumulative time')
        parent_migration = value.get('parent_migration_path')
        if parent_migration:
            if (Path(parent_migration).resolve() != parent / 'migration.json'
                    or sha(parent_migration) != value['parent_migration_sha256']):
                raise ValueError('Immutable parent migration receipt differs')
        elif parent.name != original_id or (parent / 'migration.json').exists():
            raise ValueError('Source migration does not terminate at its original canonical R1 identity')
        if current.name in ledger.get('reservations', {}):
            raise ValueError('A migrated source must not create a duplicate R2/R3 action reservation')
        chain.append(dict(path=str(path), sha256=sha(path), source_root=str(parent), destination_root=str(current),
            source_commit=value['source_commit'], carried_time_seconds=carried,
            carried_physical_cost=value['carried_physical_cost'],
            initial_checkpoint_sha256=value['initial_checkpoint_sha256'],
            raw_evidence_index_sha256=value['raw_evidence_index_sha256']))
        current, current_clock = parent, parent_clock
    if len(chain) < 2 or current.name != original_id:
        raise ValueError('Require explicit R1→R2→R3 repair ancestry for these frozen source policies')
    reservation = ledger['reservations'][original_id]
    if (reservation.get('run_id') != original_id or reservation.get('maximum_total_actions') != 172000
            or current_clock['contract'].get('source_commit') != reservation.get('source_commit')
            or chain[0]['source_commit'] != outer['source_commit']):
        raise ValueError('Final source is not bound to the unchanged original R1 reservation')
    return dict(terminal_root=str(root), ledger_path=str(ledger_path),
        recovery_path=str(recovery), recovery_sha256=sha(recovery),
        worker_time_path=str(clock_path), worker_time_sha256=sha(clock_path), worker_time_charge_seconds=charge,
        terminal_index_path=str(index_path), terminal_index_sha256=sha(index_path),
        terminal_exit_path=str(exit_path), terminal_exit_sha256=sha(exit_path),
        original_reservation_id=original_id, original_reservation_sha256=content_hash(reservation),
        migration_chain=list(reversed(chain)), source_commit=outer['source_commit'],
        original_source_commit=reservation['source_commit'], cumulative_physical_cost=cost)
