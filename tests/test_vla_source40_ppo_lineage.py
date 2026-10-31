"""Terminal R3 admission retains original reservations and cumulative repairs."""
from copy import deepcopy
import json

import pytest

from fluffy_bassoon.meta_rl.vla_source40_ppo_lineage import sha, terminal_source
from fluffy_bassoon.meta_rl.vla_source40_schedule import content_hash


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def chain(tmp_path):
    roots = [tmp_path / f'ppo-full-r{i}' for i in (1, 2, 3)]
    reservation = dict(run_id=roots[0].name, source_commit='a' * 40, maximum_total_actions=172000)
    ledger = tmp_path / 'ledger.json'
    write(ledger, dict(reservations={roots[0].name: reservation}))
    attempts = []
    for number, root in enumerate(roots):
        source = chr(ord('a') + number) * 40
        outer = dict(source_commit=source, memory_mode='full', maximum_total_actions=172000)
        write(root / 'experiment/source-contract.json', outer)
        checkpoint = root / 'experiment/learner/latest.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f'complete recovery {number}'.encode())
        (root / 'exit-code.txt').write_text('0' if number == 2 else '1')
        attempts.append(dict(status='finished', charged_seconds=(1738, 187, 2000)[number], exit_code=0 if number == 2 else 1))
        clock = dict(worker_seconds=21600, attempts=deepcopy(attempts),
            contract=dict(source_commit=source, run_id=root.name, memory_mode='full', maximum_total_actions=172000))
        if number:
            parent = roots[number - 1]
            initial = root / 'experiment/learner/migration-initial.pt'
            initial.write_bytes(f'preserved numerical state with new source metadata {number}'.encode())
            write(root / 'prior-artifacts.json', dict(files=[dict(path='immutable-old-raw', sha256='old-hash')]))
            previous = parent / 'migration.json'
            migration = dict(format='source40_ppo_explicit_recovery_migration_v1',
                source_root=str(parent), destination_root=str(root), source_commit=source,
                original_reservation_id=roots[0].name, original_reservation=reservation,
                reservation_sha256=content_hash(reservation), maximum_total_actions=172000, worker_seconds=21600,
                numerical_state_unchanged=True, state_sha256_before='unchanged', state_sha256_after='unchanged',
                carried_time_seconds=sum(row['charged_seconds'] for row in attempts[:-1]),
                initial_checkpoint_path=str(initial), initial_checkpoint_sha256=sha(initial),
                original_checkpoint_sha256=sha(parent / 'experiment/learner/latest.pt'),
                raw_evidence_index_sha256=sha(root / 'prior-artifacts.json'),
                carried_physical_cost=dict(total_action_charge=26700 + number * 100),
                parent_migration_path=str(previous) if previous.exists() else None,
                parent_migration_sha256=sha(previous) if previous.exists() else None)
            write(root / 'migration.json', migration)
            clock['migration'] = dict(original_time_sha256=sha(parent / 'worker-time.json'))
        write(root / 'worker-time.json', clock)
        write(root / 'experiment/index.json', dict(source_contract=outer, actual_tasks_with_committed_episodes=40,
            committed_task_exposure=dict(remaining=0), physical_cost=dict(total_action_charge=60000)))
    return roots, ledger


def test_terminal_r3_binds_both_repairs_original172k_and_all_cumulative_time(tmp_path):
    roots, ledger = chain(tmp_path)
    before = ledger.read_bytes()
    value = terminal_source(roots[-1], 'full', ledger)
    assert value['terminal_root'] == str(roots[-1])
    assert value['original_reservation_id'] == roots[0].name
    assert value['worker_time_charge_seconds'] == 3925
    assert len(value['migration_chain']) == 2
    assert value['migration_chain'][0]['source_root'] == str(roots[0])
    assert value['migration_chain'][1]['destination_root'] == str(roots[-1])
    assert value['cumulative_physical_cost']['total_action_charge'] == 60000
    assert ledger.read_bytes() == before


def test_failed_r1_r2_or_failed_r3_never_fall_back_to_a_complete40update_checkpoint(tmp_path):
    roots, ledger = chain(tmp_path)
    for root in roots[:2]:
        with pytest.raises(ValueError, match='no failed/partial fallback'):
            terminal_source(root, 'full', ledger)
    (roots[-1] / 'exit-code.txt').write_text('1')
    with pytest.raises(ValueError, match='no failed/partial fallback'):
        terminal_source(roots[-1], 'full', ledger)


@pytest.mark.parametrize('corruption', ['clock_prefix', 'parent_receipt', 'prior_raw_catalog', 'duplicate_reservation'])
def test_lost_migration_evidence_or_refunded_time_is_rejected(tmp_path, corruption):
    roots, ledger = chain(tmp_path)
    if corruption == 'clock_prefix':
        path = roots[-1] / 'worker-time.json'
        value = json.loads(path.read_text())
        value['attempts'] = value['attempts'][1:]
        write(path, value)
    elif corruption == 'parent_receipt':
        path = roots[1] / 'migration.json'
        value = json.loads(path.read_text())
        value['carried_time_seconds'] = 0
        write(path, value)
    elif corruption == 'prior_raw_catalog':
        write(roots[-1] / 'prior-artifacts.json', dict(files=[]))
    else:
        value = json.loads(ledger.read_text())
        value['reservations'][roots[-1].name] = deepcopy(value['reservations'][roots[0].name])
        write(ledger, value)
    with pytest.raises(ValueError):
        terminal_source(roots[-1], 'full', ledger)
