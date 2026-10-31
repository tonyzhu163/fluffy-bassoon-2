"""Explicit file-only PPO recovery migration under its ORIGINAL reservations.

No policy forward pass, simulator or optimizer step runs here. CPU checkpoint
inspection verifies every tensor/RNG/cursor leaf while source pin metadata is
changed for an immutable repair identity.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .vla_source40_ppo import physical_charge
from .vla_source40_schedule import content_hash


FORMAT = 'source40_ppo_explicit_recovery_migration_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprint(value):
    import torch
    digest = hashlib.sha256()
    def walk(item):
        if torch.is_tensor(item):
            array = item.detach().cpu().numpy()
            digest.update(str(('tensor', array.shape, array.dtype)).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(('array', item.shape, item.dtype)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=repr):
                walk(key)
                walk(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(str((type(item).__name__, len(item))).encode())
            for entry in item:
                walk(entry)
        else:
            digest.update(str((type(item).__name__, item)).encode())
    walk(value)
    return digest.hexdigest()


def copy_catalog(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    catalog = []
    for path in sorted(source.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original = sha(path)
        shutil.copy2(path, target)
        if sha(target) != original:
            raise ValueError('Copied physical journal bytes differ')
        catalog.append(dict(path=str(relative), sha256=original, bytes=path.stat().st_size))
    return catalog


def prepare(source_root, destination_root, source_commit, ledger_path, reservation_id, *, expected_iteration=40, camera_max_lsb=0):
    import torch
    source_root, destination_root, ledger_path = map(Path, (source_root, destination_root, ledger_path))
    if destination_root.exists():
        raise ValueError('Migration requires a fresh r2 output identity')
    with (source_root / 'launcher.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        clock_path = source_root / 'worker-time.json'
        clock = json.loads(clock_path.read_text())
        if clock.get('worker_seconds') != 21600 or any(row.get('status') != 'finished' for row in clock['attempts']):
            raise ValueError('Original PPO must be terminal with audited cumulative time')
        time_charge = sum(row['charged_seconds'] for row in clock['attempts'])
        if time_charge >= 21600:
            raise ValueError('No approved component GPU time remains')
        ledger = json.loads(ledger_path.read_text())
        reservation = ledger['reservations'][reservation_id]
        checkpoint = source_root / 'experiment/learner/latest.pt'
        parent_path = source_root / 'migration.json'
        parent = json.loads(parent_path.read_text()) if parent_path.exists() else None
        if parent is not None and (parent.get('format') != FORMAT
                or Path(parent['destination_root']).resolve() != source_root.resolve()
                or parent['original_reservation_id'] != reservation_id
                or parent['original_reservation'] != reservation):
            raise ValueError('Parent migration does not retain the same original reservation')
        before_sha = sha(checkpoint)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        outer = json.loads((source_root / 'experiment/source-contract.json').read_text())
        if (saved.get('iteration') != expected_iteration or saved.get('boundary') != 'between_trials'
                or saved['contract']['source'] != outer
                or outer.get('source_commit') != (parent['source_commit'] if parent else reservation.get('source_commit'))
                or outer.get('format') != 'openpi_pi05_source40_shared_ppo_v1'
                or outer.get('source_only') is not True or outer.get('source_data_from_targets') is not False
                or reservation.get('run_id') != reservation_id
                or (parent is None and reservation_id != source_root.name)
                or reservation.get('maximum_total_actions') != 172000 or outer.get('maximum_total_actions') != 172000
                or clock['contract']['source_commit'] != outer['source_commit']
                or clock['contract']['run_id'] != source_root.name or clock['contract']['maximum_total_actions'] != 172000
                or clock['contract']['memory_mode'] != saved['contract']['mode']
                or saved.get('runtime', {}).get('task_cycle', {}).get('remaining') != 0):
            raise ValueError('Original checkpoint/time/reservation identity or complete cursor differs')
        # Only source/run metadata moves. Every numerical/scientific leaf stays.
        unchanged = {key: value for key, value in saved.items() if key != 'contract'}
        state_sha = fingerprint(unchanged)
        if parent is not None and (parent['state_sha256_after'] != state_sha
                or sha(parent['initial_checkpoint_path']) != parent['initial_checkpoint_sha256']):
            raise ValueError('Parent source recovery changed beyond this iteration40 repair chain')
        new_outer = deepcopy(outer)
        new_outer['source_commit'] = source_commit
        if camera_max_lsb:
            from .vla_reset_identity import RESET_IDENTITY
            if camera_max_lsb != 1:
                raise ValueError('Only the explicit measured one-LSB source40 profile may migrate')
            new_outer['reset_identity'] = RESET_IDENTITY
        migrated = deepcopy(saved)
        migrated['contract']['source'] = new_outer
        if camera_max_lsb:
            migrated['contract']['reset_identity'] = new_outer['reset_identity']
        if fingerprint({key: value for key, value in migrated.items() if key != 'contract'}) != state_sha:
            raise ValueError('Migration changed numerical recovery state')
        destination_root.mkdir(parents=True)
        experiment = destination_root / 'experiment'
        learner = experiment / 'learner'
        learner.mkdir(parents=True)
        write(experiment / 'source-contract.json', new_outer)
        write(learner / 'contract.json', migrated['contract'])
        initial = learner / 'migration-initial.pt'
        with initial.open('xb') as stream:
            torch.save(migrated, stream)
        shutil.copy2(initial, learner / 'latest.pt')
        restored = torch.load(learner / 'latest.pt', map_location='cpu', weights_only=False)
        after_sha = fingerprint({key: value for key, value in restored.items() if key != 'contract'})
        if after_sha != state_sha or restored['contract'] != migrated['contract'] or sha(checkpoint) != before_sha:
            raise ValueError('Serialized migration changed tensors, RNG, sampler, cursor or original checkpoint')
        journals = {}
        for folder in ('execution-journal', 'task-resets'):
            journals[folder] = copy_catalog(source_root / 'experiment' / folder, experiment / folder)
        for name in ('metrics.jsonl', 'episodes.jsonl'):
            path = source_root / 'experiment/learner' / name
            if path.exists():
                shutil.copy2(path, learner / name)
        # New episodes/traces write into new directories. Never link a writable
        # directory into r1, especially the failed trial41 behavior/raw files.
        (learner / 'traces').mkdir()
        prior = (json.loads((source_root / 'prior-artifacts.json').read_text())['files']
            if (source_root / 'prior-artifacts.json').exists() else [])
        for root in (source_root / 'experiment/raw', source_root / 'experiment/learner/traces',
                     source_root / 'attempts', source_root / 'experiment/attempts'):
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    prior.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        write(destination_root / 'prior-artifacts.json', dict(source_root=str(source_root), files=prior,
            role='Original immutable completed and uncommitted evidence; not padded into new trials'))
        new_clock = deepcopy(clock)
        new_clock['prior_contract'] = deepcopy(clock['contract'])
        new_clock['contract']['source_commit'], new_clock['contract']['run_id'] = source_commit, destination_root.name
        if camera_max_lsb:
            new_clock['contract']['reset_identity'] = new_outer['reset_identity']
        new_clock['migration'] = dict(source_root=str(source_root), original_time_sha256=sha(clock_path),
            original_time_charge_seconds=time_charge)
        write(destination_root / 'worker-time.json', new_clock)
        state = deepcopy(saved['environment_rng'])
        sampler = np.random.default_rng()
        sampler.bit_generator.state = state
        next_state, next_seed = int(sampler.choice(saved['contract']['train_states'])), int(sampler.integers(0, 2**31-1))
        cursor = saved['runtime']['task_cycle']['cursor']
        task_key = outer['source_task_keys'][cursor % len(outer['source_task_keys'])]
        receipt = dict(format=FORMAT, source_root=str(source_root), destination_root=str(destination_root),
            original_source_commit=parent['original_source_commit'] if parent else outer['source_commit'],
            parent_source_commit=outer['source_commit'], source_commit=source_commit,
            parent_migration_path=str(parent_path) if parent else None,
            parent_migration_sha256=sha(parent_path) if parent else None,
            original_checkpoint_sha256=before_sha, initial_checkpoint_path=str(initial), initial_checkpoint_sha256=sha(initial),
            iteration=saved['iteration'], counters=saved['counters'], state_sha256_before=state_sha, state_sha256_after=after_sha,
            numerical_state_unchanged=True, changed_metadata=['source_commit', 'worker.run_id'] + (['explicit_reset_identity_contract'] if camera_max_lsb else []),
            original_reservation_id=reservation_id, original_reservation=reservation,
            ledger_path=str(ledger_path), reservation_sha256=content_hash(reservation),
            maximum_total_actions=172000, worker_seconds=21600, carried_time_seconds=time_charge,
            carried_physical_cost=physical_charge(experiment / 'execution-journal'), copied_journals=journals,
            next_case=dict(task_key=task_key, initial_state=next_state, scene_seed=next_seed),
            raw_evidence_index_sha256=sha(destination_root / 'prior-artifacts.json'),
            reset_identity=new_outer.get('reset_identity'), camera_max_uint8_lsb=camera_max_lsb,
            repair='Fresh owned processes; explicit exact physical/proprio identity plus measured camera precision when opted in',
            old_trial41='Uncommitted attempts remain charged and referenced; resume repeats the same sampled case from iteration40')
        write(destination_root / 'migration.json', receipt)
        return receipt


def validate(path, *, output_root, source_commit, action_cap, mode, ledger_path, camera_max_lsb=0):
    path, output_root = Path(path), Path(output_root)
    value = json.loads(path.read_text())
    ledger = json.loads(Path(ledger_path).read_text())
    record = ledger['reservations'][value['original_reservation_id']]
    clock = json.loads((output_root / 'worker-time.json').read_text())
    if (value.get('format') != FORMAT or Path(value['destination_root']).resolve() != output_root.resolve()
            or value['source_commit'] != source_commit or action_cap != 172000
            or record != value['original_reservation'] or content_hash(record) != value['reservation_sha256']
            or record['maximum_total_actions'] != action_cap or value['numerical_state_unchanged'] is not True
            or value['state_sha256_before'] != value['state_sha256_after']
            or value.get('camera_max_uint8_lsb', 0) != camera_max_lsb
            or clock['contract']['source_commit'] != source_commit or clock['contract']['run_id'] != output_root.name
            or clock['contract']['memory_mode'] != mode or clock['worker_seconds'] != 21600
            or sha(value['initial_checkpoint_path']) != value['initial_checkpoint_sha256']):
        raise ValueError('Repair migration/reservation/state/time identity differs')
    if camera_max_lsb:
        from .vla_reset_identity import RESET_IDENTITY
        if value.get('reset_identity') != RESET_IDENTITY or clock['contract'].get('reset_identity') != RESET_IDENTITY:
            raise ValueError('Explicit renderer precision contract differs from migration')
    if value.get('parent_migration_path') and sha(value['parent_migration_path']) != value['parent_migration_sha256']:
        raise ValueError('An immutable parent migration receipt changed')
    for folder, entries in value['copied_journals'].items():
        for row in entries:
            if sha(output_root / 'experiment' / folder / row['path']) != row['sha256']:
                raise ValueError('An original physical journal entry was lost or changed')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    for name in ('source-root', 'output-root', 'ledger'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--source-commit', required=True)
    prep.add_argument('--reservation-id', required=True)
    prep.add_argument('--camera-reset-max-lsb', type=int, choices=(0, 1), default=0)
    check = sub.add_parser('validate')
    for name in ('migration', 'output-root', 'ledger'):
        check.add_argument('--' + name, type=Path, required=True)
    check.add_argument('--source-commit', required=True)
    check.add_argument('--action-cap', type=int, required=True)
    check.add_argument('--memory-mode', choices=('full', 'episode'), required=True)
    check.add_argument('--camera-reset-max-lsb', type=int, choices=(0, 1), default=0)
    args = parser.parse_args(argv)
    if args.operation == 'prepare':
        root = Path(__file__).resolve().parents[3]
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != args.source_commit:
            raise ValueError('Prepare from the exact reviewed repair code')
        value = prepare(args.source_root, args.output_root, args.source_commit, args.ledger, args.reservation_id, camera_max_lsb=args.camera_reset_max_lsb)
    else:
        value = validate(args.migration, output_root=args.output_root, source_commit=args.source_commit,
            action_cap=args.action_cap, mode=args.memory_mode, ledger_path=args.ledger, camera_max_lsb=args.camera_reset_max_lsb)
    print(json.dumps(dict(operation=args.operation, iteration=value['iteration'], carried_time_seconds=value['carried_time_seconds'],
        carried_physical_cost=value['carried_physical_cost'], next_case=value['next_case'], numerical_state_unchanged=True)), flush=True)


if __name__ == '__main__':
    main()
