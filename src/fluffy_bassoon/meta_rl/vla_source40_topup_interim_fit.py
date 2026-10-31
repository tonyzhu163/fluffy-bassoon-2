"""Explicit 25/100-update source40 fitting point on a frozen CPU-built manifest.

This profile neither collects source actions nor assembles/recaptures a dataset.
The separate 25/100/500 default and production phase remain unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

FORMAT = 'source40_topup_interim_fit_25_100_v1'
WORKER_SECONDS = 1800
SLICE_SECONDS = 1740  # Plus the generic clock's60-second scheduler allowance.
SCALES = (25, 100)
OPTIMIZATION = dict(batch_size=8, learning_rate=3e-4, width=128, layers=2,
    heads=4, context_length=300, max_window=128, eval_every=25,
    validation_batches=4, threads=4, device='cuda:0')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def profile_contract(manifest, manifest_sha256, source_commit, run):
    if (len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit)
            or len(manifest_sha256) != 64 or any(c not in '0123456789abcdef' for c in manifest_sha256)
            or not run or any(not (c.isalnum() or c == '-') for c in run)):
        raise ValueError('Require exact source/manifest identities and a fresh run name')
    manifest = Path(manifest).resolve()
    if digest(manifest) != manifest_sha256:
        raise ValueError('Frozen CPU-produced manifest checksum differs')
    return dict(format=FORMAT, operation='fit_existing_immutable_source40_manifest',
        source_commit=source_commit, run_id=run, manifest_path=str(manifest),
        manifest_file_sha256=manifest_sha256, scales=list(SCALES), seed=0,
        optimization=OPTIMIZATION, max_controller_seconds=300, max_total_seconds=1500,
        worker_seconds=WORKER_SECONDS, additional_simulator_actions=0,
        interpretation='100-update interim source-size point; full1M/500-update phase remains outstanding')


def remaining_slice(root, contract, *, resume):
    """Completed failures/setup consume time; an unclosed attempt stays charged."""
    path = Path(root) / 'worker-time.json'
    if not path.exists():
        if resume:
            raise ValueError('Cannot resume without the original cumulative worker clock')
        return SLICE_SECONDS
    previous = json.loads(path.read_text())
    if (not resume or previous['contract'] != contract
            or previous['worker_seconds'] != WORKER_SECONDS):
        raise ValueError('Only the exact interim profile/manifest may resume')
    charge = sum(row.get('charged_seconds', row['reserved_seconds']) for row in previous['attempts'])
    seconds = min(SLICE_SECONDS, WORKER_SECONDS - charge - 60)
    if seconds < 600:
        raise RuntimeError('Insufficient original worker time for another bounded slice')
    return seconds


def begin_profile(root, manifest, manifest_sha256, source_commit, run, *, resume=False, started=None):
    from .vla_source40_ppo_worker import begin
    contract = profile_contract(manifest, manifest_sha256, source_commit, run)
    root = Path(root)
    if not resume and root.exists():
        raise ValueError('Require a fresh immutable interim output identity')
    seconds = remaining_slice(root, contract, resume=resume)
    attempt = begin(root, contract, slice_seconds=seconds, worker_seconds=WORKER_SECONDS,
        resume=resume, now=started)
    profile = root / 'profile.json'
    if profile.exists() and json.loads(profile.read_text()) != contract:
        raise ValueError('Stored interim profile differs')
    if not profile.exists():
        write(profile, contract)
    return dict(attempt=str(attempt), slice_seconds=seconds)


def fitting_command(manifest, manifest_sha256, *, output, source_commit, hostname, resume=False):
    # Keep the existing strict all40/fixed8 admission and source-cost accounting.
    from .vla_source40_topup_assembly import milestone_command
    command = milestone_command(manifest, manifest_sha256, output=output,
        source_commit=source_commit, hostname=hostname, resume=resume)
    first, end = command.index('--scales'), command.index('--seed')
    if command[first + 1:end] != ['25', '100', '500']:
        raise ValueError('Reviewed default milestones changed; re-review the interim profile')
    command[first + 1:end] = [str(value) for value in SCALES]
    for flag, old, new in (('--max-controller-seconds', '600', '300'),
                          ('--max-total-seconds', '2700', '1500')):
        offset = command.index(flag) + 1
        if command[offset] != old:
            raise ValueError('Reviewed default fitting bounds changed')
        command[offset] = new
    for name, value in OPTIMIZATION.items():
        command += ['--' + name.replace('_', '-'), str(value)]
    return command


def verify_completed(output):
    """Report only actual exact-update exports, each with full recovery bytes."""
    output = Path(output)
    result = json.loads((output / 'scales.json').read_text())
    if (result.get('completed') is not True or result.get('qualification_subset') is not False
            or result.get('unique_controller_optimizer_updates') != 400
            or [row['updates'] for row in result['scales']] != list(SCALES)):
        raise ValueError('The interim four-controller fitting point is incomplete')
    archived = []
    for scale in result['scales']:
        if digest(scale['index_path']) != scale['index_sha256']:
            raise ValueError('Interim milestone matrix checksum differs')
        matrix = json.loads(Path(scale['index_path']).read_text())
        rows = matrix['controllers']
        if (matrix.get('completed') is not True or len(rows) != 4
                or {(row['objective'], row['mode']) for row in rows}
                   != {(objective, mode) for objective in ('ad', 'dpt') for mode in ('full', 'static')}
                or any(row['updates'] != scale['updates'] or row['seed'] != 0 for row in rows)):
            raise ValueError('Require all four exact-update interim controllers')
        for row in rows:
            for stem in ('final', 'full_recovery'):
                if digest(row[stem + '_path']) != row[stem + '_sha256']:
                    raise ValueError('Interim policy/full-recovery bytes changed')
            archived.append(row)
    return dict(format=FORMAT, completed=True, full_phase_complete=False,
        milestones=list(SCALES), controllers=archived, additional_simulator_actions=0,
        assembly_sha256=result['assembly_sha256'],
        source_collection_costs_count_once=result['source_collection_costs_count_once'],
        interpretation='100-update interim point only; no completed1M/500 claim or robot-performance claim')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    clock = sub.add_parser('begin')
    train = sub.add_parser('train')
    for command in (clock, train):
        for name in ('root', 'manifest'):
            command.add_argument('--' + name, type=Path, required=True)
        for name in ('manifest-sha256', 'source-commit', 'run'):
            command.add_argument('--' + name, required=True)
        command.add_argument('--resume', action='store_true')
    clock.add_argument('--started', type=float, required=True)
    train.add_argument('--expected-hostname', required=True)
    train.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.operation == 'begin':
        print(json.dumps(begin_profile(args.root, args.manifest, args.manifest_sha256,
            args.source_commit, args.run, resume=args.resume, started=args.started), allow_nan=False))
        return
    if sys.platform != 'linux' or socket.gethostname() != args.expected_hostname:
        raise ValueError('Interim fitting requires the explicitly allocated Linux host')
    code = Path(__file__).resolve().parents[3]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() != args.source_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip()):
        raise ValueError('Require the exact clean reviewed source commit')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    contract = profile_contract(args.manifest, args.manifest_sha256, args.source_commit, args.run)
    if json.loads((args.root / 'profile.json').read_text()) != contract:
        raise ValueError('Cumulative worker profile differs from fitting inputs')
    command = fitting_command(args.manifest, args.manifest_sha256, output=args.root / 'experiment',
        source_commit=args.source_commit, hostname=args.expected_hostname, resume=args.resume)
    receipt = dict(format=FORMAT, completed=False, command=command, profile=contract, started_unix=time.time())
    write(args.receipt, receipt)
    try:
        subprocess.run(command, check=True)
        receipt['result'] = verify_completed(args.root / 'experiment')
        receipt['completed'] = True
        write(args.root / 'interim-result.json', receipt['result'])
    finally:
        receipt['ended_unix'] = time.time()
        write(args.receipt, receipt)


if __name__ == '__main__':
    main()
