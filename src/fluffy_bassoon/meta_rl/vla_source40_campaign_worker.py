"""Prepare immutable SAC streams and execute one bounded campaign shard.

Preparation performs Git/filesystem operations only. Execution requires an
allocated Linux host and delegates each stream to its own pinned launcher.
This module never changes a stream's action cap or migrates its recovery.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

from .vla_source40_campaign import load_campaign
from .vla_source40_schedule import load_schedule


FORMAT = 'libero40_sac_campaign_worker_v1'
DEFAULT_ROOT = Path('/mowenjie/fluffy-bassoon-2')
PLAN_ID = 'libero40-source-1m-20261029-r1'
LAUNCHER = Path('scripts/diagnostics/vla_source40_campaign_collect.sh')
MAX_WORKER_SECONDS = 7140
SUITE_SHORT = dict(libero_spatial='s', libero_object='o', libero_goal='g', libero_10='l')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def file_lock(path, *, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            try:
                yield True
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def assigned_slots(campaign, worker, workers=8):
    if type(workers) is not int or not 1 <= workers <= 8 or type(worker) is not int or not 0 <= worker < workers:
        raise ValueError('Require a valid shard among at most eight independent workers')
    return campaign['sac_slots'][worker::workers]


def slot_key(slot):
    return slot['task_key'] + '/' + slot['partition']


def slot_name(slot):
    suite, number = slot['task_key'].split(':')
    return SUITE_SHORT[suite] + number + '-' + slot['partition']


def read_ledger(path, campaign):
    path = Path(path)
    with file_lock(path.with_suffix('.lock')):
        data = json.loads(path.read_text()) if path.exists() else dict(
            campaign_sha256=campaign['campaign_sha256'], reservations={})
    if data.get('campaign_sha256') != campaign['campaign_sha256']:
        raise ValueError('Worker ledger belongs to another campaign')
    return data['reservations']


def valid_prefix(prefix):
    if prefix is not None and (len(prefix) > 80 or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', prefix)):
        raise ValueError('Fresh source run prefix must be a lowercase hyphenated identity of at most80 characters')
    return prefix


def stream_record(slot, campaign, reservations, source_commit, fresh_run_prefix=None):
    valid_prefix(fresh_run_prefix)
    record = reservations.get(slot_key(slot))
    if record is None:
        name = (fresh_run_prefix + '-' + slot_name(slot) if fresh_run_prefix is not None else
                'libero40-1m-sac-20261029-' + slot_name(slot) + '-r1')
        return dict(slot, run_id=name,
            source_commit=source_commit, campaign_sha256=campaign['campaign_sha256'],
            source_schedule_sha256=campaign['source_schedule_sha256']), False
    expected = dict(slot, campaign_sha256=campaign['campaign_sha256'],
        source_schedule_sha256=campaign['source_schedule_sha256'])
    if (any(record.get(key) != value for key, value in expected.items())
            or not re.fullmatch(r'[a-zA-Z0-9-]+', record.get('run_id', ''))
            or not re.fullmatch(r'[0-9a-f]{40}', record.get('source_commit', ''))):
        raise ValueError('Existing source reservation changed task, cap or provenance')
    return record, True


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_preflight(path, *, source_commit, worker, hostname, seconds, now=None):
    """Admission is from this allocation's completed probe/tests, not old logs."""
    if path is None:
        raise ValueError('Allocated worker requires its own outer preflight receipt before any child reservation')
    value = json.loads(Path(path).read_text())
    probe = json.loads(Path(value['probe_path']).read_text())
    elapsed = (time.time() if now is None else now) - value['allocation_started_unix']
    if (value.get('source_commit') != source_commit or value.get('worker') != worker
            or value.get('hostname') != hostname or value.get('tests_exit_code') != 0
            or probe.get('status') != 'passed' or probe.get('purpose') != 'training'
            or len(probe.get('devices', [])) != 1
            or file_sha(value['probe_path']) != value['probe_sha256']
            or file_sha(value['tests_path']) != value['tests_sha256']
            or not 0 < value.get('maximum_outer_seconds', 0) <= MAX_WORKER_SECONDS
            or elapsed < 0 or seconds + elapsed > value['maximum_outer_seconds'] - 20):
        raise ValueError('Worker preflight source, GPU, tests or remaining allocation time differs')
    return value


def verify_worktree(path, source_commit):
    path = Path(path)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()
    if actual != source_commit or dirty or not (path / LAUNCHER).is_file():
        raise ValueError('Require the exact clean pinned source stream worktree')


def prepare_stream(project_root, repo, record, reserved):
    root = Path(project_root)
    path = root / 'worktrees' / record['run_id']
    if path.exists():
        verify_worktree(path, record['source_commit'])
    elif reserved:
        raise ValueError('Reserved source worktree is missing; recover it explicitly without migration')
    else:
        if (root / 'outputs' / record['run_id']).exists():
            raise ValueError('Unreserved run output already exists; never reuse its identity')
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-C', str(repo), 'worktree', 'add', '--detach',
                        str(path), record['source_commit']], check=True)
        verify_worktree(path, record['source_commit'])
    return str(path)


def action_charge(output):
    """Read actual RPC journals, including uncommitted interrupted episodes."""
    journal = Path(output) / 'experiment/raw/decision-journal'
    decisions = {path.stem: json.loads(path.read_text()) for path in journal.glob('decision-*.json')}
    known = sum(row['primitive_steps'] for row in decisions.values())
    uncertain = sum(json.loads(path.read_text())['maximum_primitive_steps']
        for path in journal.glob('intent-*.json')
        if 'decision-' + path.stem[7:] not in decisions)
    settling = sum(json.loads(path.read_text())['settling_steps'] for path in journal.glob('reset-*.json'))
    return dict(known_primitive_steps=known, uncertain_primitive_steps_ceiling=uncertain,
        settling_step_charge=settling, total_action_charge=known + uncertain + settling)


def collector_locked(output):
    path = Path(output) / 'experiment/run.lock'
    if not path.exists():
        return False
    with file_lock(path, blocking=False) as acquired:
        return not acquired


def stream_action(output, record, reserved, previous_worker=None):
    """Only positively identified failed recoveries may resume automatically."""
    output = Path(output)
    if not reserved:
        if output.exists():
            raise ValueError('New source stream collides with an existing output')
        return 'start'
    contract_path = output / 'experiment/contract.json'
    if contract_path.exists():
        contract = json.loads(contract_path.read_text())
        if (contract.get('source_commit') != record['source_commit']
                or contract.get('schedule_sha256') != record['source_schedule_sha256']
                or contract.get('task', {}).get('task_key') != record['task_key']
                or contract.get('partition') != record['partition']
                or contract.get('maximum_total_actions') != record['maximum_total_actions']
                or contract.get('source_only') is not True):
            raise ValueError('Source recovery contract differs from its pinned task/cap/commit')
    index_path = output / 'experiment/index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if (index.get('task_key') != record['task_key'] or index.get('partition') != record['partition']
                or index.get('maximum_total_actions') != record['maximum_total_actions']):
            raise ValueError('Source index does not match its immutable campaign reservation')
        if index.get('total_action_budget_charge', 0) > record['maximum_total_actions']:
            raise ValueError('Source action charge exceeds its immutable reservation')
        if index.get('completed') is True:
            return 'skip_completed' if index.get('collection_qualified') else 'skip_raw_only'
    if collector_locked(output):
        return 'skip_running'
    exit_path = output / 'exit-code.txt'
    failed = exit_path.exists() and int(exit_path.read_text().strip()) != 0
    if previous_worker is not None:
        if (previous_worker.get('run_id') != record['run_id']
                or previous_worker.get('source_commit') != record['source_commit']):
            raise ValueError('Worker status belongs to another source stream')
        failed |= previous_worker.get('status') == 'timed_out'
    if failed and contract_path.is_file() and (output / 'experiment/source-recovery.pkl').is_file():
        return 'resume'
    # A reservation without an exit receipt may be queued, starting or owned
    # outside this worker. Never guess that it has died from an absent lock.
    return 'skip_needs_review' if failed else 'skip_pending_or_unowned'


def child_command(project_root, record, action):
    if action not in ('start', 'resume'):
        raise ValueError('Only start or exact resume can execute a stream')
    suite, number = record['task_key'].split(':')
    command = ['bash', str(Path(project_root) / 'worktrees' / record['run_id'] / LAUNCHER),
        record['source_commit'], record['run_id'], suite, str(int(number)), record['partition']]
    if action == 'resume':
        command.append('--resume')
    return command


def observed_stream(output):
    row = action_charge(output)
    index = Path(output) / 'experiment/index.json'
    if index.exists():
        data = json.loads(index.read_text())
        row.update(collection_completed=data.get('completed') is True,
            collection_qualified=data.get('collection_qualified') is True,
            source_updates=data.get('source_updates'), query_chunks=data.get('chunks'),
            completed_episodes=len(data.get('episode_receipts', [])),
            case_coverage=data.get('case_coverage'))
    return row


def bounded_child(command, log, seconds):
    """Terminate only our child process group; preserve the RPC journal."""
    with Path(log).open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return process.returncode, timed_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation', choices=('prepare', 'run'))
    ap.add_argument('--project-root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--worker', type=int, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True, help='Fresh worker attempt receipt directory')
    ap.add_argument('--expected-hostname')
    ap.add_argument('--fresh-run-prefix', help='Only unreserved slots use this fresh identity; existing reservations are unchanged')
    ap.add_argument('--preflight-receipt', type=Path)
    ap.add_argument('--max-elapsed-seconds', type=int, default=MAX_WORKER_SECONDS)
    ap.add_argument('--child-timeout-seconds', type=int, default=2400)
    args = ap.parse_args(argv)
    if not re.fullmatch(r'[0-9a-f]{40}', args.source_commit):
        raise ValueError('Require the exact reviewed source commit')
    verify_worktree(Path(__file__).resolve().parents[3], args.source_commit)
    valid_prefix(args.fresh_run_prefix)
    if not 60 <= args.child_timeout_seconds <= 2400 or not 60 <= args.max_elapsed_seconds <= MAX_WORKER_SECONDS:
        raise ValueError('Worker ceilings are at most7140s total and40min per child within an approved2h allocation')
    if args.operation == 'run' and (sys.platform != 'linux' or socket.gethostname() != args.expected_hostname
                                   or args.project_root != DEFAULT_ROOT):
        raise ValueError('Execute only on the explicitly identified allocated Linux host and Baige project')
    preflight = None
    if args.operation == 'run':
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
        preflight = validate_preflight(args.preflight_receipt, source_commit=args.source_commit,
            worker=args.worker, hostname=args.expected_hostname, seconds=args.max_elapsed_seconds)
    started = time.monotonic()
    plan = args.project_root / 'outputs' / PLAN_ID
    campaign, schedule = load_campaign(plan / 'campaign.json'), load_schedule(plan / 'schedule.json')
    if campaign['source_schedule_sha256'] != schedule['schedule_sha256']:
        raise ValueError('Campaign and source schedule differ')
    slots = assigned_slots(campaign, args.worker, args.workers)
    args.output.mkdir(parents=True, exist_ok=False)
    contract = dict(format=FORMAT, operation=args.operation, source_commit=args.source_commit,
        campaign_sha256=campaign['campaign_sha256'], source_schedule_sha256=schedule['schedule_sha256'],
        worker=args.worker, workers=args.workers, slots=slots,
        max_elapsed_seconds=args.max_elapsed_seconds, child_timeout_seconds=args.child_timeout_seconds,
        fresh_run_prefix=args.fresh_run_prefix, allocation_preflight=preflight,
        compute_role='LIBERO approved32GPUh pool: this one of8 SAC workers uses at most2h; no RoboTwin allocation',
        action_cap='Each child retains its original campaign reservation; no new per-worker action budget')
    write_json(args.output / 'contract.json', contract)
    rows = []
    with file_lock(plan / 'worker-locks' / f'worker-{args.worker:02d}.lock', blocking=False) as owner:
        if not owner:
            raise RuntimeError('This campaign worker shard is already owned')
        for slot in slots:
            if time.monotonic() - started >= args.max_elapsed_seconds - 30:
                break
            with file_lock(plan / 'worker-locks' / (slot_name(slot) + '.lock'), blocking=False) as claimed:
                if not claimed:
                    rows.append(dict(slot=slot_key(slot), status='skip_worker_owned'))
                    continue
                reservations = read_ledger(plan / 'ledger.json', campaign)
                record, reserved = stream_record(slot, campaign, reservations, args.source_commit, args.fresh_run_prefix)
                output = args.project_root / 'outputs' / record['run_id']
                status_path = plan / 'worker-status' / (slot_name(slot) + '.json')
                previous = json.loads(status_path.read_text()) if status_path.exists() else None
                action = stream_action(output, record, reserved, previous)
                row = dict(slot=slot_key(slot), **record, status=action)
                if args.operation == 'prepare':
                    row['worktree'] = prepare_stream(args.project_root, args.project_root / 'repo', record, reserved)
                elif action in ('start', 'resume'):
                    worktree = args.project_root / 'worktrees' / record['run_id']
                    verify_worktree(worktree, record['source_commit'])
                    command = child_command(args.project_root, record, action)
                    seconds = min(args.child_timeout_seconds,
                                  int(args.max_elapsed_seconds - (time.monotonic() - started)) - 30)
                    if seconds < 60:
                        break
                    status_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(status_path, dict(row, status='running', worker_hostname=socket.gethostname(),
                        worker_output=str(args.output), command=command))
                    code, timeout = bounded_child(command, args.output / (slot_name(slot) + '.log'), seconds)
                    row.update(exit_code=code, status='timed_out' if timeout else 'exited', command=command)
                    row.update(observed_stream(output))
                    if row.get('collection_completed'):
                        row['status'] = 'completed' if row['collection_qualified'] else 'raw_only'
                    if row['total_action_charge'] > record['maximum_total_actions']:
                        raise RuntimeError('Child journal exceeded its immutable source reservation')
                    write_json(status_path, row)
                if 'total_action_charge' not in row:
                    row.update(observed_stream(output))
                if row['total_action_charge'] > record['maximum_total_actions']:
                    raise RuntimeError('Source journal exceeded its immutable campaign reservation')
                rows.append(row)
                write_json(args.output / 'progress.json', dict(contract=contract, streams=rows,
                    elapsed_seconds=time.monotonic() - started, completed=False))
        result = dict(contract=contract, streams=rows, elapsed_seconds=time.monotonic() - started,
            dispatch_complete=len(rows) == len(slots),
            all_streams_complete=len(rows) == len(slots) and all(row.get('collection_completed') for row in rows),
            all_source_histories_qualified=len(rows) == len(slots) and all(row.get('collection_qualified') for row in rows),
            note='Raw-only/partial streams are retained; saved files do not establish a competent teacher')
        write_json(args.output / 'index.json', result)
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
