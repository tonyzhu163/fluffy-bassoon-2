"""Read-only source campaign observations on the allocated Linux development host."""
import datetime
import json
import os
from pathlib import Path
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def tail(path, count=5000):
    try:
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - count))
            return stream.read().decode(errors='replace')
    except FileNotFoundError:
        return ''


def observe():
    now = time.time()
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    assert ledger['approved_GPU_hours'] == 70
    charged = reserved = remaining_storage = 0
    allocations = []
    for row in ledger['allocations']:
        began = datetime.datetime.fromisoformat(row.get('accounting_start_shanghai', row['created_shanghai'])).timestamp()
        ended = datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else now
        hours = row['GPUs'] * (ended - began) / 3600
        charged += hours
        reserved += hours if row.get('released_shanghai') else max(hours, row['max_reservation_GPU_hours'])
        output = RUN / row['output']
        worker = output / row.get('worker_subdir', 'node-00')
        manifest = read(output / 'manifest.json')
        if manifest:
            assert manifest['hostname'] == row['job_id'] + '-master-0'
        probe = read(output / 'gpu-probe.json') or read(worker / 'GPU-PROBE.json')
        written = row.get('closed_written_bytes') if row.get('released_shanghai') else None
        if written is None:
            written = 0
            for parent, _, files in os.walk(output, followlinks=False):
                for name in files:
                    path = Path(parent) / name
                    assert not path.is_symlink()
                    try:
                        written += path.stat().st_size
                    except FileNotFoundError:
                        pass
        if not row.get('released_shanghai'):
            allowance = row['storage_reservation_GiB'] * 2**30
            assert written <= allowance
            remaining_storage += allowance - written
        boundaries = []
        for checkpoint in sorted((output / row['task'] / 'checkpoints').glob('global_step_*')):
            marker = read(checkpoint / 'pilot-boundary.json')
            if marker:
                step = marker['global_step']
                exports = list((checkpoint / 'actor/model_state_dict').glob('full_weights.pt'))
                boundaries.append(dict(step=step, completed=marker['completed'],
                    worker_sidecars=len(marker['workers']), full_weight_exports=len(exports),
                    storage_schedule_correct=len(exports) == int(step == 40),
                    actor_bytes=sum(item['bytes'] for item in marker['actor_checkpoint_files'])))
        updates = []
        for file in sorted((output / 'updates').glob('iteration-*-rank-*.json'))[-2:]:
            updates.append(dict(path=str(file), record=read(file)))
        console = tail(output / 'console.log') or tail(worker / 'console.log')
        clock = read(output / 'clock.json') or read(worker / 'CLOCK.json')
        failed = clock is not None and clock.get('exit_code', 0) != 0
        errors = console if failed or any(word in console for word in ('Traceback', 'OutOfMemoryError', 'CUDA out of memory', 'RayTaskError', 'Exception occurred')) else None
        allocations.append(dict(**row, conservative_GPU_hours=hours, manifest=manifest,
            probe_passed=bool(probe and probe.get('status') == 'passed'),
            progress=read(output / 'progress.json') or read(worker / 'progress.json') or read(worker / 'PROGRESS.json'),
            clock=clock,
            result=read(output / 'RESULT.json') or read(worker / 'RESULT.json'),
            boundaries=boundaries, latest_actor_updates=updates, written_GiB=written/2**30,
            error_tail=errors, console_tail=console[-1600:]))
    pending = []
    for row in ledger.get('pending_submissions', []):
        output = RUN / row['output']
        worker = output / row.get('worker_subdir', '.')
        probe = read(worker / 'GPU-PROBE.json') or read(worker / 'gpu-probe.json')
        clock = read(worker / 'CLOCK.json') or read(worker / 'clock.json')
        cutoff = datetime.datetime.fromisoformat(row['last_confirmed_queued_shanghai']).timestamp()
        ended = clock.get('ended_unix', clock.get('finished_unix')) if clock else now
        started = (worker / 'console.log').exists()
        upper = row['GPUs'] * (ended-cutoff)/3600 if started else 0.
        charged += upper
        reserved += max(upper, row['max_reservation_GPU_hours'])
        written = sum(p.stat().st_size for p in output.rglob('*') if p.is_file() and not p.is_symlink())
        remaining_storage += max(0, row['storage_reservation_GiB']*2**30-written)
        pending.append(dict(**row, allocated_worker_artifacts=started, probe_passed=bool(probe and probe.get('status')=='passed'),
            clock=clock, progress=read(worker / 'PROGRESS.json') or read(worker / 'progress.json'), result=read(worker / 'RESULT.json'),
            conservative_GPU_hours_since_confirmed_queue_cutoff=upper))
    fs = os.statvfs(ROOT)
    free = fs.f_bavail * fs.f_frsize
    return dict(checked_shanghai=datetime.datetime.fromtimestamp(now,
        datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        cluster='Baige Pro6K', allocations=allocations, pending_submissions=pending,
        approved_GPU_hours=70, conservative_GPU_hours=charged,
        remaining_after_active_reservations_GPU_hours=70-reserved,
        storage_free_GiB=free/2**30,
        continuation_storage_reservation_GiB=110,
        continuation_required_GiB=330+remaining_storage/2**30,
        another_continuation_storage_admissible=free >= 330*2**30+remaining_storage,
        scheduler_release_verified=False)


if __name__ == '__main__':
    print(json.dumps(observe(), indent=2, allow_nan=False))
