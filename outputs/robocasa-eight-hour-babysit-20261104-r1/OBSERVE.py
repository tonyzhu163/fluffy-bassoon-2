"""Read-only observations; run on the allocated development container via SSH."""
import datetime
import json
import math
import os
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def observe():
    now = time.time()
    root = Path('/mowenjie/fluffy-bassoon-2')
    fit = root / 'outputs/robocasa-three-task-meta-baseline-20261104-r2'
    methods = {}
    for name in ('ad', 'dpt', 'dit', 'ic_iql'):
        directory = fit / 'joint/fits' / name
        metric = directory / 'metrics.jsonl'
        records = []
        if metric.exists():
            with metric.open('rb') as stream:
                stream.seek(max(0, metric.stat().st_size - 50000))
                lines = stream.read().decode(errors='replace').splitlines()
            for line in lines:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        checkpoint = directory / 'latest.pt'
        candidate = directory / 'candidate.pt'
        methods[name] = dict(status=read(directory / 'training-status.json'),
            latest_metric=records[-1] if records else None,
            recent_metrics_finite=bool(records) and finite(records),
            metric_age_seconds=round(now - metric.stat().st_mtime, 2) if metric.exists() else None,
            checkpoint_age_seconds=round(now - checkpoint.stat().st_mtime, 2) if checkpoint.exists() else None,
            candidate_bytes=candidate.stat().st_size if candidate.exists() else None)
    errors = {}
    for path in (fit / 'joint/fits').glob('*.log'):
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 6000))
            tail = stream.read().decode(errors='replace')
        if any(marker in tail for marker in ('Traceback', 'OutOfMemoryError', 'CUDA out of memory')):
            errors[path.name] = tail[-3000:]
    release = read(fit / 'joint/fit-4gpu-01/RELEASED.json')
    own = root / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
    ledger = read(own / 'META-ALLOCATION-LEDGER.json')
    if ledger is None:
        raise ValueError('Require the cumulative allocation ledger')
    upper = ledger['preparation_GPU_hours']
    reserved_upper = upper
    allocations = []
    for row in ledger['allocations']:
        created = datetime.datetime.fromisoformat(row['created_shanghai']).timestamp()
        ended = datetime.datetime.fromisoformat(row['released_shanghai']).timestamp() if row.get('released_shanghai') else now
        charged = row['GPUs'] * (ended-created) / 3600
        upper += charged
        reserved_upper += charged if row.get('released_shanghai') else max(charged, row['max_reservation_GPU_hours'])
        allocations.append(dict(row, conservative_GPU_hours=charged))
    baseline = {}
    for index in range(2):
        node = own / 'baseline-test512/attempt-01' / f'node-{index:02d}'
        proof = read(node / 'GPU-PROBE.json')
        complete_banks = sum(all((bank / (label + '.json')).exists() for label in ('SFT', 'PPO', 'GRPO'))
            for bank in (node / 'banks').glob('bank-*'))
        baseline[node.name] = dict(probe_passed=bool(proof and proof.get('status') == 'passed'),
            progress=read(node / 'progress.json'), complete_paired_cases=complete_banks * 8,
            complete=read(node / 'RESULT.json'), clock=read(node / 'CLOCK.json'))
        log = node / 'console.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                tail = stream.read().decode(errors='replace')
            if any(marker in tail for marker in ('Traceback', 'OutOfMemoryError', 'CUDA out of memory')):
                errors['baseline/' + node.name] = tail[-3000:]
    readouts = {}
    for row in ledger['allocations']:
        if not row.get('output'):
            continue
        directory = own / row['output']
        parts = {part.name:dict(progress=read(part / 'PROGRESS.json'),
            completed=bool(read(part / 'RESULT.json')), clock=read(part / 'CLOCK.json'))
            for part in directory.glob('node-*') if part.is_dir()}
        readouts[row['job_id']] = dict(output=row['output'], progress=read(directory / 'PROGRESS.json'),
            completed=bool(read(directory / 'RESULT.json')), clock=read(directory / 'CLOCK.json'),
            parts=parts, expected_parts=row.get('replicas'),
            workers_completed=bool(parts) and len(parts)==row.get('replicas') and all(part['completed'] for part in parts.values()))
        logs = [own / 'logs' / (directory.name + '.log')]
        logs += list((own / 'logs' / directory.name).glob('node-*.log'))
        for log in logs:
            if not log.exists():
                continue
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 6000))
                tail = stream.read().decode(errors='replace')
            if any(marker in tail for marker in ('Traceback', 'OutOfMemoryError', 'CUDA out of memory')):
                errors['readout/' + row['job_id'] + '/' + log.name] = tail[-3000:]
    return dict(checked_shanghai=datetime.datetime.fromtimestamp(now,
            datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        cluster='Baige Pro6K', job_id='job-967y1maorj80', learner_commit='d71cb532',
        active_fit_jobs=[row['job_id'] for row in ledger['allocations'] if row.get('kind')=='fit' and not row.get('released_shanghai')],
        methods=methods, worker_release=release, error_tails=errors,
        source_budget=dict(approved_GPU_hours=96, preparation_GPU_hours=17.57,
            allocations=allocations, creation_to_check_GPU_hours_upper=upper,
            remaining_GPU_hours_lower=96-upper,
            remaining_after_active_reservations_GPU_hours=96-reserved_upper,
            note='Creation through scheduler release; unreleased allocations charged through this check'),
        final_anchor=dict(job_id='job-icbta8h6gp49',
            progress=read(own / 'gpu/native-final-ID-qualification-01/PROGRESS.json'),
            clock=read(own / 'gpu/native-final-ID-qualification-01/CLOCK.json')),
        readout_jobs=readouts,
        baseline_test512=dict(job_id='job-ynq6hyyggfj6', incremental_cap_GPU_hours=7.5,
            nodes=baseline, complete_paired_cases=sum(node['complete_paired_cases'] for node in baseline.values())),
        storage_free_GiB=os.statvfs(root).f_bavail * os.statvfs(root).f_frsize / 2**30,
        scheduler_release_verified=False)


if __name__ == '__main__':
    print(json.dumps(observe(), indent=2, allow_nan=False))
