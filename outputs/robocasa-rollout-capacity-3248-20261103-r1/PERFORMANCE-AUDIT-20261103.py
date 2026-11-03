"""Read existing paused banks on allocated Linux CPUs; no model or simulator calls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import statistics
import sys
import time


BASE = Path('/mowenjie/fluffy-bassoon-2/outputs')
OUT = BASE / 'robocasa-rollout-capacity-3248-20261103-r1/performance-20261103-r1'
REFERENCES = []


def reference(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    record = dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())
    REFERENCES.append(record)
    return record


def read(path):
    reference(path)
    return json.loads(path.read_text())


def describe(values):
    values = sorted(values)
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), p90=values[int(.9 * (len(values) - 1))], maximum=max(values))


def union_seconds(intervals):
    total, end = 0., float('-inf')
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end))
        end = max(end, right)
    return total


def journals(root):
    groups = {}
    for rank in range(8):
        folder = root / 'accounting' / f'rank-{rank:02d}'
        if not folder.exists():
            continue
        for path in folder.glob('*.json'):
            data = json.loads(path.read_text())
            if not data.get('resolved') or data.get('phase') != 'policy_controls':
                continue
            key = data['scope']
            if key not in ('train-001', 'train-002', 'qualification-stress-velocity',
                           'qualification-stress-kv_all_layers'):
                continue
            groups.setdefault((key, rank), []).append((data['started_unix'], data['ended_unix']))
    return {scope: {str(rank): dict(durations=describe([b-a for a,b in intervals]),
        first=min(a for a,b in intervals), last=max(b for a,b in intervals),
        interval_union_seconds=union_seconds(intervals)) for (key,rank), intervals in groups.items()
        if key == scope} for scope in sorted({key for key,rank in groups})}


def artifact_sizes(root):
    counts, sizes = Counter(), Counter()
    for folder, dirs, files in os.walk(root):
        for filename in files:
            path = Path(folder) / filename
            kind = 'raw_chunks' if filename.endswith('.pkl.gz') else (
                'videos' if filename.endswith('.mp4') else 'other')
            counts[kind] += 1
            sizes[kind] += path.stat().st_size
    return dict(files=dict(counts), bytes=dict(sizes))


def main():
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert socket.gethostname() == 'aibox-rffe453e3884-5f96c78b88-rtvhb'
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError('CPU audit deadline')))
    signal.alarm(330)
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    report = dict(format='paused_scale50_readonly_performance_v1', recorded_UTC=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), python=platform.python_version(), torch=torch.__version__,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], actual_new_episodes=0,
        new_optimizer_steps=0, GPU_jobs_launched_or_modified=0, scheduler_accounting_owned_elsewhere=True,
        qualification={}, arms={})
    q = BASE / 'robocasa-joint-steering-scale50-native-qualification-20261103-r2/stress'
    qualification = read(q / 'RESULT.json')
    report['qualification']['bindings'] = qualification['bindings']
    report['qualification']['arms'] = {arm: {key: row[key] for key in ('elapsed_seconds', 'execution',
        'controls_per_second', 'optimizer_config', 'interface', 'sampled_phase_peaks')}
        for arm, row in qualification['arms'].items()}
    report['qualification']['dispatch_journals'] = journals(q)
    reference(q / 'resources.jsonl')
    resource_rows = [json.loads(line) for line in (q / 'resources.jsonl').read_text().splitlines()]
    report['qualification']['resource_summaries'] = {phase: dict(samples=len(rows),
        GPU_utilization_percent=describe([x['gpu_utilization_percent'] for x in rows]),
        CPU_quota=rows[0]['cgroup']['cpu.max'], memory_limit=rows[0]['cgroup']['memory.max'])
        for phase in ('stress_48_live/velocity', 'stress_48_live/kv_all_layers')
        if (rows := [x for x in resource_rows if x['phase'] == phase and 'gpu_utilization_percent' in x])}
    for short, arm in [('velocity', 'velocity'), ('kv', 'kv_all_layers')]:
        root = BASE / f'robocasa-joint-steering-{short}-scale50-20261103-r1'
        profile = read(root / 'PROFILE.json')
        clock = read(root / 'clock.json')
        paused = read(root / 'PAUSED.json')
        assert paused['iteration'] == 2 and paused['reason'] == 'measured_budget_decision'
        row = dict(profile=profile, clock=clock, pause=paused, phases=[], rank_banks=[])
        for path in sorted((root / 'metrics').glob('event-*.json')):
            data = read(path)
            row['phases'].append({key: data.get(key) for key in ('phase','iteration','sequence',
                'last_check_unix','elapsed_seconds','collection_and_learning_seconds',
                'physical_controls_this_event','projected_remaining_seconds','projected_all_attempt_gpu_hours')}
                | dict(optimizer_seconds=data.get('ppo',{}).get('elapsed_seconds'),
                    optimizer_steps=data.get('ppo',{}).get('optimizer_steps'), resources=data.get('resources')))
        for sequence in (4, 5):
            for rank in range(8):
                worker = root / 'workers' / f'rank-{rank:02d}'
                result_path = worker / f'result-{sequence:06d}.pt'
                request_path = worker / f'request-{sequence:06d}.pt'
                reference(result_path); reference(request_path)
                result = torch.load(result_path, map_location='cpu', weights_only=False)
                request = torch.load(request_path, map_location='cpu', weights_only=False)
                batches = result['execution']['native_batch_sizes']
                assert sum(batches) == 48 * 12
                assert all(episode['horizon'] == 600 for episode in result['episodes'])
                initial_times = [Path(episode['initial']['path']).stat().st_mtime for episode in result['episodes']]
                bank = dict(rank=rank, iteration=sequence-3, request_mtime=request_path.stat().st_mtime,
                    completed_unix=result['completed_unix'], result_mtime=result_path.stat().st_mtime,
                    worker_seconds=result['completed_unix']-request_path.stat().st_mtime,
                    initial_reset_acceptance_checkpoint_times=describe(initial_times),
                    accepted_reset_checkpoint_span_seconds=max(initial_times)-min(initial_times),
                    initial_barrier=result['execution']['initial_lane_barrier'], native_calls=len(batches),
                    average_native_batch=statistics.mean(batches), full_48_calls=batches.count(48),
                    batches=batches, task_counts=dict(Counter(case['task'] for case in request['cases'])),
                    policy_config=request['policy_config'], interface=request['spec'],
                    native_device_name=result['native_model_frozen_signature']['device_name'])
                if sequence == 5:
                    bank['artifact_sizes'] = artifact_sizes(worker / f'bank-{sequence:06d}')
                row['rank_banks'].append(bank)
        row['dispatch_journals'] = journals(root)
        # A provisional sensitivity calculation, not the source owner's terminal charge.
        # 13:14:00 is the already recorded scheduler allocation timestamp for both jobs.
        allocation_start = 1793682840
        last = row['phases'][-1]
        dev = next(x['elapsed_seconds'] for x in row['phases'] if x['phase'] == 'dev')
        final = next(x['elapsed_seconds'] for x in row['phases'] if x['phase'] == 'test_before')
        prior = profile['prior_conservative_gpu_seconds']
        spent = prior + 8 * (clock['finished_unix'] - allocation_start)
        future = 48 * last['elapsed_seconds'] + 10 * dev + final
        row['cost_sensitivity'] = dict(allocation_start_assumption_unix=allocation_start,
            terminal_scheduler_release_charge_pending_owner=True,
            provisional_spent_through_wrapper_exit_GPU_hours=spent/3600,
            wrongly_charged_zero_GPU_queue_seconds=allocation_start-profile['submitted_unix'],
            wrongly_charged_queue_GPU_hours=8*(allocation_start-profile['submitted_unix'])/3600,
            recorded_guard_projection_GPU_hours=last['projected_all_attempt_gpu_hours'],
            allocation_corrected_guard_projection_GPU_hours=last['projected_all_attempt_gpu_hours']-
                8*(allocation_start-profile['submitted_unix'])/3600,
            remaining_train_updates=48, remaining_DEV_panels=10, remaining_final_panels=1,
            remaining_train_GPU_hours=48*last['elapsed_seconds']*8/3600,
            remaining_DEV_and_final_GPU_hours=(10*dev+final)*8/3600,
            remaining_work_GPU_hours_before_restart=future*8/3600,
            train_update_seconds_to_fit_with_240s_restart_and_120s_release=
                ((40*3600-spent)/8-10*dev-final-240-120)/48,
            scenarios=[dict(train_update_seconds=t,
                total_with_240s_restart_and_120s_release_GPU_hours=
                    (spent+8*(48*t+10*dev+final+240+120))/3600)
                for t in (222, 250, 275, 300, 310, 315, 325, last['elapsed_seconds'])])
        report['arms'][arm] = row
    report['references'] = REFERENCES
    report['elapsed_seconds'] = time.monotonic()-started
    path = OUT / 'AUDIT.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elapsed_seconds=report['elapsed_seconds'],costs={arm:row['cost_sensitivity'] for arm,row in report['arms'].items()}),indent=2))


if __name__ == '__main__':
    main()
