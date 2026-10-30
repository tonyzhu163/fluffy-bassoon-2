"""Metadata-only clock/report helpers for ONE bounded source batch14 probe.

Requires a root-created funding record. Creates no canonical reservation,
scheduler allocation or model. The actual trainer is vla_source40_batch14.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .vla_source40_ppo_worker import begin
from .vla_source40_schedule import content_hash


FORMAT = 'source40_one_batch14_qualification_budget_v1'
SCHEDULE_FILE_SHA = 'a2873aafbe7d047a301a8b4f6cb51e5a72b79b0af55c40607d691c0148acaeb5'
SCHEDULE_SHA = 'a59bbbbb4b6050cd4a330d395347dd4edc6257b55beebd5c6a91e896355a0716'
MAX_SECONDS, MAX_ACTIONS = 2400, 20000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def budget(path, schedule, run, source):
    value = json.loads(Path(path).read_text())
    record = value.get('reservations', {}).get(run, {})
    if (value.get('format') != FORMAT or value.get('funding_status') != 'root_reconciled'
            or value.get('maximum_gpu_seconds') != MAX_SECONDS or value.get('maximum_total_actions') != MAX_ACTIONS
            or value.get('source_campaign_immutable') is not True or not value.get('scope_amendment')
            or len(value.get('reservations', {})) != 1 or record.get('run_id') != run
            or record.get('source_commit') != source or len(source) != 40
            or any(char not in '0123456789abcdef' for char in source)
            or record.get('worker_seconds') != MAX_SECONDS or record.get('maximum_total_actions') != MAX_ACTIONS
            or record.get('purpose') != 'source_ppo_runtime_qualification'
            or record.get('updates') != 1 or record.get('complete_trials_per_update') != 14
            or record.get('attempts') != 4 or record.get('memory_mode') != 'full'
            or record.get('source_data_from_targets') is not False
            or record.get('schedule_file_sha256') != SCHEDULE_FILE_SHA
            or sha(schedule) != SCHEDULE_FILE_SHA
            or json.loads(Path(schedule).read_text()).get('schedule_sha256') != SCHEDULE_SHA):
        raise ValueError('Require an existing exact20k/2400s ONE-batch source-only funding record and canonical inventory')
    return record


def report(root, attempt, exit_code):
    from .vla_source40_ppo import physical_charge
    root, attempt = Path(root), Path(attempt)
    clock = json.loads((root/'worker-time.json').read_text())
    charge = sum(row.get('charged_seconds',row['reserved_seconds']) for row in clock['attempts'])
    experiment = root/'experiment'
    index = json.loads((experiment/'index.json').read_text()) if (experiment/'index.json').exists() else None
    summary = json.loads((experiment/'learner/summary.json').read_text()) if (experiment/'learner/summary.json').exists() else None
    metrics_path = experiment/'learner/metrics.jsonl'
    metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
    memory, samples = [], 0
    telemetry = attempt/'device-telemetry.csv'
    if telemetry.exists():
        for row in csv.reader(telemetry.read_text().splitlines()):
            if len(row) == 4:
                try:
                    memory.append(float(row[2].strip()))
                    samples += 1
                except ValueError:
                    pass
    recovery = experiment/'learner/latest.pt'
    signatures = []
    for path in sorted((experiment/'attempts').glob('*/parameters-after.json')):
        signatures.append(dict(path=str(path),sha256=sha(path),**json.loads(path.read_text())))
    latest = signatures[-1] if signatures else {}
    before, after = latest.get('motor_before'), latest.get('motor_after')
    value = dict(format='source40_one_batch14_qualification_receipt_v1',exit_code=int(exit_code),
        cumulative_gpu_charge_seconds=charge,maximum_gpu_seconds=MAX_SECONDS,
        actual_physical_cost=physical_charge(experiment/'execution-journal'),
        batch_summary=summary, update_receipts=metrics,
        full_recovery_path=str(recovery),full_recovery_sha256=sha(recovery) if recovery.exists() else None,
        sampled_whole_device_peak_memory_mib=max(memory) if memory else None,
        whole_device_telemetry_samples=samples,whole_device_telemetry_sha256=sha(telemetry) if telemetry.exists() else None,
        frozen_motor_parameters_unchanged=latest.get('unchanged'),
        motor_parameters_before_sha256=None if before is None else before['sha256'],
        motor_parameters_after_sha256=None if after is None else after['sha256'],
        motor_parameter_receipts=signatures,
        memory_limit='1Hz NVML samples include frozen motor/rendering/PPO; Torch exact replay/update peaks are separate update metrics',
        interpretation='Runtime/capacity/recovery qualification for one gradient batch; no robot-performance or memory benefit established')
    (attempt/'qualification-receipt.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    start=sub.add_parser('begin')
    for name in ('root','budget','schedule'):
        start.add_argument('--'+name,type=Path,required=True)
    for name in ('source','run'):
        start.add_argument('--'+name,required=True)
    start.add_argument('--started-unix',type=float,required=True)
    start.add_argument('--resume',action='store_true')
    end=sub.add_parser('report')
    for name in ('root','attempt'):
        end.add_argument('--'+name,type=Path,required=True)
    end.add_argument('--exit-code',type=int,required=True)
    args=parser.parse_args(argv)
    if args.operation=='report':
        report(args.root,args.attempt,args.exit_code)
        return
    record=budget(args.budget,args.schedule,args.run,args.source)
    contract=dict(operation='source40_one_batch14_qualification',source_commit=args.source,run_id=args.run,
        funding_record_sha256=content_hash(record),schedule_file_sha256=SCHEDULE_FILE_SHA,
        schedule_sha256=SCHEDULE_SHA,maximum_total_actions=MAX_ACTIONS,maximum_worker_seconds=MAX_SECONDS,
        updates=1,complete_trials_per_update=14,attempts=4,memory_mode='full',seed=2,
        architecture=dict(width=128,layers=2,heads=4,context_length=300),optimizer=dict(learning_rate=1e-4,epochs=2))
    path=args.root/'worker-time.json'
    charged=(sum(row.get('charged_seconds',row['reserved_seconds']) for row in json.loads(path.read_text())['attempts'])
        if path.exists() else 0)
    slice_seconds=MAX_SECONDS-charged-60
    attempt=begin(args.root,contract,slice_seconds=slice_seconds,worker_seconds=MAX_SECONDS,
        resume=args.resume,now=args.started_unix)
    print(f'{attempt}\t{int(args.started_unix+slice_seconds)}\t{slice_seconds}')


if __name__=='__main__':
    main()
