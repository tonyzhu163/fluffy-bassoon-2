"""Manually recover complete fixed-case banks without changing the qualified executor."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).parent
RUN = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def fixed_test():
    spec = importlib.util.spec_from_file_location('fixed_source5_test', HERE / 'SOURCE5-TEST.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def labels(bank):
    return (['SFT', 'teacher'] if bank % 2 == 0 else ['teacher', 'SFT']) + (['repeat'] if bank == 0 else [])


def allocation(job):
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    assert ledger['approved_GPU_hours'] == 70
    return next(row for row in ledger['allocations'] + ledger.get('pending_submissions', []) if row['job_id'] == job)


def prepare(task, output, maximum_gpu_hours=1.2):
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    original = RUN / 'source5-tests' / task / 'attempt-01'
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    row = next(row for row in ledger['allocations'] if RUN / row['output'] == original)
    assert row['scheduler_release_verified'] and row['released_shanghai'] and row['kind'] == 'endpoint_test'
    console = (original / 'node-00/console.log').read_text(errors='replace')
    clock = read(original / 'node-00/CLOCK.json')
    storage_stop = 'assert fs.f_bavail*fs.f_frsize > 200 << 30' in console
    assert clock['exit_code'] != 0 and (storage_stop or 'Insufficient time for next full bank' in console or clock['exit_code'] == 124)
    retained, discarded, hashes = [], [], {}
    for bank in range(64):
        base = original / 'node-00/banks' / f'bank-{bank:03d}'
        files = [base / f'{label}.json' for label in labels(bank)]
        complete = all(path.exists() for path in files)
        (retained if complete else discarded).append(bank)
        for path in files:
            if path.exists():
                panel = read(path)
                assert panel['common_sampler'] and sha(panel['episodes_path']) == panel['episodes_sha256']
                assert read(panel['episodes_path'])['completed']
                hashes[str(path)] = sha(path)
    assert maximum_gpu_hours in (.5, 1.2)
    assert 0 in retained and 1 <= len(discarded) <= (8 if maximum_gpu_hours == .5 else 24)
    assert output.parent == original.parent and not output.exists()
    output.mkdir()
    manifest = original / 'MANIFEST.json'
    execution = deepcopy(read(manifest))
    execution['sequential_order'] = [labels(bank) if bank in discarded else [] for bank in range(64)]
    execution_path = output / 'EXECUTION-MANIFEST.json'
    write(execution_path, execution)
    write(output / 'PLAN.json', dict(task=task, original=str(original), original_job_id=row['job_id'],
        manifest=str(manifest), manifest_sha256=sha(manifest), retained_banks=retained,
        remaining_banks=discarded, original_panel_sha256=hashes,
        execution_manifest=str(execution_path), execution_manifest_sha256=sha(execution_path),
        stop_reason='shared_storage_guard' if storage_stop else 'allocation_deadline',
        maximum_GPU_hours=maximum_gpu_hours,
        scheduler_minutes=30 if maximum_gpu_hours == .5 else 72,
        command_seconds=1620 if maximum_gpu_hours == .5 else 4140,
        storage_reservation_GiB=5, scientific_comparison_unchanged=True,
        policy='Retain complete pairs only; rerun incomplete banks without inspecting outcomes',
        automatic_retry=False, SGD=0))


def run(plan_path, deadline):
    plan = read(plan_path)
    assert 1 <= len(plan['remaining_banks']) <= 24 and plan['scientific_comparison_unchanged']
    manifest = Path(plan['manifest'])
    assert sha(manifest) == plan['manifest_sha256']
    execution_path = Path(plan['execution_manifest'])
    assert sha(execution_path) == plan['execution_manifest_sha256']
    original, execution = read(manifest), read(execution_path)
    assert {k:v for k,v in original.items() if k != 'sequential_order'} == {
        k:v for k,v in execution.items() if k != 'sequential_order'}
    output = plan_path.parent / 'node-00'
    assert output.is_dir() and not (output / 'RESULT.json').exists()
    native = fixed_test().configure(plan['task'])
    original_order = native.order
    selected = set(plan['remaining_banks'])
    assert original['sequential_order'] == [original_order(bank) for bank in range(64)]
    native.order = lambda bank: original_order(bank) if bank in selected else []
    assert execution['sequential_order'] == [native.order(bank) for bank in range(64)]
    # Only the per-process scheduling table changes. Original bank indices,
    # CASE_BASE/RNG_BASE, policies, batch width and the native executor stay fixed.
    try:
        native.execute(execution_path, 0, 1, output, deadline)
    finally:
        native.order = original_order
        import ray
        ray.shutdown()
    done = read(output / 'RESULT.json')
    assert done['completed'] and done['panels'] == 2*len(selected)
    assert done['policy_controls'] == 9600*len(selected)
    assert done['manifest_sha256'] == plan['execution_manifest_sha256']
    write(output / 'RECOVERY-RESULT.json', dict(completed=True,
        banks=plan['remaining_banks'], plan_sha256=sha(plan_path), SGD=0))


def source(worker, job, *, qualification=False):
    row = allocation(job)
    root = RUN / row['output']
    assert worker.is_relative_to(root) and row['GPUs'] == 1
    if not qualification:
        assert row['scheduler_release_verified'] and row['released_shanghai']
    probe = root / row.get('worker_subdir', 'node-00') / 'GPU-PROBE.json'
    parity = worker / 'initial-parity.json'
    assert read(probe)['status'] == 'passed' and read(probe)['hostname'] == job + '-master-0'
    assert len(read(probe)['devices']) == 1 and read(parity)[0]['passed']
    record = dict(worker=str(worker), job_id=job,
        probe=dict(path=str(probe), sha256=sha(probe)), parity=dict(path=str(parity), sha256=sha(parity)))
    if not qualification:
        journal = worker / 'accounting/rank-00' / (worker.name + '.json')
        record.update(journal=dict(path=str(journal), sha256=sha(journal)),
                      scheduler_released_shanghai=row['released_shanghai'])
    return record


def collect(task, manifest, output, sources, bank_sources, *, qualification=False):
    assert not output.exists() and output.parent == RUN / 'source5-tests' / task
    output.mkdir()
    (output / 'MANIFEST.json').write_bytes(manifest.read_bytes())
    m = read(manifest)
    assert m['task'] == task and m['cases'] == 512
    assert set(bank_sources) == ({'0'} if qualification else {str(b) for b in range(64)})
    worker = output / 'node-00'
    (worker / 'banks').mkdir(parents=True)
    (worker / 'sampler').mkdir()
    panel_hashes, selected = {}, set()
    for key, index in bank_sources.items():
        number = int(key)
        original = Path(sources[index]['worker']) / 'banks' / f'bank-{number:03d}'
        assert original.is_dir()
        (worker / 'banks' / original.name).symlink_to(original, target_is_directory=True)
        for label in labels(number):
            path = original / f'{label}.json'
            panel = read(path)
            assert panel['common_sampler'] and sha(panel['episodes_path']) == panel['episodes_sha256']
            assert panel['load']['sha256'] == m['policies']['SFT' if label == 'repeat' else label]['sha256']
            panel_hashes[str(path)] = sha(path)
            selected.add((index, panel['phase']))
            for chunk in range(1, 13):
                sample = Path(sources[index]['worker']) / 'sampler' / f'{panel["phase"]}-{chunk:02d}.json'
                assert sample.is_file()
                (worker / 'sampler' / sample.name).symlink_to(sample)
    collection = dict(format='source5.complete-bank-collection.v1', task=task,
        qualification_only=qualification, manifest_sha256=sha(manifest), sources=sources,
        bank_sources=bank_sources, panel_sha256=panel_hashes, scientific_comparison_unchanged=True)
    write(output / 'COLLECTION.json', collection)
    if qualification:
        return
    calls = []
    for index, row in enumerate(sources):
        for call_index, call in enumerate(read(row['journal']['path'])['calls']):
            calls.append(dict(call, source_index=index, source_call_index=call_index,
                executed_or_conservative=call.get('executed_or_conservative', call['requested']),
                selected_for_readout=(index, call['phase'].split('/')[0]) in selected))
    assert all(call['resolved'] for call in calls if call['selected_for_readout'])
    charged = sum(call['executed_or_conservative'] for call in calls)
    write(worker / 'accounting/rank-00/node-00.json', dict(format='source5.accounting-collection.v1', calls=calls))
    write(worker / 'RESULT.json', dict(completed=True, kind='verified_artifact_collection',
        banks=list(range(64)), policy_controls=619200, charged_controls=charged,
        manifest_sha256=sha(manifest), input_collection_sha256=sha(output / 'COLLECTION.json'), SGD=0))
    fixed_test().combine(task, output / 'MANIFEST.json', output)


def assemble(plan_path, job, output):
    plan = read(plan_path)
    original = Path(plan['original'])
    for path, digest in plan['original_panel_sha256'].items():
        assert sha(path) == digest
    sources = [source(original / 'node-00', plan['original_job_id'])]
    bank_sources = {str(b): 0 for b in plan['retained_banks']}
    worker = plan_path.parent / 'node-00'
    assert read(worker / 'RECOVERY-RESULT.json')['completed']
    done = read(worker / 'RESULT.json')
    assert done['completed'] and done['policy_controls'] == 9600*len(plan['remaining_banks'])
    assert done['manifest_sha256'] == plan['execution_manifest_sha256']
    added = source(worker, job)
    added['execution_manifest'] = dict(path=plan['execution_manifest'], sha256=plan['execution_manifest_sha256'])
    sources.append(added)
    for number in plan['remaining_banks']:
        bank_sources[str(number)] = 1
    collect(plan['task'], Path(plan['manifest']), output, sources, bank_sources)


def qualify(task, output):
    original = RUN / 'source5-tests' / task / 'attempt-01'
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    job = next(row['job_id'] for row in ledger['allocations'] if RUN / row['output'] == original)
    collect(task, original / 'MANIFEST.json', output,
        [source(original / 'node-00', job, qualification=True)], {'0': 0}, qualification=True)


def promote(task, output):
    assert output.parent == RUN / 'source5-tests' / task
    audit = read(output / 'INDEPENDENT-AUDIT.json')
    assert audit['verified'] and not audit['qualification_only'] and audit['cases_per_policy'] == 512
    assert audit['combined_sha256'] == sha(output / 'RESULT.json')
    assert audit['input_collection_sha256'] == sha(output / 'COLLECTION.json')
    write(output.parent / 'ENDPOINT-RESULT-SOURCE.json', dict(task=task, path=str(output), cases=512,
        manifest_sha256=sha(output / 'MANIFEST.json'), scientific_comparison_unchanged=True,
        collection_sha256=sha(output / 'COLLECTION.json'), audit_sha256=sha(output / 'INDEPENDENT-AUDIT.json'),
        selection='Complete original banks plus repaired incomplete banks; no outcome-based selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run', 'assemble', 'qualify', 'promote'))
    parser.add_argument('--task', choices=('PickPlaceCounterToCabinet', 'TurnOnSinkFaucet'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--maximum-gpu-hours', type=float, choices=(.5, 1.2), default=1.2)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.task, args.output, args.maximum_gpu_hours)
    elif args.mode == 'run':
        run(args.plan, args.deadline)
    elif args.mode == 'assemble':
        assemble(args.plan, args.job, args.output)
    elif args.mode == 'qualify':
        qualify(args.task, args.output)
    else:
        promote(args.task, args.output)
