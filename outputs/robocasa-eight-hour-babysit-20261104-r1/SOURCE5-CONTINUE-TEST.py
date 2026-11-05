"""Manually recover complete fixed-case banks without changing the qualified executor."""
import argparse
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


def prepare(task, output):
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    original = RUN / 'source5-tests' / task / 'attempt-01'
    ledger = read(RUN / 'SOURCE5-ALLOCATION-LEDGER.json')
    row = next(row for row in ledger['allocations'] if RUN / row['output'] == original)
    assert row['scheduler_release_verified'] and row['released_shanghai'] and row['kind'] == 'endpoint_test'
    console = (original / 'node-00/console.log').read_text(errors='replace')
    clock = read(original / 'node-00/CLOCK.json')
    assert clock['exit_code'] != 0 and ('Insufficient time for next full bank' in console or clock['exit_code'] == 124)
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
    assert 0 in retained and 1 <= len(discarded) <= 4
    assert output.parent == original.parent and not output.exists()
    output.mkdir()
    manifest = original / 'MANIFEST.json'
    write(output / 'PLAN.json', dict(task=task, original=str(original), original_job_id=row['job_id'],
        manifest=str(manifest), manifest_sha256=sha(manifest), retained_banks=retained,
        remaining_banks=discarded, original_panel_sha256=hashes,
        maximum_GPU_hours=.4, scheduler_minutes=24, command_seconds=1260,
        storage_reservation_GiB=5, scientific_comparison_unchanged=True,
        policy='Retain complete pairs only; rerun incomplete banks without inspecting outcomes',
        automatic_retry=False, SGD=0))


def bank(plan_path, number, deadline):
    plan = read(plan_path)
    assert number in plan['remaining_banks'] and plan['scientific_comparison_unchanged']
    manifest = Path(plan['manifest'])
    assert sha(manifest) == plan['manifest_sha256']
    output = plan_path.parent / 'workers' / f'bank-{number:03d}'
    output.mkdir(parents=True)
    native = fixed_test().configure(plan['task'])
    # The qualified executor's partition is range(index,64,workers).
    # index=b,workers=64 selects exactly the original bank b, with no remapping.
    try:
        native.execute(manifest, number, 64, output, deadline)
    finally:
        import ray
        ray.shutdown()


def run(plan_path, deadline):
    plan = read(plan_path)
    assert 1 <= len(plan['remaining_banks']) <= 4
    for number in plan['remaining_banks']:
        subprocess.run([sys.executable, '-B', str(HERE / 'SOURCE5-CONTINUE-TEST.py'), 'bank',
            '--plan', str(plan_path), '--bank', str(number), '--deadline', str(deadline)], check=True)
    write(plan_path.parent / 'node-00/RESULT.json', dict(completed=True,
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
    assert read(plan_path.parent / 'node-00/RESULT.json')['completed']
    for number in plan['remaining_banks']:
        worker = plan_path.parent / 'workers' / f'bank-{number:03d}'
        done = read(worker / 'RESULT.json')
        assert done['completed'] and done['banks'] == [number] and done['policy_controls'] == 9600
        assert done['manifest_sha256'] == plan['manifest_sha256']
        sources.append(source(worker, job))
        bank_sources[str(number)] = len(sources)-1
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
    parser.add_argument('mode', choices=('prepare', 'run', 'bank', 'assemble', 'qualify', 'promote'))
    parser.add_argument('--task', choices=('PickPlaceCounterToCabinet', 'TurnOnSinkFaucet'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--bank', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.task, args.output)
    elif args.mode == 'run':
        run(args.plan, args.deadline)
    elif args.mode == 'bank':
        bank(args.plan, args.bank, args.deadline)
    elif args.mode == 'assemble':
        assemble(args.plan, args.job, args.output)
    elif args.mode == 'qualify':
        qualify(args.task, args.output)
    else:
        promote(args.task, args.output)
