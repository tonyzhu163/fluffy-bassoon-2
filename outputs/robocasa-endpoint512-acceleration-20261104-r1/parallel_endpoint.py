"""Resume disjoint complete case banks through the unchanged qualified evaluator.

This run changes scheduling only. The original evaluator, source manifests,
eight-lane batches, ABBA policy order, case/noise seeds and statistics are reused.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PARENT = 'robocasa-flow-sde-endpoint-test512-20261104-r1'
RUN = Path(__file__).parent.name
TASKS = ['CloseToasterOvenDoor', 'OpenDrawer']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def endpoint(metadata):
    script = Path(metadata) / 'outputs' / PARENT / 'endpoint.py'
    spec = importlib.util.spec_from_file_location('qualified_endpoint', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_banks(banks, workers=4):
    q, r = divmod(len(banks), workers)
    result, offset = [], 0
    for worker in range(workers):
        count = q + (worker < r)
        result.append(banks[offset:offset + count])
        offset += count
    assert sorted(b for group in result for b in group) == banks
    return result


def prepare(metadata, original, output, stopped, previous=None, scheduler_seconds=2100, workers=4):
    assert not output.exists(), 'Use a new migration identity'
    assert stopped['scheduler_released'] and len(stopped['jobs']) == 2
    ep = endpoint(metadata)
    prepared = Path(metadata) / 'outputs' / PARENT / 'PREPARED.json'
    tasks, shards, prior_charge = {}, [], stopped['all_attempt_gpu_seconds_upper']
    for task in TASKS:
        _, row = ep.bind(prepared, task)
        root = Path(original) / task / 'gpu-r1'
        panels = [read(p) for p in sorted((root / 'panels').glob('*.json'))]
        abandoned = []
        if previous is not None:
            old = read(previous)
            assert old['prepared_sha256'] == sha(prepared)
            assert old['endpoint_sha256'] == sha(ep.HERE)
            # The previous manifest can already include multiple released
            # attempts; preserve its canonical retained pairs recursively.
            panels = deepcopy(old['tasks'][task]['retained_panels'])
            abandoned = deepcopy(old['tasks'][task]['discarded_incomplete_panels'])
            for shard in old['shards']:
                if shard['task'] != task:
                    continue
                paths = sorted((Path(shard['output']) / 'panels').glob('*.json'))
                source_banks = {}
                for path in paths:
                    panel = read(path)
                    source_banks.setdefault(panel['bank'], set()).add(panel['step'])
                complete_here = {b for b, steps in source_banks.items() if steps == {0, 130}}
                for path in paths:
                    panel = read(path)
                    panel['source_panel_path'] = str(path)
                    panel['source_panel_sha256'] = sha(path)
                    panel['source_local_bank'] = panel['bank']
                    panel['bank'] = shard['banks'][panel['bank']]
                    (panels if panel['source_local_bank'] in complete_here else abandoned).append(panel)
        by_bank = {}
        for panel in panels:
            assert panel['seeds'] == row['seeds'][panel['bank'] * 8:(panel['bank'] + 1) * 8]
            assert panel['rng_seed'] == row['rng_seeds'][panel['bank']]
            assert sha(panel['episode_path']) == panel['episode_sha256']
            entries = by_bank.setdefault(panel['bank'], {})
            assert panel['step'] not in entries, 'Repeated retained case/policy bank'
            entries[panel['step']] = panel
        complete = sorted(b for b, values in by_bank.items() if set(values) == {0, 130})
        retained = [p for p in panels if p['bank'] in complete]
        remaining = [b for b in range(64) if b not in complete]
        tasks[task] = dict(retained_banks=complete, retained_panels=retained,
            discarded_incomplete_panels=abandoned + [p for p in panels if p['bank'] not in complete],
            remaining_banks=remaining)
        for index, banks in enumerate(split_banks(remaining, min(workers, len(remaining))) if remaining else []):
            assert banks
            shards.append(dict(task=task, shard=index, banks=banks,
                # native_eval puts accounting in output.parent/accounting.
                # Give each shard a private parent, including unequal lengths.
                output=str(output / task / f'shard-{index:02d}' / 'gpu-r1'),
                original_staging=str(Path(original) / task / 'staging-r1')))
    # Scheduler creation-to-release costs, including old failed attempts, remain charged.
    assert shards and prior_charge + len(shards) * scheduler_seconds <= 27600, 'Migration reservation exceeds original GPU-hour ceiling'
    manifest = dict(completed=True, run=RUN, metadata=str(metadata),
        original=str(original), prepared=str(prepared), prepared_sha256=sha(prepared),
        endpoint_sha256=sha(ep.HERE), tasks=tasks, shards=shards,
        stopped=stopped, maximum_gpus=len(shards), budget_gpu_seconds=27600,
        reserved_gpu_seconds_upper=prior_charge + len(shards) * scheduler_seconds,
        scheduler_seconds=scheduler_seconds,
        previous_manifest=str(previous) if previous else None,
        previous_manifest_sha256=sha(previous) if previous else None,
        original_maximum_policy_controls=1228800,
        recovery_control_reserve=96000 if previous else 19200,
        maximum_all_attempt_policy_controls=1324800 if previous else 1248000,
        scientific_comparison_unchanged=True, fixed_batch=8, SGD=0)
    write(output / 'MIGRATION.json', manifest)
    return manifest


def worker(manifest_path, index, deadline):
    manifest = read(manifest_path)
    spec = manifest['shards'][index]
    output = Path(spec['output'])
    assert not output.exists()
    output.mkdir(parents=True)
    ep = endpoint(manifest['metadata'])
    assert sha(ep.HERE) == manifest['endpoint_sha256']
    prepared = Path(manifest['prepared'])
    assert sha(prepared) == manifest['prepared_sha256']
    value, original_row = ep.bind(prepared, spec['task'])
    row = deepcopy(original_row)
    row['seeds'] = [seed for bank in spec['banks'] for seed in original_row['seeds'][bank * 8:(bank + 1) * 8]]
    row['bank_order'] = [original_row['bank_order'][bank] for bank in spec['banks']]
    row['rng_seeds'] = [original_row['rng_seeds'][bank] for bank in spec['banks']]
    # The qualified endpoint is executed without changing a byte of its source.
    # Its per-process constants and bound row restrict execution to this shard.
    ep.CASES = len(row['seeds'])
    ep.BANKS = len(spec['banks'])
    ep.bind = lambda path, task: (value, row) if str(path) == str(prepared) and task == spec['task'] else (_ for _ in ()).throw(ValueError('Shard binding mismatch'))
    staging = output / 'staging-links'
    for step in (0, 130):
        source = Path(spec['original_staging']) / str(step) / 'READY.json'
        ready = read(source)
        assert ready['completed'] and ready['task'] == spec['task']
        assert ready['prepared_sha256'] == manifest['prepared_sha256']
        assert ready['staging_script_sha256'] == manifest['endpoint_sha256']
        assert ready['checkpoint'] == next(cp for cp in row['checkpoints'] if cp['step'] == step)
        # Original staging processes retain their private weights until the original
        # result exists. Migration never writes that result or reclaims these files.
        ready['output'] = str(output)
        write(staging / str(step) / 'READY.json', ready)
    write(output / 'SHARD.json', dict(spec, migration_sha256=sha(manifest_path),
          seeds=row['seeds'], rng_seeds=row['rng_seeds'], bank_order=row['bank_order']))
    # Independent evaluator processes must not attach to each other's local Ray
    # cluster. Keep the upstream namespace and worker code; isolate only the
    # local orchestration service and cap its CPU/object-store reservation.
    import ray
    original_init = ray.init
    def local_ray(**kwargs):
        if kwargs.get('address') == 'auto' and not ray.is_initialized():
            kwargs.update(address='local', _temp_dir=f'/tmp/endpoint512-ray-{index}-{os.getpid()}',
                num_cpus=16, object_store_memory=2 << 30, include_dashboard=False)
        return original_init(**kwargs)
    ray.init = local_ray
    ep.run(prepared, spec['task'], staging, output, deadline)


def launch(manifest_path, output, deadline):
    import torch
    manifest = read(manifest_path)
    assert torch.cuda.device_count() == len(manifest['shards']) == 8
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    devices = visible.split(',') if visible else list(map(str, range(8)))
    assert len(devices) == 8 and len(set(devices)) == 8
    processes = []
    for index, device in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        log = (output / f'worker-{index:02d}.log').open('x')
        process = subprocess.Popen([sys.executable, '-B', __file__, 'worker',
            '--manifest', str(manifest_path), '--index', str(index), '--deadline', str(deadline)],
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((process, log))
    write(output / 'WORKERS.json', dict(started_unix=time.time(), pids=[p.pid for p, _ in processes]))
    codes = [process.wait() for process, _ in processes]
    for _, log in processes:
        log.close()
    write(output / 'WORKER-EXITS.json', dict(completed=all(code == 0 for code in codes), codes=codes))
    assert all(code == 0 for code in codes), codes
    combine(manifest_path, output / 'RESULT.json')


def combine(manifest_path, destination):
    manifest = read(manifest_path)
    ep = endpoint(manifest['metadata'])
    results = {}
    for task in TASKS:
        panels = list(manifest['tasks'][task]['retained_panels'])
        for spec in manifest['shards']:
            if spec['task'] != task:
                continue
            result = read(Path(spec['output']) / 'RESULT.json')
            assert result['completed'] and result['unresolved_intents'] == 0 and result['SGD'] == 0
            for path in sorted((Path(spec['output']) / 'panels').glob('*.json')):
                assert result['raw_receipt_sha256'][path.name] == sha(path)
                panels.append(read(path))
        _, row = ep.bind(Path(manifest['prepared']), task)
        outcomes, end_outcomes = {0: {}, 130: {}}, {0: {}, 130: {}}
        for panel in panels:
            if 'source_panel_path' in panel:
                assert sha(panel['source_panel_path']) == panel['source_panel_sha256']
            assert sha(panel['episode_path']) == panel['episode_sha256']
            assert panel['rng_seed'] == row['rng_seeds'][row['seeds'].index(panel['seeds'][0]) // 8]
            for episode in panel['episodes']:
                assert episode['seed'] not in outcomes[panel['step']], 'Duplicate case/policy'
                assert episode['native_counts']['controls'] - episode['native_counts']['reset_controls'] == 600
                outcomes[panel['step']][episode['seed']] = bool(episode['success_once'])
                end_outcomes[panel['step']][episode['seed']] = bool(episode['success_at_end'])
        assert all(set(values) == set(row['seeds']) and len(values) == 512 for values in outcomes.values())
        results[task] = dict(first_success=ep.paired(outcomes[0], outcomes[130], row['bootstrap_seed']),
            success_at_end=ep.paired(end_outcomes[0], end_outcomes[130], row['bootstrap_seed']),
            policy_episodes=1024, panels=128)
    result = dict(completed=True, tasks=results,
        holm=ep.holm({task: r['first_success'] for task, r in results.items()}),
        migration_sha256=sha(manifest_path), valid_policy_controls=1228800,
        valid_policy_episodes=2048, SGD=0, finished_unix=time.time())
    write(destination, result)
    return result


def barrier(manifest_path, deadline):
    """Keep the PyTorch master alive until every independent shard finishes."""
    manifest = read(manifest_path)
    paths = [Path(s['output']) / 'RESULT.json' for s in manifest['shards']]
    while time.time() < deadline:
        if all(p.exists() for p in paths):
            results = [read(p) for p in paths]
            assert all(r['completed'] and r['SGD'] == r['unresolved_intents'] == 0 for r in results)
            print(json.dumps(dict(all_shards_complete=True, count=len(paths))), flush=True)
            return
        time.sleep(1)
    raise TimeoutError('Not all independent shard results exist before the allocation deadline')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['worker', 'launch', 'combine', 'barrier'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.manifest, args.index, args.deadline)
    elif args.mode == 'launch':
        launch(args.manifest, args.output, args.deadline)
    elif args.mode == 'barrier':
        barrier(args.manifest, args.deadline)
    else:
        combine(args.manifest, args.output)


if __name__ == '__main__':
    main()
