"""One-run allocated Linux CPU checks; no model, simulator or persistent tests."""
import importlib.util
import json
from pathlib import Path
import sys
import time

root, metadata, destination = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location('acceleration', root / 'accelerate.py')
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)
destination.mkdir(parents=True, exist_ok=False)
ep = acc.endpoint(metadata)
prepared = metadata / 'outputs' / acc.PARENT / 'PREPARED.json'
for count in range(4, 65):
    remaining = list(range(64 - count, 64))
    groups = acc.split_banks(remaining)
    assert all(groups)
    assert [b for group in groups for b in group] == remaining
    assert len({b for group in groups for b in group}) == count
    for group in groups:
        assert sum(len(ep.order(b)) for b in group) == 2 * len(group)

captured = []
def fake_run(path, task, staging, output, deadline):
    value, row = ep.bind(path, task)
    captured.append(dict(task=task, cases=ep.CASES, banks=ep.BANKS,
        seeds=row['seeds'], orders=row['bank_order'], noise=row['rng_seeds']))

original_bind = ep.bind
ep.run = fake_run
acc.endpoint = lambda metadata: ep
original_root = Path('/mowenjie/fluffy-bassoon-2/outputs') / acc.PARENT
shards = []
for task in acc.TASKS:
    _, row = original_bind(prepared, task)
    for index, banks in enumerate(acc.split_banks(list(range(24, 64)))):
        shards.append(dict(task=task, shard=index, banks=banks,
            output=str(destination / task / f'shard-{index:02d}'),
            original_staging=str(original_root / task / 'staging-r1')))
manifest = dict(metadata=str(metadata), endpoint_sha256=acc.sha(ep.HERE),
    prepared=str(prepared), prepared_sha256=acc.sha(prepared), shards=shards)
manifest_path = destination / 'MIGRATION.json'
acc.write(manifest_path, manifest)
for index, shard in enumerate(shards):
    ep.bind = original_bind
    ep.CASES, ep.BANKS = 512, 64
    import ray
    original_init = ray.init
    acc.worker(manifest_path, index, time.time() + 3600)
    ray.init = original_init
    _, row = original_bind(prepared, shard['task'])
    actual = captured[-1]
    assert actual['cases'] == 8 * len(shard['banks'])
    assert actual['banks'] == len(shard['banks'])
    assert actual['seeds'] == [s for b in shard['banks'] for s in row['seeds'][b * 8:(b + 1) * 8]]
    assert actual['noise'] == [row['rng_seeds'][b] for b in shard['banks']]
    assert actual['orders'] == [row['bank_order'][b] for b in shard['banks']]

import os
probe = destination / 'write-probe'
with probe.open('x') as stream:
    stream.write('parallel endpoint qualification\n')
    stream.flush()
    os.fsync(stream.fileno())
probe.unlink()
free = os.statvfs(destination)
assert free.f_bavail * free.f_frsize > 200 * 2**30
acc.write(destination / 'QUALIFICATION.json', dict(completed=True,
    partition_frontiers_checked=61, worker_binding_checks=len(captured),
    fixed_batch=8, case_and_noise_identity_exact=True, ABBA_order_exact=True,
    endpoint_sha256=acc.sha(ep.HERE), wrapper_sha256=acc.sha(root / 'accelerate.py'),
    prepared_sha256=acc.sha(prepared), new_controls=0, SGD=0,
    free_GiB=free.f_bavail * free.f_frsize / 2**30, checked_unix=time.time()))
print((destination / 'QUALIFICATION.json').read_text(), flush=True)
