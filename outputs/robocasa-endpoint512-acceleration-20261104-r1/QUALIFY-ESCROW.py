"""Allocated Linux CPU qualification of real per-shard action escrow."""
import importlib.util
import json
from pathlib import Path
import sys
import time

source, runtime, manifest_path, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('endpoint_parallel', source)
acc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acc)
spec = importlib.util.spec_from_file_location('frozen_journal', runtime / 'src/fluffy_bassoon/meta_rl/rlinf_native_ppo.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
manifest = acc.read(manifest_path)
roots, journals = [], []
for index, shard in enumerate(manifest['shards']):
    model_output = Path(shard['output'])
    assert model_output.name == 'gpu-r1'
    root = model_output.parent / 'accounting'
    assert root not in roots
    roots.append(root)
    # Same layout as native_eval.configuration, under a CPU-only qualification
    # output. Synthetic ledger calls perform no simulator/model computation.
    private = output / str(index) / 'accounting'
    limit = 8 * len(shard['banks']) * 2 * 600 + 4800
    journal = native.ActionJournal(private, rank=0, world=1,
        attempt=model_output.name, maximum_actions=limit)
    cursor = journal.begin(400 + index, phase='CPU synthetic isolation check')
    journal.finish(cursor, 400 + index)
    journals.append(journal)
for index, journal in enumerate(journals):
    assert journal.charged() == 400 + index
assert len(roots) == manifest['maximum_gpus']
acc.write(output / 'QUALIFICATION.json', dict(completed=True,
    source_sha256=acc.sha(source), journal_source_sha256=acc.sha(runtime / 'src/fluffy_bassoon/meta_rl/rlinf_native_ppo.py'),
    manifest_sha256=acc.sha(manifest_path), distinct_private_accounting_roots=len(roots),
    real_action_journal_isolation_pass=True, unequal_shard_limits_pass=True,
    model_forwards=0, simulator_controls=0, SGD=0, checked_unix=time.time()))
print((output / 'QUALIFICATION.json').read_text(), flush=True)
