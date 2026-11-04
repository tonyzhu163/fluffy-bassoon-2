"""One-run allocated Linux qualification; no simulator, inference or SGD."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

code, output = map(Path, sys.argv[1:])
run = code / 'outputs/robocasa-flow-sde-endpoint-test512-20261104-r1'
spec = importlib.util.spec_from_file_location('endpoint_qualification', run / 'endpoint.py')
endpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(endpoint)
prepared = run / 'PREPARED.json'
assert endpoint.read(prepared)['maximum_policy_controls'] == 1228800
regenerated = output / 'REGENERATED.json'
output.mkdir(parents=True, exist_ok=False)
endpoint.prepare(code / 'outputs' / endpoint.SWEEP_RUN / 'SWEEP-PREPARED-R2.json', regenerated)
assert endpoint.read(prepared) == endpoint.read(regenerated)
assert sum(endpoint.order(b) == (0, 130) for b in range(64)) == 32
assert sum(endpoint.order(b) == (130, 0) for b in range(64)) == 32
initial = {i: False for i in range(512)}
later = dict(initial)
for i in range(10):
    later[i] = True
r = endpoint.paired(initial, later, 20261104)
assert r['wins'] == 10 and r['losses'] == 0 and r['exact_McNemar_p'] == 2 / 1024
assert endpoint.paired(initial, initial, 20261104)['exact_McNemar_p'] == 1
decisions = endpoint.holm({'a': r, 'b': dict(r, exact_McNemar_p=.04)})
assert all(v['improvement'] for v in decisions.values())
assert not any(v['improvement'] for v in endpoint.holm({
    'a': dict(r, exact_McNemar_p=.03), 'b': dict(r, exact_McNemar_p=.04)}).values())
sys.path.insert(0, str(code / 'outputs' / endpoint.SWEEP_RUN))
from native_eval import configuration
configs = []
for task in endpoint.TASKS:
    _, row = endpoint.bind(prepared, task)
    cfg = configuration(dict(task=task, config=row['source_config'],
        config_sha256=row['source_config_sha256']), output / task, time.time()+3600,
        maximum_actions=619200)
    assert cfg.runner.only_eval and cfg.env.eval.total_num_envs == 8
    assert cfg.rollout.model.precision == 'bf16'
    assert len(row['seeds']) == len(set(row['seeds'])) == 512
    configs.append(dict(task=task, seeds=[row['seeds'][0], row['seeds'][-1]],
        fixed_batch=8, strict_model_hashes=[p['sha256'] for p in row['checkpoints']]))
probe = output / 'write-probe'
with probe.open('x') as f:
    f.write('endpoint512 qualification\n')
    f.flush()
    os.fsync(f.fileno())
probe.unlink()
free = os.statvfs(output)
free_bytes = free.f_bavail * free.f_frsize
assert free_bytes > 200 * 2**30
receipt = dict(completed=True, source_commit=subprocess.check_output(
    ['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip(),
    tasks=configs, manifest_regenerated_exact=True, ABBA_balance=[32,32],
    numerical_exact_McNemar=True, Holm_reference_pass=True,
    native_eval_sha256=endpoint.sha(code/'outputs'/endpoint.SWEEP_RUN/'native_eval.py'),
    endpoint_sha256=endpoint.sha(run/'endpoint.py'), prepared_sha256=endpoint.sha(prepared),
    free_bytes=free_bytes, CPU_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    RAM_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),
    new_controls=0, SGD=0, GPU_seconds=0, checked_unix=time.time())
endpoint.write(output/'QUALIFICATION.json', receipt)
print(json.dumps(receipt), flush=True)
