"""Read-only smoke on terminal saved PPO runs; no simulator or motor loads."""
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
import torch

from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset
from fluffy_bassoon.meta_rl.vla_source40_ppo_corpus import main


root = Path('/mowenjie/fluffy-bassoon-2')
output = root / 'outputs/ppo-dpt-export-check-20261030-r1'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
results = {}


def export(name, run_id, objectives):
    source = root / 'outputs' / run_id
    assert (source / 'exit-code.txt').read_text().strip() == '0'
    source = source / 'experiment'
    recovery = source / 'learner/latest.pt'
    teacher = source / 'learner/candidate.pt'
    destination = output / name
    main(['--source-run', str(source), '--recovery-sha256', file_sha256(recovery),
          '--teacher', str(teacher), '--teacher-sha256', file_sha256(teacher),
          '--output', str(destination), '--objectives', *objectives,
          '--source-commit', commit, '--expected-hostname', socket.gethostname(),
          '--threads', '1'])
    index = json.loads((destination / 'index.json').read_text())
    assert index['metadata']['additional_simulator_actions'] == 0
    assert index['metadata']['additional_vla_queries'] == 0
    assert all(file_sha256(path) == digest for path, digest in index['metadata']['input_sha256'].items())
    loaded = {key: load_dataset(value['path'])[0] for key, value in index['datasets'].items()}
    if 'ad' in loaded:
        for ad, dpt in zip(loaded['ad'], loaded['dpt'], strict=True):
            for field in ad:
                if field != 'target_noise':
                    np.testing.assert_array_equal(ad[field], dpt[field])
            np.testing.assert_array_equal(ad['target_noise'], ad['executed_noise'])
    results[name] = dict(datasets=index['datasets'], actual_task_keys=index['metadata']['task_keys'],
        pinned_input_count=len(index['metadata']['input_sha256']),
        source_counters=index['metadata']['collection_costs_count_once'],
        all_input_hashes_unchanged=True, additional_simulator_actions=0, additional_vla_queries=0)
    return loaded


single = export('single-paired', 'libero40-1m-ppo-qual-20261029-r1', ['ad', 'dpt'])
repeat = export('single-repeat', 'libero40-1m-ppo-qual-20261029-r1', ['ad', 'dpt'])
for objective in single:
    for first, second in zip(single[objective], repeat[objective], strict=True):
        for field in first:
            np.testing.assert_array_equal(first[field], second[field])
shared = export('shared-dpt', 'libero40-1m-shared-ppo-qual-20261029-r1', ['dpt'])
try:
    export('shared-ad-rejected', 'libero40-1m-shared-ppo-qual-20261029-r1', ['ad', 'dpt'])
except ValueError as exc:
    assert 'learning trajectory' in str(exc)
    assert not (output / 'shared-ad-rejected').exists()
else:
    raise AssertionError('One fixed-policy trial per task must not become AD')
receipt = dict(source_commit=commit, hostname=socket.gethostname(), torch=torch.__version__,
    results=results, deterministic_repeat_identical=True, fixed_policy_ad_rejected=True,
    no_robot_performance_claim=True)
(output / 'real-data-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
