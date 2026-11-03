"""Finish only missing logical ranks, sequentially on one allocated GPU."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import torch

m = runpy.run_path(str(Path(__file__).parents[1] / 'opendrawer-memory-controls-20261102-r1/CONTROL.py'))
p = argparse.ArgumentParser()
p.add_argument('--deadline', required=True, type=float)
args = p.parse_args()
assert torch.cuda.device_count() == 1 and int(os.environ['WORLD_SIZE']) == 1
assert int(os.environ['RANK']) == 0
binding = json.loads((m['OUT'] / 'binding.json').read_text())
assert m['BASE']['head'](m['CODE']) == binding['source'] and m['BASE']['clean'](m['CODE'])
assert binding['fresh_episodes'] == 48 and len(binding['recovered']) == 1052
assert binding['fresh_policy_controls'] == 36000
probe = m['OUT'] / 'job' / 'gpu-probe.json'
probe.parent.mkdir(parents=True, exist_ok=False)
subprocess.run([sys.executable, '-B', str(m['CODE'] / 'scripts/[train]gpu_probe.py'),
    '--expected-count', '1', '--purpose', 'evaluation', '--baige-container', '--output', str(probe)], check=True)
device = json.loads(probe.read_text())['devices'][0]
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device['visibility_token']),
    LYCO_BAIGE_GPU_MAP=str(probe), LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
    MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
counts = {str(rank): len(m['pending_keys'](binding, rank)) for rank in range(m['WORLD'])}
assert sum(counts.values()) == 48
m['write'](m['OUT'] / 'job' / 'manifest.json', dict(source=binding['source'],
    binding=m['ref'](m['OUT'] / 'binding.json'), logical_ranks=counts,
    allocated_gpus=1, device=device, deadline=args.deadline))
for rank in range(m['WORLD']):
    assert time.time() < args.deadline
    log = m['OUT'] / 'logs' / f'worker-{rank:02d}.log'
    with log.open('x') as stream:
        subprocess.run([sys.executable, '-B', str(m['CODE'] /
            'outputs/opendrawer-memory-controls-20261102-r1/CONTROL.py'),
            '--worker', str(rank), '--deadline', str(args.deadline)],
            env=env, stdout=stream, stderr=subprocess.STDOUT, check=True,
            timeout=max(1, args.deadline - time.time()))
m['write'](m['OUT'] / 'COLLECTED.json', dict(complete=True, new_episodes=48,
    recovered_episodes=1052, logical_ranks=counts, finished_unix=time.time(),
    next='Release GPU; run committed SUMMARIZE.py on allocated development CPUs'))
