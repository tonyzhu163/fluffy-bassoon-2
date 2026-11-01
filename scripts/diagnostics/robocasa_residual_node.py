#!/usr/bin/env python
"""Run this allocated node's two workers in the existing complete-batch path."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    code = Path(__file__).resolve().parents[2]
    queue_root = Path(profile['dispatch_root'])
    active, seen, streams = {}, set(), {}
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        while time.time() < profile['deadline_unix']:
            if (queue_root / 'shutdown.json').exists():
                return
            for local_rank, device in enumerate(profile['devices']):
                rank = profile['node_rank'] * 2 + local_rank
                if rank in active:
                    child, task = active[rank]
                    if child.poll() is None:
                        continue
                    if child.returncode:
                        write_new(Path(task['output']) / 'error.json', dict(rank=rank, return_code=child.returncode))
                        raise RuntimeError(f'Native rollout rank {rank} failed')
                    streams.pop(rank).close()
                    del active[rank]
                tasks = sorted((queue_root / f'rank-{rank:02d}').glob('*.json'))
                fresh = [path for path in tasks if str(path) not in seen]
                if not fresh:
                    continue
                if len(fresh) != 1:
                    raise ValueError('Only one complete pending bank may be outstanding')
                path = fresh[0]
                task = json.loads(path.read_text())
                if task['source_commit'] != profile['source_commit'] or sha(task['request']) != task['sha256']:
                    raise ValueError('Exact immutable producing policy request required')
                seen.add(str(path))
                env = dict(os.environ, **profile['native_env'])
                env.update(CUDA_VISIBLE_DEVICES=device['visibility_token'],
                    LYCO_BAIGE_ISOLATED_UUID=device['uuid'], LYCO_BAIGE_GPU_MAP=profile['probe_path'],
                    MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']),
                    OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                stream = (Path(task['output']) / 'console.log').open('x')
                streams[rank] = stream
                active[rank] = (subprocess.Popen([sys.executable, '-B',
                    str(code / 'scripts/diagnostics/robocasa_residual_rollout.py'), '--request', task['request']],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True), task)
            time.sleep(.2)
        raise TimeoutError('Finite distributed-node deadline')
    finally:
        for child, _ in active.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child, _ in active.values():
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    main()
