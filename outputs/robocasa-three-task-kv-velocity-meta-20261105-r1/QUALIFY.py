"""Two allocated GPUs, one disposable native qualification per steering interface."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    proof = json.loads(args.probe.read_text())
    if proof['status'] != 'passed' or len(proof['devices']) != 2:
        raise ValueError('Require the passed actual two-GPU allocation probe')
    args.output.mkdir(exist_ok=False, parents=True)
    processes = []
    started = time.time()
    try:
        for interface, device in zip(('kv_bridge', 'velocity_bridge'), proof['devices'], strict=True):
            command = [sys.executable, '-B', str(Path(__file__).with_name('TRAIN.py')),
                '--code', str(args.code), '--commit', args.commit, '--probe', str(args.probe),
                '--output', str(args.output / interface), '--interface', interface,
                '--objective', 'dpt', '--qualify', '--qualify-updates', '50', '--execution-batch', '8']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            log = (args.output / (interface + '.log')).open('x')
            processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        while any(p.poll() is None for p, _ in processes):
            if any(p.poll() not in (None, 0) for p, _ in processes):
                raise RuntimeError('Native qualification failed; preserve logs and diagnose before a new attempt')
            time.sleep(2)
        if any(p.returncode for p, _ in processes):
            raise RuntimeError('Native qualification failed')
        result = {name: json.loads((args.output / name / 'QUALIFICATION.json').read_text())
                  for name in ('kv_bridge', 'velocity_bridge')}
        (args.output / 'RESULT.json').write_text(json.dumps(dict(passed=True, interfaces=result), indent=2)+'\n')
    finally:
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for process, log in processes:
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            log.close()
        (args.output / 'CLOCK.json').write_text(json.dumps(dict(started_unix=started,
            ended_unix=time.time(), gpus=2, exit_codes=[p.returncode for p, _ in processes]), indent=2)+'\n')


if __name__ == '__main__':
    main()
