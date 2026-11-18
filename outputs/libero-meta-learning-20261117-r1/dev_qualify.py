"""Run the two fixed qualification cases serially on an allocated Baige dev GPU."""
import argparse
import json
import os
import socket
import subprocess
import time

from job import CODE, HERE, POLICY_PYTHON, PRO, PYTHON, ROOT, SETUP, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostname', required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--checkpoint', choices=('openpi', 'rlinf'), default='rlinf')
    parser.add_argument('--reference-cadence', action='store_true')
    parser.add_argument('--stage', choices=('qualify', 'collect', 'reference'), default='qualify')
    args = parser.parse_args()
    if not (args.hostname.startswith('aibox-') and socket.gethostname() == args.hostname
            and args.attempt.isalnum()):
        parser.error('Require the selected Baige development machine and a fresh attempt name')
    pin = subprocess.check_output(['git', '-C', CODE, 'rev-parse', 'HEAD'], text=True).strip()
    if pin != args.pin or subprocess.check_output(['git', '-C', CODE, 'status', '--porcelain']):
        raise ValueError('Use the reviewed, clean checkout')
    env = environment()
    assigned = env['NVIDIA_VISIBLE_DEVICES']
    if assigned != env['NVIDIA_VISIBLE_GPUS_UUID'] or ',' in assigned or env['CGPU_COUNT'] != '1':
        raise RuntimeError('Require one allocated development GPU')
    env['CUDA_VISIBLE_DEVICES'] = assigned
    out = ROOT/'outputs'/HERE.name/args.attempt
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + 90 * 60
    success = False
    try:
        subprocess.run([PYTHON, '-Bu', CODE/'scripts/[train]gpu_probe.py', '--expected-count', '1',
                        '--purpose', 'training', '--direct-hostname', args.hostname,
                        '--output', out/'gpu.json'], env=env, check=True, timeout=180)
        if args.stage == 'reference':
            legacy = ROOT.parent/'fluffy-bassoon'
            reference_env = dict(env, JAX_PLATFORMS='cuda', OPENPI_DATA_HOME=str(legacy/'cache/openpi-215abfb'),
                                 PYTHONPATH=f'{PRO}:{SETUP}/simulator-source:{HERE}')
            with (out/'reference.log').open('x') as log:
                subprocess.run([legacy/'envs/openpi-215abfb/bin/python', '-Bu', HERE/'reference.py',
                                '--output', out/'reference'], env=reference_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=max(1, deadline-time.time()))
            success = True
            return
        workers = [(f'rank{rank}', [HERE/'qualify.py', '--rank', str(rank)]) for rank in range(2)]
        if args.stage == 'collect':
            workers = [('corpus', [HERE/'collect_pilot.py'])]
        for name, command in workers:
            if args.reference_cadence:
                command.append('--reference-cadence')
            with (out/f'{name}.log').open('x') as log:
                subprocess.run([POLICY_PYTHON, '-Bu', *command, '--checkpoint', args.checkpoint,
                                '--output', out/name, '--deadline', str(deadline)],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=max(1, deadline - time.time()))
        success = True
    finally:
        (out/'end.json').write_text(json.dumps(dict(pin=pin, hostname=args.hostname,
            pid=os.getpid(), started=started, ended=time.time(), complete=success)) + '\n')


if __name__ == '__main__':
    main()
