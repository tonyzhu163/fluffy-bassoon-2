"""Eight independent one-GPU bridge pipelines inside one exclusive allocation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-three-task-kv-velocity-meta-20261105-r1')


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def single(args):
    scripts = Path(__file__).resolve().parent
    fit = ROOT / 'fits' / args.interface / args.objective
    state = ROOT / 'workflows' / args.interface / args.objective
    state.mkdir(parents=True, exist_ok=True)
    lock = (state / '.workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    base = [sys.executable, '-B']
    common = ['--code', str(args.code), '--commit', args.commit, '--probe', str(args.probe),
              '--interface', args.interface, '--objective', args.objective]
    cycle = 0
    while time.time() + 600 < args.deadline and not stopped:
        status = read(fit / 'training-status.json')
        if status and status.get('converged'):
            break
        seconds = min(6300, int(args.deadline-time.time()-600))
        log_path = state / f'train-{args.attempt}-{cycle:02d}.log'
        with log_path.open('x') as log:
            command = base + [str(scripts / 'TRAIN.py')] + common + [
                '--output', str(fit), '--execution-batch', str(args.execution_batch), '--max-seconds', str(seconds)]
            if args.resume_contract_commit:
                command += ['--resume-contract-commit', args.resume_contract_commit]
            if args.member_cache:
                command += ['--member-cache', str(args.member_cache)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if stopped:
                    process.terminate()
                time.sleep(2)
            if process.returncode:
                write(state / 'RESULT.json', dict(completed=False, status='failed',
                    attempt=args.attempt, source_commit=args.commit, log_path=str(log_path),
                    training_status=read(fit / 'training-status.json'), checked_unix=time.time()))
                raise RuntimeError(f'Fit failed; inspect {log_path}')
        cycle += 1
        write(state / 'PROGRESS.json', dict(stage='fitting', status=read(fit / 'training-status.json'),
            checked_unix=time.time(), source_commit=args.commit))
    status = read(fit / 'training-status.json')
    if stopped or not status or not status.get('converged'):
        write(state / 'RESULT.json', dict(completed=False, status='operational_checkpoint',
            training_status=status, source_commit=args.commit))
        return
    if args.fit_only:
        write(state / 'RESULT.json', dict(completed=False, status='fit_complete_evaluation_held',
            training_status=status, source_commit=args.commit,
            reason='Separate shared-storage admission required before ID/OOD rollout writing'))
        return
    for split in ('ID', 'OOD'):
        if stopped or time.time()+600 >= args.deadline:
            break
        output = ROOT / 'readouts' / args.interface / args.objective / split
        result = read(output / 'RESULT.json')
        if result and result.get('completed'):
            continue
        with (state / f'eval-{split}-{args.attempt}.log').open('x') as log:
            command = base + [str(scripts / 'READOUT.py')] + common + [
                '--fit', str(fit), '--output', str(output), '--split', split,
                '--attempt', args.attempt, '--deadline', str(args.deadline)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                if stopped:
                    process.terminate()
                time.sleep(2)
            if process.returncode:
                raise RuntimeError(f'Evaluation failed; inspect {state}')
        result = read(output / 'RESULT.json')
        if not result or not result.get('completed'):
            break
        write(state / 'PROGRESS.json', dict(stage='evaluated_'+split, checked_unix=time.time()))
    completed = all((read(ROOT / 'readouts' / args.interface / args.objective / s / 'RESULT.json') or {}).get('completed') for s in ('ID', 'OOD'))
    write(state / 'RESULT.json', dict(completed=completed, training_status=status,
        source_commit=args.commit, optimizer_updates_at_deployment=0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('code', 'probe'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--resume-contract-commit')
    p.add_argument('--member-cache', type=Path)
    p.add_argument('--fit-only', action='store_true')
    p.add_argument('--attempt', required=True)
    p.add_argument('--qualification', type=Path, required=True)
    p.add_argument('--execution-batch', type=int, default=8)
    p.add_argument('--single', action='store_true')
    p.add_argument('--interface')
    p.add_argument('--objective')
    p.add_argument('--deadline', type=float)
    args = p.parse_args()
    if args.single:
        single(args)
        return
    qualification = read(args.qualification)
    if not qualification or not qualification.get('passed'):
        raise ValueError('Require successful native qualification of both steering interfaces')
    proof = read(args.probe)
    if not proof or proof['status'] != 'passed' or len(proof['devices']) != 8:
        raise ValueError('Require the actual passed eight-GPU allocation probe')
    directory = ROOT / 'allocations' / args.attempt
    directory.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 30600
    children = {}
    def stop(signum, frame):
        raise TimeoutError('Owned allocation ending; preserve complete per-model checkpoints')
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try:
        matrix = [(i, m) for i in ('kv_bridge', 'velocity_bridge') for m in ('ad', 'dpt', 'dit', 'ic_iql')]
        for (interface, objective), device in zip(matrix, proof['devices'], strict=True):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['visibility_token'],
                       LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            command = [sys.executable, '-B', __file__, '--single', '--code', str(args.code),
                '--commit', args.commit, '--attempt', args.attempt, '--probe', str(args.probe),
                '--qualification', str(args.qualification), '--interface', interface,
                '--objective', objective, '--execution-batch', str(args.execution_batch), '--deadline', str(deadline)]
            log = (directory / f'{interface}-{objective}.log').open('x')
            children[interface+'/'+objective] = (subprocess.Popen(command, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True), log)
        while any(p.poll() is None for p, _ in children.values()):
            write(directory / 'PROGRESS.json', dict(checked_unix=time.time(),
                processes={k: p.poll() for k, (p, _) in children.items()},
                methods={i+'/'+m: read(ROOT/'fits'/i/m/'training-status.json') for i,m in matrix}))
            time.sleep(20)
        write(directory / 'RESULT.json', dict(source_commit=args.commit,
            exit_codes={k:p.returncode for k,(p,_) in children.items()},
            workflows={i+'/'+m: read(ROOT/'workflows'/i/m/'RESULT.json') for i,m in matrix}))
        if any(p.returncode for p, _ in children.values()):
            raise RuntimeError('One or more independent pipelines failed; no automatic retry')
    finally:
        for process, _ in children.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, log in children.values():
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            log.close()
        write(directory / 'RELEASED.json', dict(ended_unix=time.time(),
            exit_codes={k:p.returncode for k,(p,_) in children.items()}, scheduler_release_pending=True))


if __name__ == '__main__':
    main()
