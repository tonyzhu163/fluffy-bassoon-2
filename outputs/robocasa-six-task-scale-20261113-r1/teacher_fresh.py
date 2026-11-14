"""Fresh corrected-PPO source, with the same recipe and endpoint for every task."""
import argparse
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
sys.path.insert(0, str(HERE.parent / 'robocasa-six-source-decision-20261112-r1'))
from teacher_local import archive, snapshot, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--task', required=True, choices=(
        'OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug', 'TurnOffStove'))
    parser.add_argument('--attempt', default='attempt01')
    args = parser.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert re.fullmatch(r'attempt\d+', args.attempt)
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    for name in ('ROBOCASA_SIM_PPO_RESUME', 'ROBOCASA_SIM_PPO_BEFORE_REFERENCE', 'ROBOCASA_SIM_PPO_BEFORE_SHA256'):
        assert not os.environ.get(name), 'Fresh source must start from original SFT, never a historical PPO checkpoint'
    source = Path('/tmp') / HERE.name / 'teachers' / args.task / args.attempt
    remote = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name / 'teachers' / args.task / args.attempt
    assert not source.exists() and not remote.exists()
    source.parent.mkdir(parents=True, exist_ok=True)
    assert source.parent.stat().st_dev != PROJECT.stat().st_dev
    remote.mkdir(parents=True, exist_ok=False)
    probe = remote / '.write-probe'
    probe.write_text('corrected-PPO-source'); assert probe.read_text() == 'corrected-PPO-source'; probe.unlink()
    # The development machine's scratch capacity says nothing about the assigned
    # job node. Retain its actual measurements even if admission fails.
    usage = shutil.disk_usage(source.parent)
    required = 400 * 2**30
    storage = dict(hostname=socket.gethostname(), source_commit=args.pin,
        task=args.task, attempt=args.attempt, path=str(source.parent),
        total_bytes=usage.total, free_bytes=usage.free, required_bytes=required,
        project_free_bytes=shutil.disk_usage(PROJECT).free,
        separate_from_project=source.parent.stat().st_dev != PROJECT.stat().st_dev,
        passed=usage.free >= required, checked_unix=time.time(),
        rationale='Completed30-update sources used136GiB; retain200GiB floor plus checkpoint/transient headroom')
    write(remote / 'STORAGE.json', storage)
    print(storage, flush=True)
    if not storage['passed']:
        raise RuntimeError(f'Assigned node has {usage.free / 2**30:.1f}GiB scratch free; '
                           '400GiB required. No model started; see STORAGE.json.')
    probe = source.parent / (args.attempt + '-write-probe')
    with probe.open('x') as stream:
        stream.write('corrected-PPO-local-scratch')
    assert probe.read_text() == 'corrected-PPO-local-scratch'
    probe.unlink()
    started = time.time()
    write(remote / 'LAUNCH.json', dict(source_commit=args.pin, task=args.task, attempt=args.attempt,
        hostname=socket.gethostname(), started_unix=started, local_output=str(source),
        initial_step=0, target_step=30, maximum_scheduler_minutes=150, maximum_GPU_hours=20,
        runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1', automatic_retry=False,
        initialization='Original Human300 SFT; fresh critic, optimizer, sampler and RNG'))
    done = threading.Event()
    monitor = threading.Thread(target=snapshot, args=(source, remote, done), daemon=False)
    monitor.start()
    try:
        result = subprocess.run(['bash', str(HERE / 'teacher.sh'), args.pin, args.task, args.attempt],
                                env=dict(os.environ, TASK_LOCAL_OUTPUT=str(source)))
    finally:
        done.set(); monitor.join()
    assert (source / 'EXIT.json').exists()
    receipt = archive(source, remote)
    import json
    for name in ('progress.json', 'MANIFEST.json', 'before-evaluation.json', 'after-evaluation.json',
                 'operator-checkpoint-exit.json', 'EXIT.json'):
        path = source / name
        if path.exists():
            write(remote / name, json.loads(path.read_text()))
    write(remote / 'COMPLETE.json', dict(archive_verified=receipt['verified'], learner_exit_code=result.returncode,
        started_unix=started, ended_unix=time.time(), complete_checkpoint_boundaries=receipt['complete_checkpoint_boundaries']))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
