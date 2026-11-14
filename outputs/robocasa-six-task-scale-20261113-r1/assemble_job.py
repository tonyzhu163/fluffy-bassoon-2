"""CPU-only assembly of completed label tasks, with one shared corpus manifest."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from corpus import TASKS
from labels import sha, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and args.attempt.isalnum()
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    local = Path('/tmp') / HERE.name / args.attempt
    cold = COLD / args.attempt
    local.mkdir(parents=True, exist_ok=False)
    cold.mkdir(parents=True, exist_ok=False)
    assert local.stat().st_dev != ROOT.stat().st_dev
    # Two target variants, complete observation/action arrays, and temporary copies.
    required = 4 * 2 * 128 * 360 * (2073 * 4 + 600 * 4 * 2 + 14)
    assert shutil.disk_usage(local).free > required
    probe = cold / '.write-probe'
    probe.write_text('CPU dataset assembly')
    assert probe.read_text() == 'CPU dataset assembly'
    probe.unlink()
    started = time.time()
    deadline = started + 4 * 3600
    env = dict(os.environ, PYTHONPATH=str(CODE / 'src'), PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    write(cold / 'LAUNCH.json', dict(source_commit=args.pin, hostname=socket.gethostname(),
        started_unix=started, deadline_unix=deadline, CPU_only=True,
        local_free_bytes=shutil.disk_usage(local).free, required_local_bytes=required))
    ready = {}
    try:
        while len(ready) < len(TASKS):
            assert time.time() < deadline, 'CPU assembly deadline; preserve completed task exports'
            for task in TASKS:
                if task in ready:
                    continue
                label_attempt = 'labels03' if task in ('CloseToasterOvenDoor', 'TurnOffStove') else 'labels04'
                label_root = COLD / label_attempt
                complete = label_root / task / 'COMPLETE.json'
                if not complete.exists():
                    failure = label_root / (task + '-EXIT.json')
                    assert not failure.exists(), f'Label worker closed without complete labels: {failure}'
                    continue
                digest = sha(complete)
                with (cold / (task + '.log')).open('x') as log:
                    subprocess.run([sys.executable, '-B', str(HERE / 'assemble.py'),
                        '--pin', args.pin, '--complete', str(complete), '--complete-sha256', digest,
                        '--task', task, '--output', str(local / task), '--cold', str(cold / task)],
                        env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                        timeout=max(1, deadline - time.time()))
                result = cold / task / 'ASSEMBLY.json'
                assembly = json.loads(result.read_text())
                assert assembly['student_fit_ready'] and assembly['all_arrays_roundtrip_exact']
                ready[task] = dict(path=str(result), sha256=sha(result))
                write(cold / 'PROGRESS.json', dict(checked_unix=time.time(), tasks=ready,
                    remaining=[name for name in TASKS if name not in ready]))
            if len(ready) < len(TASKS):
                time.sleep(30)
        spec = cold / 'CORPORA.json'
        write(spec, dict(source_commit=args.pin, tasks={task: ready[task] for task in TASKS},
            original_three=list(TASKS[:3]), representative_six=list(TASKS),
            shared_original_task_references=True, collection_updates=30))
        write(cold / 'COMPLETE.json', dict(completed=True, CPU_only=True,
            corpora=dict(path=str(spec), sha256=sha(spec)), tasks=list(TASKS),
            ended_unix=time.time(), next_action='Validate actual corpus loading and source queries before GPU fits'))
    except BaseException as error:
        write(cold / 'ERROR.json', dict(error=repr(error), completed_tasks=ready, checked_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
