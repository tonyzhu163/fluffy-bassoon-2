"""Run a precommitted paired panel in whole, disjoint waves on four GPUs."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from job import sha
from readout import TASKS, bank_wave, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
COLD = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--decision-sha256', required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert args.attempt.isalnum() and sha(args.decision) == args.decision_sha256
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    decision = json.loads(args.decision.read_text())
    count, minutes = decision['cases_per_task'], decision['scheduler_maximum_minutes']
    assert type(count) is int and 8 <= count <= 500
    assert type(minutes) is int and 30 <= minutes <= 720
    assert decision['shards'] == 4 and decision['outcome_blind_count_fixed']
    assert decision['reserved_GPU_hours'] == 4 * minutes / 60
    assert sha(HERE / 'readout.py') == decision['qualified_readout_sha256']
    qualification = decision['qualification']
    assert sha(Path(qualification['path'])) == qualification['sha256']
    qualified = json.loads(Path(qualification['path']).read_text())
    assert qualified['completed'] and qualified['binding']['stage'] == 'qualification'
    assert {row['task'] for row in qualified['results']} == set(TASKS)
    assert all(row['archive']['every_member_verified'] and row['archive']['cold_full_readback']
               for row in qualified['results'])
    restored_path = COLD / 'inputs01/RESTORED.json'
    assert sha(restored_path) == decision['restored_manifest_sha256']
    restored = json.loads(restored_path.read_text())
    assert restored == qualified['binding']['source_inputs'] and restored['completed']
    local = Path('/tmp') / HERE.name / args.attempt
    cold = COLD / args.attempt
    local.mkdir(parents=True, exist_ok=False)
    cold.mkdir(parents=True, exist_ok=False)
    input_bytes = sum(Path(record['path']).stat().st_size for record in restored['inputs'].values())
    # Shared inference inputs plus four bounded waves and their archive copies.
    assert local.stat().st_dev != PROJECT.stat().st_dev
    assert shutil.disk_usage(local).free > input_bytes + 4 * 8 * 2**30
    check = cold / '.write-probe'
    check.write_text('paired confirmation')
    assert check.read_text() == 'paired confirmation'
    check.unlink()
    env = dict(os.environ, RLINF_PATH=str(PROJECT / 'sources/critic-selected-a1229239-r1'),
        PYTHONPATH=str(CODE / 'src') + ':' + str(PROJECT / 'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(PROJECT / 'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(PROJECT / 'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', NCCL_P2P_DISABLE='1', FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        assigned = env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible) == len(assigned) == 4 and set(visible) == set(assigned)
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    started = time.time()
    deadline = started + (minutes - 15) * 60
    write(cold / 'LAUNCH.json', dict(source_commit=args.pin, hostname=socket.gethostname(),
        decision=decision, decision_sha256=args.decision_sha256, started_unix=started,
        maximum_GPU_hours=4 * minutes / 60, automatic_retries=False,
        local_free_bytes=shutil.disk_usage(local).free))
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'),
        '--expected-count', '4', '--purpose', 'evaluation', '--baige-container',
        '--output', str(cold / 'gpu-probe.json')], env=env, check=True, timeout=180)
    probe = json.loads((cold / 'gpu-probe.json').read_text())
    assert probe['status'] == 'passed'
    inputs = local / 'inputs'
    inputs.mkdir()
    for name, record in restored['inputs'].items():
        dest = inputs / name
        assert dest.is_relative_to(inputs) and '..' not in Path(name).parts
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(record['path'], dest)
        assert sha(dest) == record['sha256']
    write(inputs / 'RESTORED.json', restored)

    def worker(pair):
        shard, device = pair
        directory = local / f'shard-{shard:02d}'
        output = cold / f'shard-{shard:02d}'
        output.mkdir()
        rc = 1
        try:
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=str(cold / 'gpu-probe.json'), LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            with (output / 'console.log').open('x') as log:
                rc = subprocess.run([sys.executable, '-B', str(HERE / 'readout.py'),
                    '--pin', args.pin, '--output', str(directory), '--cold', str(output / 'evaluation'),
                    '--inputs', str(inputs), '--probe', str(cold / 'gpu-probe.json'),
                    '--stage', 'confirmation', '--count', str(count), '--shards', '4', '--shard', str(shard),
                    '--deadline', str(deadline)], env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline + 300 - time.time())).returncode
        except BaseException as error:
            write(output / 'ERROR.json', dict(error=repr(error), checked_unix=time.time()))
        complete = (output / 'evaluation/COMPLETE.json').exists()
        if rc != 0 or not complete:
            try:
                if directory.exists():
                    bank_wave(directory, output / 'partial-evidence')
            except BaseException as error:
                write(output / 'PARTIAL-ARCHIVE-ERROR.json', dict(error=repr(error), local_evidence=str(directory)))
        result = dict(shard=shard, exit_code=rc, complete=complete, ended_unix=time.time())
        write(output / 'EXIT.json', result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, enumerate(probe['devices'])))
    complete = all(row['exit_code'] == 0 and row['complete'] for row in results)
    write(cold / 'EXIT.json', dict(completed=complete, results=results, ended_unix=time.time(),
        allocation_GPU_hours=(time.time() - started) * 4 / 3600,
        incomplete_policy='Preserve all evidence and fixed planned denominator; never stop on significance'))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
