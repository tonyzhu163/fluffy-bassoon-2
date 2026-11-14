"""Bounded four-GPU paired evaluation of the eight fixed-final matched fits."""
import argparse
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from labels import sha, write
from matched_readout import MODELS

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--max-minutes', type=int, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert args.attempt.isalnum() and 60 <= args.max_minutes <= 720
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == args.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    assert sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    assert spec['shards'] == 4 and spec['cases_per_task'] == 40 and set(spec['models']) == set(MODELS)
    assert spec['scheduler_maximum_minutes'] == args.max_minutes
    assert spec['reserved_GPU_hours'] == 4 * args.max_minutes / 60
    for name in MODELS:
        ref = spec['models'][name]
        assert sha(ref['complete']['path']) == ref['complete']['sha256']
        complete = json.loads(Path(ref['complete']['path']).read_text())
        assert complete['completed'] and complete['progress']['actor_update'] == 10000
        assert complete['candidate'] == ref['candidate'] and sha(ref['candidate']['path']) == ref['candidate']['sha256']
    local = Path('/tmp') / HERE.name / args.attempt
    cold = Path('/mnt/bos/adgfyl/strong-plant-archives') / HERE.name / args.attempt
    local.mkdir(parents=True, exist_ok=False)
    cold.mkdir(parents=True, exist_ok=False)
    assert local.stat().st_dev != ROOT.stat().st_dev
    # Four complete eight-policy waves plus tar copies, bounded at20GiB each.
    assert shutil.disk_usage(local).free > 4 * 20 * 2**30
    probe_path = cold / '.write-probe'
    probe_path.write_text('eight matched policies')
    assert probe_path.read_text() == 'eight matched policies'
    probe_path.unlink()
    env = dict(os.environ, RLINF_PATH=str(ROOT / 'sources/critic-selected-a1229239-r1'),
        PYTHONPATH=str(CODE / 'src') + ':' + str(ROOT / 'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(ROOT / 'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(ROOT / 'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', NCCL_P2P_DISABLE='1', FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
        assigned = env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible) == len(assigned) == 4 and set(visible) == set(assigned)
        env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    started = time.time()
    deadline = started + (args.max_minutes - 10) * 60
    write(cold / 'LAUNCH.json', dict(source_commit=args.pin, spec=spec, spec_sha256=args.spec_sha256,
        hostname=socket.gethostname(), started_unix=started, maximum_GPU_hours=4 * args.max_minutes / 60,
        scheduler_maximum_minutes=args.max_minutes, automatic_retries=False,
        local_free_bytes=shutil.disk_usage(local).free))
    subprocess.run([sys.executable, '-B', str(CODE / 'scripts/[train]gpu_probe.py'),
        '--expected-count', '4', '--purpose', 'evaluation', '--baige-container',
        '--output', str(cold / 'gpu-probe.json')], env=env, check=True, timeout=180)
    probe = json.loads((cold / 'gpu-probe.json').read_text())
    assert probe['status'] == 'passed'
    helper_path = HERE.parent / 'robocasa-dit-paired-confirmation-20261114-r1/readout.py'
    helper_spec = importlib.util.spec_from_file_location('paired_native_archive', helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)

    def worker(pair):
        shard, device = pair
        output, directory = cold / f'shard-{shard:02d}', local / f'shard-{shard:02d}'
        output.mkdir()
        rc = 1
        try:
            worker_env = dict(env, CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=str(cold / 'gpu-probe.json'), LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            with (output / 'console.log').open('x') as log:
                rc = subprocess.run([sys.executable, '-B', str(HERE / 'matched_readout.py'),
                    '--pin', args.pin, '--spec', str(args.spec), '--spec-sha256', args.spec_sha256,
                    '--probe', str(cold / 'gpu-probe.json'), '--output', str(directory),
                    '--cold', str(output / 'evaluation'), '--shards', '4', '--shard', str(shard),
                    '--deadline', str(deadline)], env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(1, deadline + 300 - time.time())).returncode
        except BaseException as error:
            write(output / 'ERROR.json', dict(error=repr(error), checked_unix=time.time()))
        complete = (output / 'evaluation/COMPLETE.json').exists()
        if rc != 0 or not complete:
            try:
                if directory.exists():
                    helper.bank_wave(directory, output / 'partial-evidence')
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
        incomplete_policy='Retain all planned cases and both corpus conditions; disclose failures instead of selecting successful cases'))
    raise SystemExit(0 if complete else 1)


if __name__ == '__main__':
    main()
