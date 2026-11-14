"""Bounded single-GPU native qualification; all durable evidence in private BOS."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
PROJECT = Path('/mowenjie/fluffy-bassoon-2')
PYTHON = PROJECT/'setups/robocasa365-native-20261101-r1/env/bin/python'


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pin', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--max-minutes', type=int, default=120)
    args=p.parse_args()
    started=time.time()
    assert sys.platform=='linux' and socket.gethostname().startswith('job-')
    assert args.attempt.isalnum()
    assert 30 <= args.max_minutes <= 120
    launch_limit_seconds = (args.max_minutes - 5) * 60
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==args.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    local=Path('/tmp')/HERE.name/args.attempt
    cold=Path('/mnt/bos/adgfyl/strong-plant-archives')/HERE.name
    output=cold/args.attempt
    local.mkdir(parents=True,exist_ok=False)
    output.mkdir(parents=True,exist_ok=False)
    usage=shutil.disk_usage(local)
    # 20GiB inference inputs plus one bounded wave/archive and system reserve.
    assert local.stat().st_dev != PROJECT.stat().st_dev and usage.free>80*2**30
    probe=output/'.write-probe';probe.write_text('paired-native-qualification');assert probe.read_text()=='paired-native-qualification';probe.unlink()
    env=dict(os.environ,RLINF_PATH=str(PROJECT/'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(PROJECT/'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(PROJECT/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        PYTHONPATH=str(CODE/'src')+':'+str(PROJECT/'sources/critic-selected-a1229239-r1'),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',NCCL_P2P_DISABLE='1',FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        assigned=env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible)==len(assigned)==1 and set(visible)==set(assigned)
        env['CUDA_VISIBLE_DEVICES']='0'
    write(output/'LAUNCH.json',dict(source_commit=args.pin,hostname=socket.gethostname(),started_unix=started,
        maximum_GPU_hours=args.max_minutes/60,automatic_retry=False,stage='qualification',cases_per_task=8,
        local_free_bytes=usage.free,local_output=str(local),launch_limit_seconds=launch_limit_seconds))
    rc=1
    try:
        subprocess.run([str(PYTHON),'-B',str(CODE/'scripts/[train]gpu_probe.py'),
            '--expected-count','1','--purpose','evaluation','--baige-container',
            '--output',str(output/'gpu-probe.json')],env=env,check=True,timeout=120)
        inputs=local/'inputs';inputs.mkdir()
        restored=json.loads((cold/'inputs01/RESTORED.json').read_text())
        for name,record in restored['inputs'].items():
            destination=inputs/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(record['path'],destination)
            assert sha(destination)==record['sha256']
        write(inputs/'RESTORED.json',restored)
        deadline=started+launch_limit_seconds
        with (output/'console.log').open('x') as log:
            result=subprocess.run([str(PYTHON),'-B',str(HERE/'readout.py'),'--pin',args.pin,
                '--output',str(local/'evaluation'),'--cold',str(output/'evaluation'),
                '--inputs',str(inputs),'--probe',str(output/'gpu-probe.json'),'--stage','qualification',
                '--count','8','--deadline',str(deadline)],env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=max(1,deadline-time.time()))
        rc=result.returncode
        if rc==0 and not (output/'evaluation/COMPLETE.json').exists():
            rc=2
    except BaseException as error:
        write(output/'ERROR.json',dict(error=repr(error),checked_unix=time.time()))
        raise
    finally:
        # Failed audits must not strand physical evidence in an exited container.
        # Preserve it without accepting it as a complete or valid paired result.
        partial_archive = None
        if rc != 0 and (local/'evaluation').exists():
            try:
                from readout import bank_wave
                partial_archive = bank_wave(local/'evaluation', output/'partial-evidence')
            except BaseException as archive_error:
                write(output/'PARTIAL-ARCHIVE-ERROR.json', dict(error=repr(archive_error),
                    local_evidence=str(local/'evaluation'), checked_unix=time.time()))
        write(output/'EXIT.json',dict(exit_code=rc,started_unix=started,ended_unix=time.time(),
            worker_GPU_hours=(time.time()-started)/3600,complete=(output/'evaluation/COMPLETE.json').exists(),
            partial_evidence_archive=partial_archive,partial_evidence_location=str(local/'evaluation')))
    raise SystemExit(rc)


if __name__=='__main__':
    main()
