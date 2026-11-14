"""One allocated GPU per task; at most three GPU-hours for each label pass."""
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

from labels import sha, write


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
PYTHON = ROOT/'setups/robocasa365-native-20261101-r1/env/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pin', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--spec-sha256', required=True)
    p.add_argument('--tasks', nargs='+', required=True)
    args = p.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert args.attempt.isalnum() and len(set(args.tasks)) == len(args.tasks)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip() == args.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    assert sha(args.spec) == args.spec_sha256
    spec = json.loads(args.spec.read_text())
    assert set(args.tasks) <= spec['tasks'].keys()
    local = Path('/tmp')/HERE.name/args.attempt
    cold = Path('/mnt/bos/adgfyl/strong-plant-archives')/HERE.name/args.attempt
    local.mkdir(parents=True,exist_ok=False); cold.mkdir(parents=True,exist_ok=False)
    assert local.stat().st_dev != ROOT.stat().st_dev
    assert shutil.disk_usage(local).free > (20+8*len(args.tasks))*2**30
    probe = cold/'.write-probe';probe.write_text('label pass');assert probe.read_text()=='label pass';probe.unlink()
    env = dict(os.environ, RLINF_PATH=str(ROOT/'sources/critic-selected-a1229239-r1'),
        PYTHONPATH=str(CODE/'src')+':'+str(ROOT/'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(ROOT/'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(ROOT/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',NCCL_P2P_DISABLE='1',FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        assigned=env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible)==len(assigned)==len(args.tasks) and set(visible)==set(assigned)
        env['CUDA_VISIBLE_DEVICES']=','.join(str(i) for i in range(len(visible)))
    began=time.time();deadline=began+10500
    write(cold/'LAUNCH.json',dict(source_commit=args.pin,spec_sha256=args.spec_sha256,tasks=args.tasks,
        hostname=socket.gethostname(),started_unix=began,maximum_GPU_hours=3*len(args.tasks),
        scheduler_maximum_minutes=180,automatic_retry=False,local_free_bytes=shutil.disk_usage(local).free))
    subprocess.run([str(PYTHON),'-B',str(CODE/'scripts/[train]gpu_probe.py'),'--expected-count',str(len(args.tasks)),
        '--baige-container','--output',str(cold/'gpu-probe.json')],env=env,check=True,timeout=180)
    gpu=json.loads((cold/'gpu-probe.json').read_text());assert gpu['status']=='passed'

    def worker(task, device):
        started=time.time();rc=1
        try:
            teacher=spec['tasks'][task]['teacher']
            directory=local/task;directory.mkdir()
            weights=directory/'teacher/full_weights.pt';weights.parent.mkdir()
            shutil.copyfile(teacher['path'],weights);assert sha(weights)==teacher['sha256']
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=str(cold/'gpu-probe.json'),LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            with (cold/(task+'.log')).open('x') as log:
                result=subprocess.run([str(PYTHON),'-B',str(HERE/'labels.py'),'--pin',args.pin,
                    '--spec',str(args.spec),'--spec-sha256',args.spec_sha256,'--task',task,
                    '--output',str(directory/'labels'),'--cold',str(cold/task),'--teacher',str(weights),
                    '--probe',str(cold/'gpu-probe.json'),'--deadline',str(deadline)],
                    env=worker_env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()))
            rc=result.returncode
            if rc==0 and not (cold/task/'COMPLETE.json').exists():rc=2
        except BaseException as error:
            write(cold/(task+'-ERROR.json'),dict(error=repr(error),checked_unix=time.time()))
        finally:
            record=dict(task=task,exit_code=rc,started_unix=started,ended_unix=time.time(),
                complete=(cold/task/'COMPLETE.json').exists())
            write(cold/(task+'-EXIT.json'),record)
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.tasks)) as pool:
        results=list(pool.map(lambda pair:worker(*pair),zip(args.tasks,gpu['devices'],strict=True)))
    complete=all(x['complete'] and x['exit_code']==0 for x in results)
    write(cold/'EXIT.json',dict(completed=complete,results=results,ended_unix=time.time(),
        allocation_GPU_hours=(time.time()-began)*len(args.tasks)/3600))
    raise SystemExit(0 if complete else 1)


if __name__=='__main__':
    main()
