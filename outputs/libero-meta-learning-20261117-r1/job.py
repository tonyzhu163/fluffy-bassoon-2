"""Two independent finite qualification workers on scheduler-owned GPUs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
CODE=HERE.parents[1]
ROOT=Path('/mowenjie/fluffy-bassoon-2')
SETUP=ROOT/'setups/libero-meta-learning-20261117-r1'
PRO=ROOT/'sources/libero-pro2-e5500c69'
RUNTIME=ROOT/'sources/critic-selected-a1229239-r1'
OPENPI=ROOT/'outputs/robocasa-sft-reproduction-20261109-r1/sources/openpi'
PYTHON=SETUP/'env/bin/python'
POLICY_PYTHON=ROOT/'setups/robocasa365-native-20261101-r1/env/bin/python'


def environment():
    return dict(os.environ,PYTHONPATH=f'{CODE}/src:{RUNTIME}:{OPENPI}/src:{OPENPI}/packages/openpi-client/src:{PRO}:{SETUP}/simulator-source:{HERE}',
        RLINF_PATH=str(RUNTIME),LIBERO_PRO_ROOT=str(SETUP/'simulator-source'),
        LIBERO_CONFIG_PATH=str(SETUP/'libero-config'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',
        MUJOCO_EGL_DEVICE_ID='0',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        NCCL_P2P_DISABLE='1',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
        OPENPI_DATA_HOME=str(ROOT/'outputs/robocasa-sft-reproduction-20261109-r1/openpi-cache'),
        JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONNOUSERSITE='1',
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--minutes',type=int,default=45)
    a=p.parse_args()
    assert a.attempt.isalnum() and 10<=a.minutes<=55
    assert socket.gethostname().startswith('job-')
    for path,pin in ((CODE,a.pin),(RUNTIME,'8f9bfa7661928ef8c17494b04f4d7bf3125634e1'),
                     (PRO,'e5500c690c1498eca1b7050350654964190cdf70')):
        assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    slots=ROOT/'outputs'/HERE.name/a.attempt; slots.mkdir(parents=True,exist_ok=True)
    for rank in range(2):
        out=slots/f'rank{rank}'
        try:
            out.mkdir(); break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('Both slots claimed; retries need a fresh attempt identity')
    free=shutil.disk_usage(out).free
    assert free>4*2**30  # Eight short compressed episodes plus ample recovery space.
    probe=out/'write-probe'; probe.write_text('probe'); probe.unlink()
    env=environment()
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        assert len(visible)==1 and visible==env['NVIDIA_VISIBLE_DEVICES'].split(',')
        env['CUDA_VISIBLE_DEVICES']='0'
    started=time.time(); rc=1
    receipt=dict(source=a.pin,rank=rank,hostname=socket.gethostname(),started=started,
        maximum_gpu_hours=a.minutes/60,campaign_cap=200,free_bytes=free)
    (out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        with (out/'run.log').open('w') as log:
            subprocess.run([str(PYTHON),'-B',str(CODE/'scripts/[train]gpu_probe.py'),
                '--expected-count','1','--purpose','training','--baige-container',
                '--baige-exclusive-training','--output',str(out/'gpu.json')],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            env['LYCO_BAIGE_GPU_MAP']=str(out/'gpu.json')
            rc=subprocess.run([str(POLICY_PYTHON),'-Bu',str(HERE/'qualify.py'),'--rank',str(rank),
                '--output',str(out/'native'),'--deadline',str(started+(a.minutes-3)*60)],
                env=env,stdout=log,stderr=subprocess.STDOUT,
                timeout=max(1,started+a.minutes*60-time.time())).returncode
    finally:
        (out/'terminal.json').write_text(json.dumps(dict(receipt,ended=time.time(),exit_code=rc),indent=2)+'\n')
    while rc==0:
        terminals=[slots/f'rank{i}'/'terminal.json' for i in range(2)]
        if any(p.exists() and json.loads(p.read_text())['exit_code']!=0 for p in terminals):
            rc=1; break
        if all(p.exists() for p in terminals):
            break
        if time.time()>=started+a.minutes*60:
            rc=124; break
        time.sleep(5)
    (out/'allocation-end.json').write_text(json.dumps(dict(ended=time.time(),exit_code=rc,
        worker_gpu_hours=(time.time()-started)/3600),indent=2)+'\n')
    return rc


if __name__=='__main__':
    sys.exit(main())
