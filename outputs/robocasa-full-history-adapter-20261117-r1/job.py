"""One finite qualification shard per scheduler GPU; no automatic retries."""
import argparse
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
ROOT = Path('/mowenjie/fluffy-bassoon-2')
PYTHON = ROOT/'setups/robocasa365-native-20261101-r1/env/bin/python'
STUDY = ROOT/'outputs'/HERE.name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--attempt', required=True)
    p.add_argument('--minutes', type=int, default=55)
    p.add_argument('--workers', type=int, choices=range(1,5), default=4)
    p.add_argument('--rank-offset', type=int, default=0)
    p.add_argument('--stage', choices=('qualify','privileged','collect','fit','fixed','authority'), default='qualify')
    p.add_argument('--scene', type=int)
    p.add_argument('--split', choices=('train','development','test'))
    p.add_argument('--corpus', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--scale', type=float, default=.1)
    p.add_argument('--privileged-updates', type=int, default=1024)
    a = p.parse_args()
    assert a.attempt.isalnum() and 10 <= a.minutes <= (60 if a.stage == 'qualify' else 240)
    assert a.stage != 'privileged' or (a.workers == 1 and a.rank_offset == 0)
    assert (a.split is not None) == (a.stage == 'collect')
    assert (a.corpus is not None) == (a.stage == 'fit')
    assert a.stage != 'fixed' or (a.workers == 3 and a.rank_offset == 0)
    assert (a.checkpoint is not None) == (a.stage == 'authority')
    assert a.stage != 'authority' or (a.workers == 1 and a.rank_offset == 0)
    assert a.scene is None or (a.stage == 'qualify' and a.workers == 1)
    assert 0 <= a.rank_offset and a.rank_offset + a.workers <= (6 if a.stage == 'fit' else 4)
    assert socket.gethostname().startswith('job-')
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip() == a.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    slots = STUDY/a.attempt
    slots.mkdir(parents=True, exist_ok=True)
    ranks = range(a.rank_offset, a.rank_offset + a.workers)
    for rank in ranks:
        out = slots/f'rank{rank}'
        try:
            out.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('All slots already claimed; retries need a fresh attempt ID')
    free = shutil.disk_usage(STUDY).free
    assert free > 16*2**30  # Four shards need at most ~6 GB of decision images; retain recovery space.
    probe_write = out/'write-probe'; probe_write.write_text('probe'); probe_write.unlink()
    runtime = ROOT/'sources/critic-selected-a1229239-r1'
    base = ROOT/'outputs/robocasa-sft-reproduction-20261109-r1'
    env = dict(os.environ, RLINF_PATH=str(runtime),
        PYTHONPATH=f'{CODE}/src:{runtime}:{base}/sources/openpi/src:{base}/sources/openpi/packages/openpi-client/src:{base}/sources/robocasa:{HERE}',
        MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID='0',
        PYTHONHASHSEED='73', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
        NCCL_P2P_DISABLE='1', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
        OPENPI_DATA_HOME=str(base/'openpi-cache'), ROBOCASA_ASSETS_PATH=str(ROOT/'setups/robocasa365-native-20261101-r1/assets'),
        JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false', PYTHONNOUSERSITE='1')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        assert len(visible)==1 and visible == env['NVIDIA_VISIBLE_DEVICES'].split(',')
        env['CUDA_VISIBLE_DEVICES']='0'
    started=time.time()
    receipt=dict(source=a.pin, rank=rank, hostname=socket.gethostname(), started=started,
                 maximum_gpu_hours=a.minutes/60, campaign_gpu_hour_cap=200, free_bytes=free)
    (out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
    rc=1
    try:
        with (out/'run.log').open('w') as log:
            subprocess.run([str(PYTHON),'-B',str(CODE/'scripts/[train]gpu_probe.py'),
                '--expected-count','1','--purpose','training','--baige-container',
                '--baige-exclusive-training','--output',str(out/'gpu.json')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
            env['LYCO_BAIGE_GPU_MAP']=str(out/'gpu.json')
            if a.stage == 'qualify':
                command=[str(PYTHON),'-Bu',str(HERE/'qualify.py'),'--pin',a.pin,'--rank',str(rank),
                    '--output',str(out/'native'),'--deadline',str(started+(a.minutes-3)*60)]
                if a.scene is not None: command += ['--scene',str(a.scene)]
            elif a.stage == 'collect':
                command=[str(PYTHON),'-Bu',str(HERE/'collect.py'),'--pin',a.pin,'--rank',str(rank),
                    '--output',str(out/'native'),'--split',a.split,
                    '--deadline',str(started+(a.minutes-5)*60)]
            elif a.stage == 'fixed':
                command=[str(PYTHON),'-Bu',str(HERE/'fixed.py'),'--pin',a.pin,'--rank',str(rank),
                    '--output',str(out/'native'),'--deadline',str(started+(a.minutes-5)*60)]
            elif a.stage == 'authority':
                command=[str(PYTHON),'-Bu',str(HERE/'authority.py'),'--pin',a.pin,
                    '--checkpoint',str(a.checkpoint),'--manifest',str(STUDY/'priv01/rank0/corpus.json'),
                    '--output',str(out/'authority')]
            elif a.stage == 'fit':
                command=[str(PYTHON),'-Bu',str(HERE/'fit.py'),'--pin',a.pin,
                    '--manifest',str(a.corpus),'--output',str(out/'fit'),
                    '--mode','full' if rank % 2 == 0 else 'current','--seed',str(rank//2),
                    '--updates','4096','--execution-batch','8',
                    '--deadline',str(started+(a.minutes-5)*60)]
            else:
                trials = []
                for index in range(8):
                    source_rank = index % 4
                    phase = 'qual03' if source_rank == 0 else 'qual06' if source_rank == 2 else 'qual04'
                    source = STUDY/phase/f'rank{source_rank}'/'native'
                    for fraction in (0,1):
                        episode = source/f'condition-{index:02d}-f{fraction}'
                        assert (episode/'result.json').exists() and (episode/'episode.npz').exists()
                        trials.append([str(episode)])
                manifest = out/'corpus.json'
                manifest.write_text(json.dumps(dict(trials=trials,
                    role='privileged engineering fit; training conditions on old development scenes only'),indent=2)+'\n')
                command=[str(PYTHON),'-Bu',str(HERE/'fit.py'),'--pin',a.pin,
                    '--manifest',str(manifest),'--output',str(out/'fit'),'--mode','privileged',
                    '--updates',str(a.privileged_updates),'--scale',str(a.scale),'--execution-batch','8',
                    '--native-eval',
                    '--deadline',str(started+(a.minutes-5)*60)]
            result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,started+a.minutes*60-time.time()))
            rc=result.returncode
    finally:
        (out/'terminal.json').write_text(json.dumps(dict(receipt,ended=time.time(),exit_code=rc),indent=2)+'\n')
    # A PyTorchJob can end when master exits; wait until every shard is banked.
    while rc == 0:
        terminals = [slots/f'rank{i}'/'terminal.json' for i in ranks]
        if any(p.exists() and json.loads(p.read_text())['exit_code'] != 0 for p in terminals):
            rc = 1
            break
        if all(p.exists() for p in terminals):
            break
        if time.time() >= started + a.minutes*60:
            rc = 124
            break
        time.sleep(5)
    (out/'allocation-end.json').write_text(json.dumps(dict(ended=time.time(),exit_code=rc,
        charged_gpu_hours=(time.time()-started)/3600),indent=2)+'\n')
    raise SystemExit(rc)


if __name__=='__main__':
    main()
