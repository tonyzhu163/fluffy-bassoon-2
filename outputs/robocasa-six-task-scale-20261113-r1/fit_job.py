"""Independent methods on one fixed corpus, sharing one cache per allocation."""
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

from corpus import TASKS
from labels import sha, write

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
PYTHON = ROOT/'setups/robocasa365-native-20261101-r1/env/bin/python'
METHODS = ('ad','dpt','dit','ic_iql')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',required=True)
    p.add_argument('--attempt',required=True)
    p.add_argument('--corpora',type=Path,required=True)
    p.add_argument('--corpora-sha256',required=True)
    p.add_argument('--corpus-size',type=int,choices=(3,6),required=True)
    p.add_argument('--data-qualification',type=Path,required=True)
    p.add_argument('--data-qualification-sha256',required=True)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    a=p.parse_args()
    assert sys.platform=='linux' and socket.gethostname().startswith('job-')
    assert a.attempt.isalnum()
    assert len(a.methods)==len(set(a.methods))
    gpu_count=len(a.methods)
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==a.pin
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'])
    assert sha(a.corpora)==a.corpora_sha256
    assert sha(a.data_qualification)==a.data_qualification_sha256
    qualification=json.loads(a.data_qualification.read_text())
    assert qualification['passed'] and qualification['CPU_only']
    # Scheduling-only changes may split independent methods across allocations.
    # Reuse qualification only when every checked/model-execution module is
    # identical to the receipt's actual source, verified through Git objects.
    qualified_pin=qualification['source_commit']
    subprocess.run(['git','-C',str(CODE),'diff','--exit-code',qualified_pin,a.pin,'--',
        'src','benchmarks','scripts',str(HERE.relative_to(CODE)/'fit.py'),
        str(HERE.relative_to(CODE)/'corpus.py'),str(HERE.relative_to(CODE)/'labels.py'),
        str(HERE.relative_to(CODE)/'qualify_corpora.py')],check=True,stdout=subprocess.DEVNULL)
    assert qualification['corpora']==dict(path=str(a.corpora),sha256=a.corpora_sha256)
    assert qualification['sampler_sha256']==sha(HERE/'corpus.py')
    assert {(cell['corpus_size'],cell['objective'],cell['split']) for cell in qualification['checked_cells']}=={
        (size,method,split) for size in (3,6) for method in METHODS for split in ('train','validation')}
    spec=json.loads(a.corpora.read_text());tasks=TASKS[:a.corpus_size]
    sources={}
    for task in tasks:
        ref=spec['tasks'][task];assert sha(ref['path'])==ref['sha256']
        assembly=json.loads(Path(ref['path']).read_text())
        assert assembly['student_fit_ready'] and assembly['task']==task
        sources[task]=assembly['sources'][task]
    local=Path('/tmp')/HERE.name/a.attempt
    cold=Path('/mnt/bos/adgfyl/strong-plant-archives')/HERE.name/a.attempt
    local.mkdir(parents=True,exist_ok=False);cold.mkdir(parents=True,exist_ok=False)
    assert local.stat().st_dev!=ROOT.stat().st_dev
    needed=sum(Path(source['archive']).stat().st_size for source in sources.values())+48*2**30
    assert shutil.disk_usage(local).free>needed
    check=cold/'.write-probe';check.write_text('matched KV fits');assert check.read_text()=='matched KV fits';check.unlink()
    env=dict(os.environ,RLINF_PATH=str(ROOT/'sources/critic-selected-a1229239-r1'),
        PYTHONPATH=str(CODE/'src')+':'+str(ROOT/'sources/critic-selected-a1229239-r1'),
        ROBOCASA_ASSETS_PATH=str(ROOT/'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(ROOT/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',NCCL_P2P_DISABLE='1',FLUFFY_GPU_EXECUTION='baige',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',WANDB_MODE='disabled')
    if 'CUDA_VISIBLE_DEVICES' not in env:
        visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
        assigned=env['NVIDIA_VISIBLE_DEVICES'].split(',')
        assert len(visible)==len(assigned)==gpu_count and set(visible)==set(assigned)
        env['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(gpu_count)))
    began=time.time();deadline=began+20700
    write(cold/'LAUNCH.json',dict(source_commit=a.pin,corpus_size=a.corpus_size,methods=a.methods,
        hostname=socket.gethostname(),started_unix=began,maximum_GPU_hours=6*gpu_count,
        maximum_scheduler_minutes=360,automatic_retries=False,required_local_bytes=needed,
        local_free_bytes=shutil.disk_usage(local).free,corpus_spec_sha256=a.corpora_sha256,
        qualification_code_equivalence=dict(source_commit=qualified_pin,checked_modules_unchanged=True),
        data_qualification=dict(path=str(a.data_qualification),sha256=a.data_qualification_sha256)))
    subprocess.run([str(PYTHON),'-B',str(CODE/'scripts/[train]gpu_probe.py'),'--expected-count',str(gpu_count),
        '--purpose','training','--baige-container','--baige-exclusive-training',
        '--output',str(cold/'gpu-probe.json')],env=env,check=True,timeout=180)
    probe=json.loads((cold/'gpu-probe.json').read_text());assert probe['status']=='passed'
    raw=local/'raw';raw.mkdir();cache={}
    for task,source in sources.items():
        dest=raw/task/'raw.tar';dest.parent.mkdir()
        shutil.copyfile(source['archive'],dest)
        assert sha(dest)==source['archive_sha256']
        cache[task]=dict(path=str(dest),sha256=source['archive_sha256'],bytes=dest.stat().st_size)
        write(cold/'STAGING.json',dict(completed_tasks=list(cache),tasks=list(tasks),checked_unix=time.time()))
    write(raw/'CACHE.json',dict(completed=True,tasks=cache,all_archives_rehashed=True))
    cache_sha=sha(raw/'CACHE.json')

    def worker(pair):
        method,device=pair;started=time.time();rc=1
        try:
            worker_env=dict(env,CUDA_VISIBLE_DEVICES=device['visibility_token'],
                LYCO_BAIGE_GPU_MAP=str(cold/'gpu-probe.json'),LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
            with (cold/(method+'.log')).open('x') as log:
                rc=subprocess.run([str(PYTHON),'-B',str(HERE/'fit.py'),'--pin',a.pin,
                    '--corpora',str(a.corpora),'--corpora-sha256',a.corpora_sha256,
                    '--corpus-size',str(a.corpus_size),'--objective',method,
                    '--probe',str(cold/'gpu-probe.json'),'--output',str(local/method),
                    '--cold',str(cold/method),'--deadline',str(deadline),'--execution-batch','8',
                    '--raw-root',str(raw),'--raw-receipt-sha256',cache_sha],env=worker_env,
                    stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline+300-time.time())).returncode
        except BaseException as error:
            write(cold/(method+'-ERROR.json'),dict(error=repr(error),checked_unix=time.time()))
        complete=(cold/method/'COMPLETE.json').exists()
        result=dict(method=method,exit_code=rc,complete=complete,started_unix=started,ended_unix=time.time())
        write(cold/(method+'-EXIT.json'),result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count) as pool:
        results=list(pool.map(worker,zip(a.methods,probe['devices'],strict=True)))
    complete=all(x['exit_code']==0 and x['complete'] for x in results)
    write(cold/'EXIT.json',dict(completed=complete,results=results,ended_unix=time.time(),
        allocation_GPU_hours=(time.time()-began)*gpu_count/3600,
        incomplete_policy='Resume preserved full state under a separately reconciled reservation; never publish a shorter fit as10000 updates'))
    raise SystemExit(0 if complete else 1)


if __name__=='__main__':main()
