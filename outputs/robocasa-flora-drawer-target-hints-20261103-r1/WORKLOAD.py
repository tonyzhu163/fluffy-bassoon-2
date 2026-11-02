"""Qualify one allocated Baige GPU, run the bounded sample, then exit."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

code=Path(os.environ['TASK_META']);run=os.environ['TASK_RUN']
root=Path('/mowenjie/fluffy-bassoon-2');out=root/'outputs'/run/'allocation'
out.mkdir(exist_ok=False)
assert socket.gethostname().startswith('job-')
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
assert int(os.environ['WORLD_SIZE'])==int(os.environ['NPROC_PER_NODE'])==1
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert int(cpu[0])/int(cpu[1])>=32
assert int(Path('/sys/fs/cgroup/memory.max').read_text())>=128<<30
space=os.statvfs(out);assert space.f_bavail*space.f_frsize>=8<<30
p=out/'write-probe';p.write_text('sample output probe\n');p.unlink()
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','1',
    '--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')],check=True,timeout=120)
device=json.loads((out/'gpu-probe.json').read_text())['devices'][0]
inputs=json.loads((code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json').read_text())
py=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python';toolkit=Path(inputs['toolkit']);cache=out/'cache'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_GPU_MAP=str(out/'gpu-probe.json'),
    LYCO_BAIGE_ISOLATED_UUID=device['uuid'],PYTHONPATH=str(code/'src')+':'+inputs['official_code'],
    CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),CUDACXX=str(toolkit/'bin/nvcc'),
    PATH=str(toolkit/'bin')+':'+str(py.parent)+':'+os.environ['PATH'],
    LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
    LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),
    TMPDIR='/tmp/flora-drawer-hints-r1',XDG_CACHE_HOME=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'torch'),
    FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),
    HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MAX_JOBS='4')
Path(env['TMPDIR']).mkdir(exist_ok=True)
remaining=int(os.environ['TASK_STARTED'])+810-int(time.time());assert remaining>0
script=code/'outputs'/run/'SAMPLE.py'
with (out/'scorer.log').open('x') as stream:
    child=subprocess.Popen([str(py),'-B',str(script),'worker','--code',str(code)],env=env,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:rc=child.wait(timeout=remaining)
    except BaseException:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        raise
assert rc==0,f'Sample scorer failed: {rc}; preserve partial evidence, no retry'
subprocess.run([sys.executable,'-B',str(script),'summary','--code',str(code)],check=True,timeout=30)
