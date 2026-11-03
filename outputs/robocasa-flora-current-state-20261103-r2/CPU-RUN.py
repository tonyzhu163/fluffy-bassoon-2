"""Finite allocated-Linux CPU preparation, without weights or CUDA."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import zipfile

root=Path('/mowenjie/fluffy-bassoon-2')
run='robocasa-flora-current-state-20261103-r2'
code=Path(os.environ['TASK_META']);pin=os.environ['TASK_CODE_PIN']
out=root/'outputs'/run
assert not out.exists()
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==pin
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
space=os.statvfs(root);assert space.f_bavail*space.f_frsize>=8<<30
probe=root/'outputs/.flora-current-state-cpu-probe';probe.write_text('ok\n');probe.unlink()
inputs=json.loads((code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json').read_text())
py=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
    PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    PYTHONPATH=str(code/'src')+':'+inputs['official_code'])
started=time.time()
child=subprocess.Popen([str(py),'-B',str(code/'outputs'/run/'SAMPLE.py'),'prepare','--code',str(code)],
    env=env,start_new_session=True)
try:rc=child.wait(timeout=1000)
except BaseException:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    raise
receipt=dict(source_commit=pin,exit_code=rc,seconds=time.time()-started,gpu_seconds=0,
    cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
    memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),free_bytes=space.f_bavail*space.f_frsize)
(Path('/root/workspace')/(run+'-cpu-exit.json')).write_text(json.dumps(receipt,indent=2)+'\n')
if out.exists():(out/'CPU-EXIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert rc==0,f'CPU qualification failed: {rc}; preserve output identity'
paths=sorted(out.glob('*.json'))
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
archive=Path('/root/workspace/flora-current-state-r2-cpu.zip')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,p.name)
    z.writestr('TRANSFER-MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
print('CPU_COMPLETE',json.dumps(receipt),flush=True)
print('CPU_ARCHIVE',archive.name,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
