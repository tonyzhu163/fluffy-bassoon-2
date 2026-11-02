"""One finite8-GPU job; two independent replicas per lane count, no SGD."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import write_new
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
run = root/'outputs'/os.environ['TASK_RUN']; job = run/'job'
start = int(os.environ['TASK_STARTED']); deadline = start+1620
rank = int(os.environ['RANK'])
assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='2' and 0<=rank<4
assert torch.cuda.device_count()==2
if not os.environ.get('CUDA_VISIBLE_DEVICES'): os.environ['CUDA_VISIBLE_DEVICES']='0,1'
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])==64 and int(Path('/sys/fs/cgroup/memory.max').read_text())==256<<30
capacity=os.statvfs(out); assert capacity.f_bavail*capacity.f_frsize>=32<<30
(out/'write-probe').write_text('capacity benchmark write probe\n')
probe=out/'gpu-probe.json'
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','2',
    '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(probe)],check=True,timeout=90)
devices=json.loads(probe.read_text())['devices']
write_new(out/'manifest.json',dict(source_commit=os.environ['TASK_CODE_PIN'], hostname=socket.gethostname(),
    node_rank=rank,devices=devices,allocated_start_unix=start,cpus=64,ram_gib=256,
    free_bytes=capacity.f_bavail*capacity.f_frsize,quota='UNVERIFIED',batch_size=(1,2,4,8)[rank]))
manifests=[job/f'node-{i:02d}/manifest.json' for i in range(4)]
while not all(path.exists() for path in manifests):
    if time.time()>min(start+240,deadline-180): raise TimeoutError('Eight actual GPUs required')
    time.sleep(1)
all_devices=[device for path in manifests for device in json.loads(path.read_text())['devices']]
assert len(all_devices)==len({d['uuid'] for d in all_devices})==8
pipeline=json.loads((root/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
native_env=dict(os.environ,**pipeline['policy_env'],PYTHONPATH=str(code/'src')+':'+os.environ['TASK_RLINF'])
qualification=json.loads((code/'outputs/robocasa-rollout-batch-capacity-20261102-r1/CPU-QUALIFICATION.json').read_text())
assert qualification['completed'] and qualification['tests']==5 and qualification['failures']==qualification['errors']==0
assert sha(code/qualification['motor_source']['path'])==qualification['motor_source']['sha256']
assert sha(qualification['xml']['path'])==qualification['xml']['sha256']
cases=case_bank('OpenDrawer','baseline',count=8)
for index,case in enumerate(cases):
    case.update(bank='capacity_benchmark',native_seed=6_600_100+index,slot=index)
write_new(out/'cases.json',cases)
children=[]; streams=[]

def terminate(signum, frame): raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
try:
    for local,device in enumerate(devices):
        global_rank=rank*2+local; directory=out/f'worker-{local}'
        profile=dict(output=str(directory),rank=global_rank,batch_size=(1,2,4,8)[rank],cases=cases,
            initializer=pipeline['training']['initializer'],
            initial_checkpoint=str(root/'outputs/robocasa-residual-approved-20261101-r1/S/boundary-000.pt'),
            accounting_root=str(run/'actions'),deadline_unix=deadline,closure_seconds=120,
            source_commit=os.environ['TASK_CODE_PIN'])
        profile_path=out/f'profile-{local}.json';write_new(profile_path,profile)
        env=dict(native_env,CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
            LYCO_BAIGE_GPU_MAP=str(probe),MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        stream=(out/f'worker-{local}.log').open('x');streams.append(stream)
        children.append(subprocess.Popen([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_rollout_batch_benchmark.py'),
            '--profile',str(profile_path)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    while any(child.poll() is None for child in children):
        if time.time()>deadline: raise TimeoutError('Finite benchmark deadline')
        time.sleep(1)
    results=[json.loads((out/f'worker-{local}/result.json').read_text()) for local in range(2)]
    write_new(out/'results.json',dict(results=results,exit_codes=[child.returncode for child in children]))
    if not all(row['completed'] for row in results): raise RuntimeError('A configuration failed; preserve its result without retry')
    if rank==0:
        paths=[job/f'node-{i:02d}/results.json' for i in range(4)]
        while not all(path.exists() for path in paths):
            if time.time()>deadline: raise TimeoutError('Incomplete capacity sweep')
            time.sleep(1)
        results=[row for path in paths for row in json.loads(path.read_text())['results']]
        summary=[]
        for batch in (1,2,4,8):
            rows=[row for row in results if row['batch_size']==batch]
            assert len(rows)==2
            summary.append(dict(batch_size=batch,replicates=2,
                controls_per_second_each=[row['controls_per_second'] for row in rows],
                aggregate_controls_per_second=sum(row['controls_per_second'] for row in rows),
                peak_nvml_mib_each=[row['observed_memory']['peak_nvml_mib'] for row in rows],
                collection_seconds_each=[row['collection_wall_seconds'] for row in rows]))
        write_new(run/'RESULT.json',dict(completed=True,source_commit=os.environ['TASK_CODE_PIN'],
            configurations=summary,optimizer_updates=0,physical_episodes=64,policy_controls=48000,
            actual_all_in_controls=sum(row['actual_all_in_controls'] for row in results),
            interpretation='Fixed eight-case OpenDrawer speed test; twoGPU replicas per setting. No new calibration/learning evidence.'))
finally:
    for child in children:
        if child.poll() is None:
            try: os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError: pass
    for child in children:
        try: child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
    for stream in streams: stream.close()
