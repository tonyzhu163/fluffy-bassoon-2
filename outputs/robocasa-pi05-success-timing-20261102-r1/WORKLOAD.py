"""Finite four-node native timing job. No learner, rewarder or optimizer."""
import json
import math
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
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank, PPO_TASKS, TASKS

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
run = root/'outputs'/os.environ['TASK_RUN']; job = run/'job'
start = int(os.environ['TASK_STARTED']); deadline = start+1080; rank = int(os.environ['RANK'])
assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='2' and 0<=rank<4
assert torch.cuda.device_count()==2
if not os.environ.get('CUDA_VISIBLE_DEVICES'): os.environ['CUDA_VISIBLE_DEVICES']='0,1'
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])==64 and int(Path('/sys/fs/cgroup/memory.max').read_text())==256<<30
capacity=os.statvfs(out); assert capacity.f_bavail*capacity.f_frsize>=64<<30
(out/'write-probe').write_text('native timing write probe\n')
probe=out/'gpu-probe.json'
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','2',
    '--baige-container','--output',str(probe)],check=True,timeout=90)
devices=json.loads(probe.read_text())['devices']
write_new(out/'manifest.json',dict(source_commit=os.environ['TASK_CODE_PIN'], hostname=socket.gethostname(),
    node_rank=rank,devices=devices,allocated_start_unix=start,cpus=64,ram_gib=256,
    free_bytes=capacity.f_bavail*capacity.f_frsize,quota='UNVERIFIED'))
manifests=[job/f'node-{i:02d}/manifest.json' for i in range(4)]
while not all(path.exists() for path in manifests):
    if time.time()>min(start+180,deadline-180): raise TimeoutError('Eight actual GPUs required')
    time.sleep(1)
all_devices=[device for path in manifests for device in json.loads(path.read_text())['devices']]
assert len(all_devices)==len({d['uuid'] for d in all_devices})==8
pipeline=json.loads((root/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
native_env=dict(os.environ); native_env.update(pipeline['policy_env'])
native_env['PYTHONPATH']=str(code/'src')+':'+os.environ['TASK_RLINF']
cases=[case for task in PPO_TASKS for case in case_bank(task,'baseline',count=50)]
assert len(cases)==250 and sum(TASKS[c['task']][0] for c in cases)==180000
assert sum(TASKS[c['task']][0]+128 for c in cases)==212000
write_new(out/'all-cases.json',cases)
children=[]; streams=[]
def terminate(signum, frame): raise SystemExit(128+signum)
signal.signal(signal.SIGTERM,terminate); signal.signal(signal.SIGINT,terminate)
try:
    for local,device in enumerate(devices):
        global_rank=rank*2+local; directory=out/f'worker-{local}'
        profile=dict(output=str(directory),rank=global_rank,cases=cases[global_rank::8],code=str(code),
            initializer=pipeline['training']['initializer'], accounting_root=str(run/'actions'),
            deadline_unix=deadline,closure_seconds=90,source_commit=os.environ['TASK_CODE_PIN'])
        path=out/f'profile-{local}.json'; write_new(path,profile)
        env=dict(native_env,CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
            LYCO_BAIGE_GPU_MAP=str(probe),MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        stream=(out/f'worker-{local}.log').open('x');streams.append(stream)
        children.append(subprocess.Popen([sys.executable,'-B',str(code/'outputs'/os.environ['TASK_RUN']/'collect.py'),
            str(path)],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
    while any(child.poll() is None for child in children):
        if any(child.poll() not in (None,0) for child in children): raise RuntimeError('Worker failed; no automatic retry')
        if time.time()>deadline: raise TimeoutError('Finite native timing deadline')
        time.sleep(1)
    assert all(child.returncode==0 for child in children)
    results=[json.loads((out/f'worker-{local}/result.json').read_text()) for local in range(2)]
    write_new(out/'results.json',dict(results=results,exit_codes=[child.returncode for child in children]))
    if rank==0:
        paths=[job/f'node-{i:02d}/results.json' for i in range(4)]
        while not all(path.exists() for path in paths):
            if time.time()>deadline: raise TimeoutError('Incomplete native timing panel')
            time.sleep(1)
        results=[row for path in paths for row in json.loads(path.read_text())['results']]
        rows=[row for result in results for row in result['rows']]
        assert len(rows)==len({(r['case']['task'],r['case']['native_seed']) for r in rows})==250
        journals=[json.loads(p.read_text()) for p in (run/'actions').rglob('*.json')]
        calls=[call for journal in journals for call in journal['calls']]
        assert all(call['resolved'] for call in calls)
        actions=sum(call['executed_or_conservative'] for call in calls)
        assert actions<=212000
        report=[]
        for task in PPO_TASKS:
            cohort=[row for row in rows if row['case']['task']==task]
            assert len(cohort)==50
            metrics={}
            for metric,key in [('final_success','simulator_final_success'),('ever_success','simulator_success_once')]:
                times=sorted(row['first_success_step'] for row in cohort if row[key])
                assert all(type(t) is int and 1<=t<=TASKS[task][0] for t in times)
                p99=times[math.ceil(.99*len(times))-1] if times else None
                metrics[metric]=dict(successes=len(times),first_success_steps=times,
                    empirical_p99=p99,maximum=max(times) if times else None,
                    rounded_up_to50=50*math.ceil(p99/50) if p99 is not None else None)
            candidates=[m['rounded_up_to50'] for m in metrics.values() if m['rounded_up_to50'] is not None]
            conservative=max(candidates) if candidates else None
            report.append(dict(task=task,episodes=50,horizon=TASKS[task][0],metrics=metrics,
                conservative_candidate=conservative,sub750_supported=conservative is not None and conservative<750))
        write_new(run/'TIMING-ROWS.json',rows)
        write_new(run/'RESULT.json',dict(completed=True,source_commit=os.environ['TASK_CODE_PIN'],
            tasks=report,episodes=250,policy_controls=180000,actual_all_in_controls=actions,
            optimizer_updates=0,quantile='nearest rank ceil(.99*n), conditional on success',
            limitation='With under100 successes the empirical p99 is the sample maximum, not population99% coverage.',
            node_results=[dict(path=str(p),sha256=sha(p)) for p in paths]))
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
