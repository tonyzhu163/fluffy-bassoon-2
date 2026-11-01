"""First retained native three-view cases, through the qualified eight-worker path."""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS

root,code,out=map(Path,(os.environ['TASK_ROOT'],os.environ['TASK_META'],os.environ['TASK_OUT']))
start=int(os.environ['TASK_STARTED']); deadline=start+1560; rank=int(os.environ['RANK'])
assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='2' and 0<=rank<4
assert torch.cuda.device_count()==2
if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']='0,1'
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])==32 and int(Path('/sys/fs/cgroup/memory.max').read_text())==128<<30
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
capacity=os.statvfs(out); assert capacity.f_bavail*capacity.f_frsize>=32<<30
probe=out/'gpu-probe.json'
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','2',
    '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(probe)],check=True,timeout=120)
devices=json.loads(probe.read_text())['devices']
run=root/'outputs'/os.environ['TASK_RUN'];job=run/'job';dispatch=job/'dispatch';dispatch.mkdir(exist_ok=True)
write_new(out/'manifest.json',dict(node_rank=rank,hostname=socket.gethostname(),source_commit=os.environ['TASK_CODE_PIN'],
    allocated_start_unix=start,probe_path=str(probe),devices=devices,cpus=32,ram_gib=128,
    free_bytes=capacity.f_bavail*capacity.f_frsize,quota='UNVERIFIED'))
pipeline=json.loads((root/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
native_env=dict(pipeline['policy_env'],PYTHONPATH=str(code/'src')+':'+os.environ['TASK_RLINF'],ROBOCASA_NATIVE_OUTPUT=str(out))
node_profile=dict(node_rank=rank,devices=devices,dispatch_root=str(dispatch),native_env=native_env,
    probe_path=str(probe),source_commit=os.environ['TASK_CODE_PIN'],deadline_unix=deadline)
write_new(out/'node-profile.json',node_profile)
service=subprocess.Popen([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_node.py'),
    '--profile',str(out/'node-profile.json')],start_new_session=True)
try:
    if rank==0:
        paths=[job/f'node-{i:02d}/manifest.json' for i in range(4)]
        while not all(p.exists() for p in paths):
            if time.time()>min(start+300,deadline-180):raise TimeoutError('All eight real GPUs required before loading')
            time.sleep(1)
        manifests=[json.loads(p.read_text()) for p in paths]
        global_devices=[dict(d,node_rank=i,probe_path=m['probe_path']) for i,m in enumerate(manifests) for d in m['devices']]
        assert len(global_devices)==len({d['uuid'] for d in global_devices})==8
        profile=copy.deepcopy(pipeline['training'])
        for key in ['resume','resume_sha256','rollout_migration','baseline_validation','calibration_exception','initial_dev_evaluation']:
            profile.pop(key,None)
        profile.update(mode='headroom',arm='S',task='OpenDrawer',source_commit=os.environ['TASK_CODE_PIN'],
            output=str(run/'native8'),monitoring_root=str(run/'monitoring'),accounting_root=str(run/'actions'),
            attempt='first8-three-view',allocated_start_unix=start,deadline_unix=deadline,scheduler_seconds=1740,
            closure_seconds=180,maximum_all_in_actions=7024,rollout_devices=global_devices,
            actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),core_qualification=[dict(path=str(code/'outputs/robocasa-three-view-native-first8-20261102-r1/QUALIFIED-CODE.json'),sha256=sha(code/'outputs/robocasa-three-view-native-first8-20261102-r1/QUALIFIED-CODE.json'))],
            headroom_cases=case_bank('OpenDrawer','baseline'),motor_seed=610100,
            evaluation_motor_seeds=dict(baseline=610100),distributed_rollout_root=str(dispatch),
            gpu_allocation_starts_unix=[m['allocated_start_unix'] for m in manifests for _ in range(2)])
        write_new(out/'native-profile.json',profile)
        environment=dict(os.environ,**native_env)
        environment.update(CUDA_VISIBLE_DEVICES=global_devices[0]['visibility_token'],LYCO_BAIGE_GPU_MAP=str(probe),
            LYCO_BAIGE_ISOLATED_UUID=global_devices[0]['uuid'],MUJOCO_EGL_DEVICE_ID='0')
        subprocess.run([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_ppo.py'),
            '--profile',str(out/'native-profile.json')],env=environment,check=True,timeout=max(1,deadline-time.time()))
        episodes_path=run/'native8/episodes.json';episodes=json.loads(episodes_path.read_text())
        assert len(episodes)==8 and all(e['camera_order']==list(REWARD_CAMERAS) and len(e['videos'])==3 for e in episodes)
        assert all(e['horizon']==750 and e['frame_clock']['frame_count']==751 for e in episodes)
        assert all(sha(v['path'])==v['sha256'] for e in episodes for v in e['videos'])
        write_new(run/'THREE-VIEW-FIXTURE.json',dict(completed=True,source_commit=os.environ['TASK_CODE_PIN'],
            episodes=dict(path=str(episodes_path),sha256=sha(episodes_path)),count=8,camera_order=list(REWARD_CAMERAS),
            native_result=json.loads((run/'native8/headroom.json').read_text()),
            scope='First planned8 of OpenDrawer50 qualification; native video software/clock check, not a50-case gate or learning result'))
        write_new(dispatch/'shutdown.json',dict(completed=True))
    else:
        while not (dispatch/'shutdown.json').exists():
            if service.poll() is not None:raise RuntimeError('Owned native worker service exited')
            if time.time()>=deadline:raise TimeoutError('Finite native qualification deadline')
            time.sleep(1)
except BaseException as error:
    write_new(out/'error.json',dict(error=f'{type(error).__name__}: {error}'))
    if rank==0 and not (dispatch/'shutdown.json').exists():write_new(dispatch/'shutdown.json',dict(completed=False,error=str(error)))
    raise
finally:
    if service.poll() is None:service.terminate()
    try:service.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(service.pid,signal.SIGKILL);service.wait()
