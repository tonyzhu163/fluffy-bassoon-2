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
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank, protocol
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS

root,code,out=map(Path,(os.environ['TASK_ROOT'],os.environ['TASK_META'],os.environ['TASK_OUT']))
start=int(os.environ['TASK_STARTED']); deadline=start+5100; rank=int(os.environ['RANK'])
assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='2' and 0<=rank<4
assert torch.cuda.device_count()==2
if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']='0,1'
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0])/int(cpu[1])==32 and int(Path('/sys/fs/cgroup/memory.max').read_text())==128<<30
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
capacity=os.statvfs(out); assert capacity.f_bavail*capacity.f_frsize>=128<<30
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
        bindings=json.loads((code/'outputs/robocasa-five-task-native-qualification-20261102-r1/BINDINGS.json').read_text())
        retained=bindings['retained_prefix']; assert sha(retained['path'])==retained['sha256']
        previous=json.loads(Path(retained['path']).read_text()); assert len(previous)==8
        qualification=[]
        for item in bindings['tasks']:
            task=item['task']; count=item['headroom_count']; offset=item['headroom_offset']
            profile=copy.deepcopy(pipeline['training'])
            for key in ['resume','resume_sha256','rollout_migration','baseline_validation','calibration_exception','initial_dev_evaluation']:
                profile.pop(key,None)
            output=run/'native'/task; monitor=run/'monitoring'/task
            profile.update(mode='headroom',arm='S',task=task,source_commit=os.environ['TASK_CODE_PIN'],
                output=str(output),monitoring_root=str(monitor),accounting_root=str(run/'actions'),
                attempt='five-task-qualification-'+task,allocated_start_unix=start,deadline_unix=deadline,scheduler_seconds=5340,
                closure_seconds=240,maximum_all_in_actions=204976,rollout_devices=global_devices,
                parallel_controller_cpu_offload=True,episode_id_namespace=task+'-qualification-',
                actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),
                core_qualification=[dict(path=str(code/'outputs/robocasa-five-task-native-qualification-20261102-r1/QUALIFIED-CODE.json'),
                    sha256=sha(code/'outputs/robocasa-five-task-native-qualification-20261102-r1/QUALIFIED-CODE.json'))],
                headroom_cases=case_bank(task,'baseline',count=count,offset=offset),headroom_count=count,headroom_offset=offset,
                motor_seed=item['motor_seed'],evaluation_motor_seeds=dict(baseline=item['motor_seed']),
                distributed_rollout_root=str(dispatch),
                gpu_allocation_starts_unix=[m['allocated_start_unix'] for m in manifests for _ in range(2)])
            profile_path=out/(task+'-profile.json');write_new(profile_path,profile)
            environment=dict(os.environ,**native_env)
            environment.update(CUDA_VISIBLE_DEVICES=global_devices[0]['visibility_token'],LYCO_BAIGE_GPU_MAP=str(probe),
                LYCO_BAIGE_ISOLATED_UUID=global_devices[0]['uuid'],MUJOCO_EGL_DEVICE_ID='0')
            subprocess.run([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_ppo.py'),
                '--profile',str(profile_path)],env=environment,check=True,timeout=max(1,deadline-time.time()))
            fresh=json.loads((output/'episodes.json').read_text()); assert len(fresh)==count
            episodes=(previous if offset else [])+fresh
            assert len(episodes)==len({e['id'] for e in episodes})==50
            assert all(e['camera_order']==list(REWARD_CAMERAS) and len(e['videos'])==3 and e['horizon']==item['horizon']
                and e['frame_clock']['frame_count']==item['horizon']+1 for e in episodes)
            assert all(sha(v['path'])==v['sha256'] for e in fresh for v in e['videos'])
            cases=case_bank(task,'baseline',count=50); rows=[]
            for slot,(case,episode) in enumerate(zip(cases,episodes)):
                directory=Path(retained['monitoring_root']) if slot<offset else monitor/'baseline-000'
                native=json.loads((directory/(episode['id']+'.json')).read_text())
                assert native['case']==case
                rows.append(dict(id=episode['id'],case=case,simulator_final_success=native['simulator_final_success']))
            destination=run/'cohorts'/task;destination.mkdir(parents=True)
            write_new(destination/'episodes.json',episodes);write_new(destination/'CASE-ROWS.json',rows)
            successes=sum(row['simulator_final_success'] for row in rows)
            result=dict(task=task,count=50,successes=successes,native_headroom=.05<=successes/50<=.70,
                protocol=protocol(task,baseline_count=50),motor_seed=item['motor_seed'],retained_prefix_count=offset,
                episodes=dict(path=str(destination/'episodes.json'),sha256=sha(destination/'episodes.json')),
                monitor=dict(path=str(destination/'CASE-ROWS.json'),sha256=sha(destination/'CASE-ROWS.json')),
                robot_controls=50*item['horizon'],new_robot_controls=count*item['horizon'],optimizer_updates=0)
            write_new(destination/'NATIVE-QUALIFICATION.json',result);qualification.append(result)
            actual_bytes=sum(path.stat().st_size for path in run.rglob('*') if path.is_file())
            if actual_bytes>128<<30:raise RuntimeError('Owned finite128GiB output limit reached; retain all evidence, no expansion')
            write_new(run/(task+'-completed.json'),dict(completed=True,result=result,actual_output_bytes=actual_bytes))
        write_new(run/'NATIVE-QUALIFICATION.json',dict(completed=True,tasks=qualification,
            all_tasks_native_headroom=all(row['native_headroom'] for row in qualification),new_episodes=242,
            new_controls=174000,retained_controls=6000,new_all_in_control_cap=204976,optimizer_updates=0,
            scope='Per-task qualification only. No task substitution or five-task training from this ticket.'))
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
