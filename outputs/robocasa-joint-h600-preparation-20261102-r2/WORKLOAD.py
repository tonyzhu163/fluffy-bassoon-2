"""Approved finite600-control calibration and conditional common-before; no SGD."""
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
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest, sha, write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import (
    PPO_TASKS, case_bank, private_action_seed, protocol, require_calibration_permission,
)

ROOT=Path('/mowenjie/fluffy-bassoon-2')
CODE=Path(os.environ['TASK_META'])
RUN=ROOT/'outputs'/os.environ['TASK_RUN']
JOB=RUN/'job';EXCHANGE=RUN/'calibration';DISPATCH=JOB/'dispatch'
RANK=int(os.environ['RANK']);START=int(os.environ['TASK_STARTED'])
OUT=Path(os.environ['TASK_OUT']);DEADLINE=START+6300
children=[];streams=[];service=None;controller=None;last_size=0.


def stop(child):
    if child is None or child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait()


def spawn(python,script,args,env,name):
    log=(OUT/(name+'.log')).open('x');streams.append(log)
    child=subprocess.Popen([str(python),'-B',str(script),*args],env=env,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(child);return child


def check():
    global last_size
    if time.time()>=DEADLINE-90:raise TimeoutError('Approved finite resource clock; preserve partial evidence')
    if list((JOB/'errors').glob('*.json')) or list((EXCHANGE/'errors').glob('*.json')):
        raise RuntimeError('An owned preparation worker failed; no retry or scope expansion')
    if service is not None and service.poll() is not None:
        raise RuntimeError('Owned native node service exited before shutdown')
    if any(child.poll() not in (None,0) for child in children):
        raise RuntimeError('An owned scorer/controller failed')
    if RANK==0 and time.monotonic()-last_size>60:
        size=int(subprocess.check_output(['du','-sk',str(RUN)],text=True,timeout=30).split()[0])*1024
        if size>128<<30:raise RuntimeError('Approved128GiB output band exceeded; retain evidence and stop')
        print(json.dumps(dict(phase='finite_preparation_watch',used_output_bytes=size,deadline_unix=DEADLINE)),flush=True)
        last_size=time.monotonic()


def wait(paths,limit=None):
    while not all(path.exists() for path in paths):
        if limit is not None and time.time()>=limit:raise TimeoutError('Finite startup barrier expired')
        check();time.sleep(1)


def run_controller(profile,native_env,name):
    global controller
    path=OUT/(name+'-profile.json');write_new(path,profile)
    env=dict(os.environ,**native_env)
    device=profile['rollout_devices'][0]
    env.update(CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_GPU_MAP=device['probe_path'],
        LYCO_BAIGE_ISOLATED_UUID=device['uuid'],MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
    controller=spawn(sys.executable,CODE/'scripts/diagnostics/robocasa_residual_ppo.py',
        ['--profile',str(path)],env,name)
    try:
        while controller.poll() is None:check();time.sleep(1)
        if controller.returncode:raise RuntimeError(f'{name} failed with exit{controller.returncode}')
    finally:stop(controller)


def main():
    global service,DEADLINE
    authority=read_ref(dict(path=str(CODE/'outputs/robocasa-joint-h600-preparation-20261102-r2/HUMAN-RESOURCE-AUTHORITY.json'),
        sha256=sha(CODE/'outputs/robocasa-joint-h600-preparation-20261102-r2/HUMAN-RESOURCE-AUTHORITY.json')))
    assert authority['approved'] and authority['maximum_gpus']==4 and authority['maximum_wall_seconds']==6360
    assert authority['maximum_gpu_seconds']==25440 and authority['optimizer_updates']==0
    assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='1' and 0<=RANK<4
    assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==1
    assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
    assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain'],text=True).strip()
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert int(cpu[0])/int(cpu[1])>=32 and int(Path('/sys/fs/cgroup/memory.max').read_text())>=128<<30
    assert os.statvfs('/dev/shm').f_blocks*os.statvfs('/dev/shm').f_frsize>=16<<30
    free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize;assert free>=128<<30
    probe_write=OUT/'write-probe'
    with probe_write.open('x') as stream:stream.write('fresh allocated namespace\n');stream.flush();os.fsync(stream.fileno())
    probe_write.unlink()
    probe=OUT/'gpu-probe.json'
    subprocess.run([sys.executable,'-B',str(CODE/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(probe)],check=True,timeout=120)
    devices=json.loads(probe.read_text())['devices']
    assert len(devices)==1 and all(d['inventory']['memory_total_mib']>=90000 for d in devices)
    write_new(OUT/'manifest.json',dict(node_rank=RANK,hostname=socket.gethostname(),source_commit=os.environ['TASK_CODE_PIN'],
        allocated_start_unix=START,devices=devices,probe_path=str(probe),cpus=32,ram_gib=128,shm_gib=16,
        free_bytes=free,write_probe_pass=True,quota='UNVERIFIED',resource_authority_sha256=digest(authority)))
    manifests=[JOB/f'node-{r:02d}/manifest.json' for r in range(4)];wait(manifests,START+300)
    manifests=[json.loads(p.read_text()) for p in manifests]
    assert [m['node_rank'] for m in manifests]==list(range(4))
    assert all(m['source_commit']==os.environ['TASK_CODE_PIN'] and m['resource_authority_sha256']==digest(authority) for m in manifests)
    global_devices=[dict(d,node_rank=r,probe_path=m['probe_path']) for r,m in enumerate(manifests) for d in m['devices']]
    assert len(global_devices)==len({d['uuid'] for d in global_devices})==4
    DEADLINE=min(m['allocated_start_unix'] for m in manifests)+6300
    pipeline=json.loads((ROOT/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
    native_env=dict(pipeline['policy_env'])
    for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','LYCO_BAIGE_GPU_MAP','LYCO_BAIGE_ISOLATED_UUID','MUJOCO_EGL_DEVICE_ID'):
        native_env.pop(key,None)
    native_env.update(PYTHONPATH=str(CODE/'src')+':'+os.environ['TASK_RLINF'],RLINF_PATH=os.environ['TASK_RLINF'],
        ROBOCASA_NATIVE_OUTPUT=str(OUT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    node_profile=dict(node_rank=RANK,devices=devices,dispatch_root=str(DISPATCH),native_env=native_env,
        probe_path=str(probe),source_commit=os.environ['TASK_CODE_PIN'],deadline_unix=DEADLINE)
    node_path=OUT/'node-profile.json';write_new(node_path,node_profile)
    service=spawn(sys.executable,CODE/'scripts/diagnostics/robocasa_residual_node.py',['--profile',str(node_path)],
        dict(os.environ),'native-service')
    flora=json.loads((CODE/'outputs/robocasa-flora-throughput-20261102-r2/INPUTS.json').read_text())
    warmup=RUN/'warmup-episodes.json';assert len(json.loads(warmup.read_text()))==8
    warmup_ref=dict(path=str(warmup),sha256=sha(warmup))
    scorer_python=ROOT/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
    scorer_children=[]
    for local,device in enumerate(devices):
        rank=RANK+local
        profile=dict(mode='fifty_case_baseline_only',annotation_world=4,annotation_rank=rank,
            source_commit=os.environ['TASK_CODE_PIN'],output=str(OUT/f'scorer-{rank:02d}'),exchange=str(EXCHANGE),
            deadline_unix=DEADLINE,closure_seconds=180,batch_size=8,gpu_memory_utilization=.75,
            model=flora['model'],official_code=flora['official_code'],model_receipt=flora['model_receipt'],
            actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),expected_gpu_uuid=device['uuid'],
            warmup_episodes=warmup_ref,warmup_protocol=protocol('CoffeeSetupMug',baseline_count=50),
            cohorts=[dict(task=task,native_qualification=str(RUN/'cohorts'/task/'NATIVE-QUALIFICATION.json')) for task in PPO_TASKS])
        path=OUT/f'scorer-{rank:02d}-profile.json';write_new(path,profile)
        toolkit=Path(flora['toolkit']);cache=OUT/f'scorer-{rank:02d}-cache'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_GPU_MAP=str(probe),
            LYCO_BAIGE_ISOLATED_UUID=device['uuid'],PYTHONPATH=str(CODE/'src')+':'+flora['official_code'],
            CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),CUDACXX=str(toolkit/'bin/nvcc'),
            PATH=str(toolkit/'bin')+':'+str(scorer_python.parent)+':'+os.environ['PATH'],
            LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),
            XDG_CACHE_HOME=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'torch'),
            FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),VLLM_CACHE_ROOT=str(cache/'vllm'),
            TRITON_CACHE_DIR=str(cache/'triton'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MAX_JOBS='4',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
        scorer_children.append(spawn(scorer_python,CODE/'scripts/diagnostics/robocasa_flora_fifty_score.py',
            ['--profile',str(path)],env,f'scorer-{rank:02d}'))
    if RANK!=0:
        while not (DISPATCH/'shutdown.json').exists():
            if all(c.poll()==0 for c in scorer_children) and not (OUT/'scorers-closed.json').exists():
                write_new(OUT/'scorers-closed.json',dict(completed=True,annotation_ranks=[RANK]))
            check();time.sleep(1)
        if not json.loads((DISPATCH/'shutdown.json').read_text())['completed']:raise RuntimeError('Coordinator failed')
        return
    engines=[EXCHANGE/f'engines/rank-{r:02d}.json' for r in range(4)];wait(engines,min(DEADLINE-180,START+1200))
    for rank,path in enumerate(engines):
        engine=json.loads(path.read_text())
        assert engine['completed'] and engine['annotation_rank']==rank and engine['gpu_uuid']==global_devices[rank]['uuid']
        assert engine['source_commit']==os.environ['TASK_CODE_PIN'] and engine['warmup_videos']==8
        assert engine['gpu_memory_utilization']==.75 and engine['batch_size']==8 and engine['model_load_count']==1
        assert engine['memory_snapshot']['free_mib']>=21973
    write_new(RUN/'HEADROOM-ALL-FOUR.json',dict(completed=True,rollout_lanes_per_gpu=4,
        required_free_mib=21973,engines=[dict(path=str(p),sha256=sha(p)) for p in engines]))
    template=json.loads((CODE/'outputs/robocasa-joint-six-update-proposal-20261102-r1/BINDING-TEMPLATE.json').read_text())
    base=copy.deepcopy(template['policy_profile'])
    base.update(source_commit=os.environ['TASK_CODE_PIN'],allocated_start_unix=min(m['allocated_start_unix'] for m in manifests),
        deadline_unix=DEADLINE,scheduler_seconds=6300,closure_seconds=180,maximum_all_in_actions=None,
        rollout_lanes_per_gpu=4,rollout_devices=global_devices,parallel_controller_cpu_offload=True,
        accounting_root=str(RUN/'actions'),actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),
        distributed_rollout_root=str(DISPATCH),gpu_allocation_starts_unix=[m['allocated_start_unix'] for m in manifests for _ in range(1)])
    base['core_qualification'].append(dict(path=str(RUN/'CPU-QUALIFICATION.json'),sha256=sha(RUN/'CPU-QUALIFICATION.json')))
    for index,task in enumerate(PPO_TASKS):
        profile=copy.deepcopy(base)
        for key in ('baseline_validations','calibration_exceptions','tasks'):profile.pop(key,None)
        native=RUN/'native'/task;monitor=RUN/'monitoring'/task
        cases=case_bank(task,'baseline',count=50)
        seed=private_action_seed('baseline',index*10000)
        profile.update(mode='headroom',arm='S',task=task,output=str(native),monitoring_root=str(monitor),
            attempt=os.environ['TASK_RUN']+'-'+task,episode_id_namespace='H600-'+task+'-',
            headroom_cases=cases,headroom_count=50,headroom_offset=0,
            motor_seed=seed,evaluation_motor_seeds=dict(baseline=seed))
        run_controller(profile,native_env,'native-'+task)
        episodes=json.loads((native/'episodes.json').read_text());assert len(episodes)==50
        rows=[]
        for episode,case in zip(episodes,cases,strict=True):
            assert episode['horizon']==episode['frame_clock']['physical_steps']==600 and episode['frame_clock']['frame_count']==601
            assert len(episode['videos'])==3
            truth=json.loads((monitor/'baseline-000'/(episode['id']+'.json')).read_text())
            assert truth['case']==case and truth['native_counts']['controls']-truth['native_counts']['reset_controls']==600
            rows.append(dict(id=episode['id'],case=case,simulator_final_success=truth['simulator_final_success']))
        destination=RUN/'cohorts'/task
        write_new(destination/'episodes.json',episodes);write_new(destination/'CASE-ROWS.json',rows)
        write_new(destination/'NATIVE-QUALIFICATION.json',dict(task=task,count=50,successes=sum(r['simulator_final_success'] for r in rows),
            protocol=protocol(task,baseline_count=50),motor_seed=seed,retained_prefix_count=0,optimizer_updates=0,
            robot_controls=30000,new_robot_controls=30000,
            episodes=dict(path=str(destination/'episodes.json'),sha256=sha(destination/'episodes.json')),
            monitor=dict(path=str(destination/'CASE-ROWS.json'),sha256=sha(destination/'CASE-ROWS.json'))))
        print(json.dumps(dict(phase='native_calibration_bank_complete',task=task,episodes=50)),flush=True)
    wait([EXCHANGE/'ALL-FIVE.json'])
    while not all(c.poll()==0 for c in scorer_children):check();time.sleep(1)
    write_new(OUT/'scorers-closed.json',dict(completed=True,annotation_ranks=[0]))
    wait([JOB/f'node-{rank:02d}/scorers-closed.json' for rank in range(4)])
    gates={task:dict(path=str(EXCHANGE/'cohorts'/task/'baseline-validation.json'),
        sha256=sha(EXCHANGE/'cohorts'/task/'baseline-validation.json')) for task in PPO_TASKS}
    measured={task:read_ref(ref) for task,ref in gates.items()}
    failed=[]
    for task,gate in measured.items():
        try:require_calibration_permission(gate,protocol(task,baseline_count=50),calibration_sha256=gates[task]['sha256'])
        except ValueError as error:failed.append(dict(task=task,reason=str(error),initial_success_rate=gate['initial_success_rate'],AUROC=gate['AUROC']))
    write_new(RUN/'CALIBRATION-GATES.json',dict(completed=True,gates=gates,failed=failed,all_passed=not failed,
        episodes=250,policy_controls=150000,optimizer_updates=0))
    if failed:
        write_new(RUN/'COMPLETED.json',dict(completed=True,calibration_completed=True,common_before_completed=False,
            reason='Fresh measured calibration gate failed; no exception or partial common panel authorized',failed=failed,
            optimizer_updates=0,elapsed_wall_seconds=time.time()-min(m['allocated_start_unix'] for m in manifests)))
    else:
        profile=copy.deepcopy(base)
        profile.update(mode='common_before',arm='S',task='joint_five',tasks=list(PPO_TASKS),baseline_validations=gates,
            calibration_exceptions={},output=str(RUN/'common'),monitoring_root=str(RUN/'common-monitoring'),
            attempt=os.environ['TASK_RUN']+'-common',episode_id_namespace='joint-')
        run_controller(profile,native_env,'common-before')
        common=json.loads((RUN/'common/COMMON-FINAL.json').read_text())
        assert common['completed'] and common['policy_updates']==0 and common['ID_episodes']==common['OOD_episodes']==250
        write_new(RUN/'COMPLETED.json',dict(completed=True,calibration_completed=True,common_before_completed=True,
            calibration_episodes=250,common_before_episodes=540,optimizer_updates=0,
            common_final=dict(path=str(RUN/'common/COMMON-FINAL.json'),sha256=sha(RUN/'common/COMMON-FINAL.json')),
            elapsed_wall_seconds=time.time()-min(m['allocated_start_unix'] for m in manifests)))
    write_new(DISPATCH/'shutdown.json',dict(completed=True))


def halt(signum,frame):raise SystemExit(128+signum)


if __name__=='__main__':
    signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
    try:main()
    except BaseException as error:
        path=JOB/'errors'/f'node-{RANK:02d}.json'
        if not path.exists():write_new(path,dict(error=f'{type(error).__name__}: {error}'))
        if RANK==0 and not (DISPATCH/'shutdown.json').exists():write_new(DISPATCH/'shutdown.json',dict(completed=False,error=str(error)))
        raise
    finally:
        stop(controller)
        for child in reversed(children):stop(child)
        for stream in streams:stream.close()
        write_new(OUT/'node-terminal.json',dict(node_rank=RANK,child_exit_codes=[c.returncode for c in children],
            elapsed_wall_seconds=time.time()-START,optimizer_updates=0))
