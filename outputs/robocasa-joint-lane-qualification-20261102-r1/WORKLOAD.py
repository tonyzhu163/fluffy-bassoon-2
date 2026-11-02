"""Finite scheduler qualification of current lane collection beside Flora .78."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank, PPO_TASKS
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import native_chunk_offsets, native_rollout_memory_reserve_mib

root=Path('/mowenjie/fluffy-bassoon-2'); code=Path(os.environ['TASK_META']); out=Path(os.environ['TASK_OUT'])
started=int(os.environ['TASK_STARTED']); deadline=started+1440
assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==1
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split(); assert int(cpu[0])/int(cpu[1])>=32
assert int(Path('/sys/fs/cgroup/memory.max').read_text())>=128<<30
space=os.statvfs(out); assert space.f_bavail*space.f_frsize>=32<<30
(out/'write-probe').write_text('fresh scheduler write probe\n')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','1',
    '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(out/'gpu-probe.json')],check=True,timeout=90)
device=json.loads((out/'gpu-probe.json').read_text())['devices'][0]
assert device['inventory']['memory_total_mib']>=90000
write_new(out/'allocation.json',dict(hostname=socket.gethostname(),source_commit=os.environ['TASK_CODE_PIN'],
    requested_gpus=1,CPU=32,RAM_GiB=128,SHM_GiB=16,deadline_unix=deadline,quota='UNVERIFIED',free_bytes=space.f_bavail*space.f_frsize))
base=dict(os.environ, CUDA_VISIBLE_DEVICES=device['visibility_token'], LYCO_BAIGE_GPU_MAP=str(out/'gpu-probe.json'),
    LYCO_BAIGE_ISOLATED_UUID=device['uuid'],MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']))
inputs=json.loads((code/'outputs/robocasa-flora-throughput-20261102-r2/INPUTS.json').read_text())
assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
flora_python=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
toolkit=Path(inputs['toolkit'])
scorer_env=dict(base, PYTHONPATH=str(code/'src')+':'+inputs['official_code'], TASK_FLORA_MODEL=inputs['model'],
    CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit), CUDACXX=str(toolkit/'bin/nvcc'),
    PATH=str(toolkit/'bin')+':'+str(flora_python.parent)+':'+os.environ['PATH'],
    LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
    LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),
    XDG_CACHE_HOME=str(out/'flora-cache'), TORCHINDUCTOR_CACHE_DIR=str(out/'flora-cache/torch'),
    FLASHINFER_WORKSPACE_BASE=str(out/'flora-cache/flashinfer'), VLLM_CACHE_ROOT=str(out/'flora-cache/vllm'),
    TRITON_CACHE_DIR=str(out/'flora-cache/triton'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MAX_JOBS='4')
children=[]; streams=[]; stop=threading.Event(); samples=[]; stage='engine_loading'
def memory():
    with (out/'memory.jsonl').open('x') as stream:
        while not stop.is_set():
            row=dict(unix=time.time(),stage=stage)
            try:
                raw=subprocess.check_output(['nvidia-smi','--id='+device['uuid'],
                    '--query-gpu=memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
                row.update(zip(('used_mib','free_mib','total_mib','utilization_percent'),map(float,raw.strip().split(','))))
            except Exception as error:row['error']=str(error)
            samples.append(row);stream.write(json.dumps(row)+'\n');stream.flush();stop.wait(1)
thread=threading.Thread(target=memory,daemon=True); thread.start()
def start(python,script,args,env,name):
    stream=(out/(name+'.log')).open('x');streams.append(stream)
    child=subprocess.Popen([str(python),'-B',str(script),*args],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(child);return child
def wait(path,child):
    while not path.exists():
        if child.poll() is not None:raise RuntimeError(f'{path.name} absent; child exit {child.returncode}')
        if time.time()>deadline:raise TimeoutError('Finite qualification deadline')
        time.sleep(1)
result=dict(completed=False,optimizer_updates=0)
try:
    scorer=start(flora_python,code/'outputs/robocasa-joint-lane-qualification-20261102-r1/SCORER.py',[],scorer_env,'scorer')
    wait(out/'engine-ready.json',scorer)
    ready=json.loads((out/'engine-ready.json').read_text());free=ready['memory_mib'][1]
    candidates=[dict(lanes=n,required_free_mib=native_rollout_memory_reserve_mib(dict(rollout_lanes_per_gpu=n)),
        measured_free_mib=free,admitted=free>=native_rollout_memory_reserve_mib(dict(rollout_lanes_per_gpu=n))) for n in (8,4)]
    write_new(out/'headroom.json',candidates)
    selected=8 if free>=19216+512 else 4 if free>=13781+512 else None
    if selected is None:
        result.update(completed=True,selected_lanes=None,reason='Neither8nor4 fits measured standalone peak plus512MiB measurement margin',headroom=candidates)
    else:
        while True:
            stage=f'native_{selected}_lanes_with_streaming_Flora'
            write_new(out/f'selection-{selected}.json',dict(lanes=selected,model_receipt=inputs['model_receipt']))
            pipeline=json.loads((root/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
            checkpoint=torch.load(root/'outputs/robocasa-residual-approved-20261101-r1/S/boundary-000.pt',map_location='cpu',weights_only=False)
            cases=[]
            for repeat in range(2):
                for task in PPO_TASKS:
                    case=case_bank(task,'baseline',count=8)[repeat]
                    case.update(bank='lane_qualification',native_seed=6700100+len(cases),slot=len(cases)); cases.append(case)
            config=checkpoint['contract']['ppo'];config.update(epochs=10,minibatch_size=32)
            count=native_chunk_offsets(cases)[-1]
            profile=dict(source_commit=os.environ['TASK_CODE_PIN'],rlinf_execution_commit='8a1233c3fae6f5905adfda7684c455842b0ecfb8',
                initializer=pipeline['training']['initializer'],motor_seed=710100,maximum_all_in_actions=None,
                rollout_lanes_per_gpu=selected,accounting_root=str(out/'actions'))
            directory=out/f'lanes-{selected}/train/videos/parallel-000000/rank-00';directory.mkdir(parents=True)
            request=dict(profile=profile,config=config,policy=checkpoint['policy'],bound=checkpoint['contract']['bound'],
                cases=cases,horizon=None,rank=0,world=8,attempt='qualification',output=str(directory),
                monitoring_root=str(out/'monitoring'),episode_id_prefix=f'lane-qual-{selected}-',deterministic=True,
                verify_zero=True,seeds=list(range(6800000,6800000+count)),noise=None)
            torch.save(request,out/'request.pt')
            env=dict(base,**{k:v for k,v in pipeline['policy_env'].items() if k not in base})
            env.update(PYTHONPATH=str(code/'src')+':'+str(root/'sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf'),
                ROBOCASA_NATIVE_OUTPUT=str(out/'native'))
            tick=time.monotonic()
            native=start(sys.executable,code/'scripts/diagnostics/robocasa_residual_rollout.py',['--request',str(out/'request.pt')],env,f'native-{selected}')
            while native.poll() is None:
                if scorer.poll() is not None:raise RuntimeError('Persistent scorer exited')
                if time.time()>deadline:raise TimeoutError('Finite native collection deadline')
                time.sleep(1)
            if native.returncode:
                failure=(out/f'native-{selected}.log').read_text()
                write_new(out/f'failed-lanes-{selected}.json',dict(exit_code=native.returncode,OOM='out of memory' in failure.lower()))
                if selected==8 and 'out of memory' in failure.lower():
                    selected=4
                    continue
                raise RuntimeError(f'Native lanes{selected} failed; see retained log')
            assert (directory/'result.pt').exists()
            elapsed=time.monotonic()-tick
            pending=torch.load(directory/'result.pt',map_location='cpu',weights_only=False)
            write_new(out/f'completed-bank-{selected}.json',dict(episodes=pending['episodes'],cases=cases))
            wait(out/f'scoring-result-{selected}.json',scorer)
            controls=sum(e['frame_clock']['physical_steps'] for e in pending['episodes'])
            result.update(completed=True,selected_lanes=selected,episodes=len(pending['episodes']),policy_controls=controls,
                inclusive_native_load_collect_seconds=elapsed,inclusive_controls_per_second=controls/elapsed,
                combined_collect_score_seconds=time.monotonic()-tick,scoring=json.loads((out/f'scoring-result-{selected}.json').read_text()),
                headroom=candidates,records=len(pending['records']),result_sha256=sha(directory/'result.pt'))
            break
except BaseException as error:
    result['error']=f'{type(error).__name__}: {error}'
    raise
finally:
    write_new(out/'shutdown.json',dict(reason='qualification closed'))
    for child in children:
        if child.poll() is None:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    stop.set();thread.join(timeout=6)
    for stream in streams:stream.close()
    result.update(child_exit_codes=[c.returncode for c in children],sampled_peak_used_mib=max((r.get('used_mib',0) for r in samples),default=None),
        minimum_free_mib=min((r['free_mib'] for r in samples if 'free_mib' in r),default=None),
        source_commit=os.environ['TASK_CODE_PIN'],total_wall_seconds=time.time()-started)
    write_new(out/'RESULT.json',result)
