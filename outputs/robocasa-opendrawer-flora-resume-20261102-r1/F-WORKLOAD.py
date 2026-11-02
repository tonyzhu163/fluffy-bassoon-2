"""Resume update4 of the original OpenDrawer F entry: existing controller, eight native workers and persistent scorer services.

Human requested completing the original matched Flora arm before scaling.
Restore the preserved policy/critic/Adam/sampler/RNG from update4; no fresh initializer. The first retained
64-episode update verifies combined inference, annotation and PPO.
"""
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
import numpy as np
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol, require_calibration_permission

root,code,out=map(Path,(os.environ['TASK_ROOT'],os.environ['TASK_META'],os.environ['TASK_OUT']))
run=root/'outputs'/os.environ['TASK_RUN'];job=run/'job';dispatch=job/'dispatch'
spec=json.loads((code/'outputs/robocasa-joint-six-update-proposal-20261102-r1/F-HANDOFF.json').read_text())
binding_ref=dict(path=os.environ['TASK_BINDINGS'],sha256=os.environ['TASK_BINDINGS_SHA256'])
binding=read_ref(binding_ref)
rank,start=int(os.environ['RANK']),int(os.environ['TASK_STARTED'])
deadline=start+14340-300
services=[];streams=[];controller=None


def halt(signum,frame):
    raise SystemExit(128+signum)


def fail(path,error):
    if not path.exists():write_new(path,dict(completed=False,error=str(error),node_rank=rank))


def wait_for(paths,limit):
    while True:
        if list((job/'errors').glob('*.json')):raise RuntimeError('An owned F node failed; retain all recoveries')
        if time.time()>=limit:raise TimeoutError('Finite F startup or shutdown boundary')
        if all(p.exists() for p in paths):return
        time.sleep(1)


def disk_usage_bytes(path):
    # Atomic video/metadata publication can rename a .tmp during du traversal.
    # Retry that race; other errors and an exhausted retry still fail closed.
    for attempt in range(5):
        result=subprocess.run(['du','-sk',str(path)],text=True,capture_output=True,
            timeout=20,env=dict(os.environ,LC_ALL='C'))
        if result.returncode==0:
            fields=result.stdout.split()
            if not fields or not fields[0].isdigit():raise RuntimeError('Malformed disk-usage result')
            return int(fields[0])*1024
        lines=result.stderr.strip().splitlines()
        transient=bool(lines) and all('cannot access' in line and 'No such file or directory' in line for line in lines)
        if not transient:raise RuntimeError('Disk-usage check failed: '+result.stderr)
        if attempt==4:raise RuntimeError('Disk-usage rename race persisted after five retries: '+result.stderr)
        time.sleep(.1*(attempt+1))


def export_full_history(run, binding):
    from fluffy_bassoon.meta_rl.residual_ppo import history_trial
    from fluffy_bassoon.meta_rl.robocasa_residual import install_terminal_labels
    from fluffy_bassoon.meta_rl.rlinf_flora_ppo import parse_outputs
    from fluffy_bassoon.meta_rl.vla_history import load_dataset,save_dataset
    prefix=[]
    for item in binding['history_prefix']:
        pending=read_ref(item['request'])
        assert sha(pending['pending_checkpoint']['path'])==pending['pending_checkpoint']['sha256']
        saved=torch.load(pending['pending_checkpoint']['path'],map_location='cpu',weights_only=False)
        ready=read_ref(item['ready'])
        assert ready['pending_checkpoint_sha256']==pending['pending_checkpoint']['sha256']
        labels=parse_outputs(saved['pending_episodes'],read_ref(ready['outputs']),pending['protocol'])
        prefix.append((saved['iteration'],install_terminal_labels(saved['pending_records'],saved['pending_episodes'],labels,arm='F')))
    full=history_trial(prefix)
    suffix=run/'policy/history.npz'
    if suffix.exists():
        trials,manifest=load_dataset(suffix)
        assert len(trials)==1 and trials[0]['teacher_steps'][0]==4
        full={key:np.concatenate([full[key],trials[0][key]],axis=0) for key in full}
    return save_dataset(run/'full-history.npz',[full],objective='ad',metadata=dict(
        evaluation_data=False,logical_lineage='original OpenDrawer Flora PPO',
        prefix=binding['history_prefix'],recovery=binding['recovery'],
        final_checkpoint_sha256=sha(run/'policy/final.pt'),failed_actions_charged=binding['prior_action_charge']))


def main():
    global controller,deadline
    assert binding['status']=='ADMITTED' and binding['stage']=='F'
    assert binding['source_commit']==os.environ['TASK_CODE_PIN']
    assert binding['tasks']==['OpenDrawer'] and binding['maximum_iterations']==6
    authority=read_ref(binding['human_action_authority'])
    assert authority['allow_opendrawer_flora_first'] is True and authority['direct_human_instruction']
    assert authority['maximum_all_in_actions']==500000 and authority['old_trial_spend_preserved'] is True
    assert binding['remaining_all_in_actions']==500000-binding['prior_action_charge']
    assert read_ref(binding['recovery'])['resume_iteration']==4
    profile=copy.deepcopy(binding['policy_profile'])
    assert profile['task']=='OpenDrawer' and profile['maximum_iterations']==6
    assert profile['baseline_validation']==spec['baseline_validations']['OpenDrawer']
    assert profile['calibration_baseline_count']==profile['heldout_evaluation_count']==50
    require_calibration_permission(read_ref(profile['baseline_validation']),protocol('OpenDrawer',baseline_count=50),
        calibration_sha256=profile['baseline_validation']['sha256'])
    assert profile['initializer']==spec['initializer']
    assert profile['rlinf_execution_commit']==spec['rlinf_execution_commit']
    assert profile['loading_qualification']['sha256']==spec['strict_loading_saved_byte_sha256']
    read_ref(profile['loading_qualification'])
    for ref in profile['core_qualification']:read_ref(ref)
    assert profile['resume']==binding['resume_checkpoint']['path']
    assert profile['resume_sha256']==binding['resume_checkpoint']['sha256']
    assert read_ref(profile['rollout_migration'])['kind']=='eight_gpu_rollout_only'
    read_ref(profile['initial_dev_evaluation'])
    assert sha(binding['resume_checkpoint']['path'])==binding['resume_checkpoint']['sha256']
    assert os.environ['WORLD_SIZE']=='4' and os.environ['NPROC_PER_NODE']=='2' and 0<=rank<4
    assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==2
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert int(cpu[0])/int(cpu[1])==32 and int(Path('/sys/fs/cgroup/memory.max').read_text())==256<<30
    head=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    assert head(code)==binding['source_commit'] and head(os.environ['TASK_RLINF'])==spec['rlinf_execution_commit']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
    storage=read_ref(binding['storage_proof'])
    assert storage['completed'] is True and storage['write_probe_pass'] is True and storage['old_recoveries_preserved'] is True
    assert storage['stage_supported_upper_bytes']<=256<<30
    free=os.statvfs(out).f_bavail*os.statvfs(out).f_frsize
    assert free>=storage['remaining_output_reservation_bytes']>=binding['remaining_output_bytes']
    write_probe=out/'write-probe'
    with write_probe.open('x') as stream:
        stream.write('fresh F production namespace\n');stream.flush();os.fsync(stream.fileno())
    write_probe.unlink()
    flora=spec['flora'];model_receipt=read_ref(flora['model_receipt'])
    assert head(flora['official_code'])==flora['official_code_commit']
    assert model_receipt['completed'] is True and model_receipt['model_revision']==flora['model_revision']
    assert Path(model_receipt['model_directory']).resolve()==Path(flora['model']).resolve()
    probe=out/'gpu-probe.json'
    subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count','2',
        '--purpose','training','--baige-container','--baige-exclusive-training','--output',str(probe)],
        check=True,timeout=min(120,max(1,deadline-time.time())))
    devices=json.loads(probe.read_text())['devices']
    assert len(devices)==2 and len({d['uuid'] for d in devices})==2
    assert all(d['inventory']['memory_total_mib']>=90000 for d in devices)
    dispatch.mkdir(exist_ok=True)
    write_new(out/'manifest.json',dict(node_rank=rank,hostname=socket.gethostname(),source_commit=binding['source_commit'],
        allocated_start_unix=start,probe_path=str(probe),devices=devices,cpus=32,ram_gib=256,free_bytes=free,
        quota=storage.get('quota','UNVERIFIED'),write_probe_pass=True,binding=binding_ref,
        human_action_authority=binding['human_action_authority'],resume_checkpoint=binding['resume_checkpoint']))
    paths=[job/f'node-{r:02d}/manifest.json' for r in range(4)]
    wait_for(paths,min(start+300,deadline))
    manifests=[json.loads(p.read_text()) for p in paths]
    assert [m['node_rank'] for m in manifests]==list(range(4))
    assert all(m['source_commit']==binding['source_commit'] and m['binding']==binding_ref for m in manifests)
    global_devices=[dict(d,node_rank=r,probe_path=m['probe_path']) for r,m in enumerate(manifests) for d in m['devices']]
    assert len(global_devices)==len({d['uuid'] for d in global_devices})==8
    deadline=min(m['allocated_start_unix'] for m in manifests)+14340-300
    qualification=read_ref(binding['affected_core_qualification'])
    assert qualification['completed'] is True and qualification['failures']==qualification['errors']==0
    for name,checksum in qualification['files'].items():assert sha(code/name)==checksum
    native_env=dict(PYTHONPATH=str(code/'src')+':'+os.environ['TASK_RLINF'],RLINF_PATH=os.environ['TASK_RLINF'],
        ROBOCASA_ASSETS_PATH=str(root/'setups/robocasa365-native-20261101-r1/assets'),
        OPENPI_DATA_HOME=str(root/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',ROBOCASA_NATIVE_OUTPUT=str(out),
        ROBOCASA_PI05_TORCH_MODEL=profile['initializer']['model_directory'],
        ROBOCASA_PI05_NATIVE_NORM=profile['initializer']['norm_path'])
    node_profile=dict(node_rank=rank,devices=devices,dispatch_root=str(dispatch),native_env=native_env,
        probe_path=str(probe),source_commit=binding['source_commit'],deadline_unix=deadline)
    write_new(out/'node-profile.json',node_profile)
    service=subprocess.Popen([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_node.py'),
        '--profile',str(out/'node-profile.json')],start_new_session=True)
    services.append(service)
    errors=[str(job/'errors'/f'node-{r:02d}.json') for r in range(4)]
    errors.extend(str(job/'errors'/f'scorer-{r:02d}.json') for r in range(8))
    scorer_refs=[]
    for local,device in enumerate(devices):
        annotation_rank=rank*2+local
        scorer=dict(mode='train',task='OpenDrawer',baseline_count=50,annotation_world=8,annotation_rank=annotation_rank,
            source_commit=binding['source_commit'],F_output=str(run/'policy'),output=str(out/f'scorer-{annotation_rank:02d}'),
            model=flora['model'],official_code=flora['official_code'],model_receipt=flora['model_receipt'],
            baseline_validation=profile['baseline_validation'],
            batch_size=8,gpu_memory_utilization=.78,minimum_free_mib_after_engine_warmup=17536,
            engine_ready_path=str(job/'engines'/f'rank-{annotation_rank:02d}.json'),
            actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),expected_gpu_uuid=device['uuid'],
            scheduler_seconds=14340,deadline_unix=deadline,closure_seconds=300,maximum_batches=2,
            shutdown_file=str(job/'annotation-shutdown.json'))
        p=out/f'scorer-{annotation_rank:02d}-profile.json';write_new(p,scorer)
        scorer_refs.append((annotation_rank,p))
        toolkit=Path(flora['toolkit']);python=Path(spec['scorer_python']);cache=out/f'scorer-{annotation_rank:02d}-cache'
        temporary=Path(f'/tmp/fb-j6F-{annotation_rank}');temporary.mkdir(exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['visibility_token'],LYCO_BAIGE_GPU_MAP=str(probe),
            LYCO_BAIGE_ISOLATED_UUID=device['uuid'],PYTHONPATH=str(code/'src')+':'+flora['official_code'],
            CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),CUDACXX=str(toolkit/'bin/nvcc'),
            PATH=str(toolkit/'bin')+':'+str(python.parent)+':'+os.environ['PATH'],
            LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
            LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),TMPDIR=str(temporary),
            XDG_CACHE_HOME=str(cache),TORCHINDUCTOR_CACHE_DIR=str(cache/'torch'),FLASHINFER_WORKSPACE_BASE=str(cache/'flashinfer'),
            VLLM_CACHE_ROOT=str(cache/'vllm'),TRITON_CACHE_DIR=str(cache/'triton'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MAX_JOBS='4')
        log=(out/f'scorer-{annotation_rank:02d}-console.log').open('x');streams.append(log)
        services.append(subprocess.Popen([str(python),'-B',str(code/'scripts/diagnostics/robocasa_residual_score.py'),
            '--profile',str(p)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    if rank==0:
        scorer_paths=[job/f'node-{r//2:02d}/scorer-{r:02d}-profile.json' for r in range(8)]
        wait_for(scorer_paths,min(start+300,deadline))
        profile.update(mode='train',arm='F',source_commit=binding['source_commit'],output=str(run/'policy'),
            monitoring_root=str(run/'monitoring'),accounting_root=str(run/'actions'),attempt=os.environ['TASK_RUN'],
            episode_id_namespace='flora-single-',allocated_start_unix=min(m['allocated_start_unix'] for m in manifests),
            deadline_unix=deadline,scheduler_seconds=14340,closure_seconds=300,maximum_all_in_actions=binding['remaining_all_in_actions'],
            rollout_devices=global_devices,parallel_controller_cpu_offload=True,independent_stopping=True,
            actual_gpu_probe=dict(path=str(probe),sha256=sha(probe)),distributed_rollout_root=str(dispatch),
            gpu_allocation_starts_unix=[m['allocated_start_unix'] for m in manifests for _ in range(2)],
            initial_dev_evaluation=profile['initial_dev_evaluation'],
            training_cases=profile['training_cases'],
            first_iteration_wall_ceiling_seconds=3600,dev_evaluation_wall_ceiling_seconds=900,
            final_evaluation_wall_reserve_seconds=7200,annotation_startup_wall_ceiling_seconds=1800,
            annotation_error_paths=errors,annotation_engines=[dict(profile=dict(path=str(p),sha256=sha(p)),
                ready_path=str(job/'engines'/f'rank-{r:02d}.json')) for r,p in enumerate(scorer_paths)])
        p=out/'policy-profile.json';write_new(p,profile)
        environment=dict(os.environ,**native_env,CUDA_VISIBLE_DEVICES=global_devices[0]['visibility_token'],
            LYCO_BAIGE_GPU_MAP=str(probe),LYCO_BAIGE_ISOLATED_UUID=global_devices[0]['uuid'],MUJOCO_EGL_DEVICE_ID='0')
        controller=subprocess.Popen([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_ppo.py'),
            '--profile',str(p)],env=environment,start_new_session=True)
    last_memory,last_size=0.,0.
    completed_scorers=set(); initial_verified=False
    while True:
        if rank==0 and not initial_verified and (run/'policy/RESTORED.json').exists():
            restored=json.loads((run/'policy/RESTORED.json').read_text())
            assert restored['completed'] is True and restored['iteration']==4 and restored['stage']=='boundary'
            assert restored['checkpoint']==binding['resume_checkpoint']
            write_new(run/'RESUME-VERIFIED.json',dict(completed=True,restore=restored,recovery=binding['recovery']))
            initial_verified=True
        if rank!=0 and (dispatch/'shutdown.json').exists():
            assert json.loads((dispatch/'shutdown.json').read_text()).get('completed') is True
            break
        if list((job/'errors').glob('*.json')):raise RuntimeError('Owned F service error; preserve pending/recovery/action escrows')
        if service.poll() is not None:raise RuntimeError('Native F node service exited before coordinated shutdown')
        for (r,_),child in zip(scorer_refs,services[1:]):
            if child.poll() not in (None,0):
                fail(job/'errors'/f'scorer-{r:02d}.json',f'Scorer exit{child.returncode}')
                raise RuntimeError('Official scorer failed; no partial labels or fallback')
            if child.poll()==0 and r not in completed_scorers:
                # Non-coordinator scorers finish after publishing their own
                # sixth shard. Rank0 may still be assembling the global ready;
                # that independent barrier must not turn a normal exit into a
                # false failure. Verify this worker's own completed contribution.
                own=[json.loads(path.read_text()) for path in
                     (run/'policy').glob(f'labels-*/shards/rank-{r:02d}.json')]
                count=sum(row.get('completed') is True and row.get('annotation_rank')==r
                          and row.get('code_commit')==flora['official_code_commit'] for row in own)
                if not (run/'policy/FINAL.json').exists() and count!=2:
                    raise RuntimeError('Scorer ended before its own two remaining complete shards or controller final boundary')
                completed_scorers.add(r)
        now=time.time()
        if now>=deadline:raise TimeoutError('Finite F closure; no new bank or process extension')
        if now-last_memory>=10:
            row=subprocess.check_output(['nvidia-smi','--id='+','.join(d['uuid'] for d in devices),
                '--query-gpu=uuid,memory.used,memory.free,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=5)
            with (out/'memory.jsonl').open('a') as log:log.write(json.dumps(dict(unix=now,nvml_rows=row.strip().splitlines(),scope='sampledobservednotlifetimepeak'))+'\n')
            last_memory=now
        if rank==0:
            if now-last_size>=60:
                size=disk_usage_bytes(run)
                assert size<=binding['remaining_output_bytes'],'Owned F112GiB output limit reached; no deletion/expansion'
                last_size=now
            if controller.poll() is not None:
                if controller.returncode:raise RuntimeError(f'F controller exited{controller.returncode}; retain saved full/pending state')
                final=read_ref(dict(path=str(run/'policy/FINAL.json'),sha256=sha(run/'policy/FINAL.json')))
                assert initial_verified
                panels={phase:json.loads((run/f'policy/{phase}-{final["iteration"]:03d}/evaluation.json').read_text()) for phase in ('ID','OOD')}
                assert final['completed'] is True and 4<=final['iteration']<=6
                checkpoint=dict(path=final['checkpoint'],sha256=final['sha256'])
                assert sha(checkpoint['path'])==checkpoint['sha256']
                assert all(row['episodes']==50 for row in panels.values())
                write_new(run/'policy/EVALUATION-COMPLETE.json',dict(completed=True,checkpoint=checkpoint,
                    ID_episodes=50,OOD_episodes=50,checkpoint_selection=False,panels=panels))
                size=disk_usage_bytes(run)
                assert size<=binding['remaining_output_bytes'],'Final F output band exceeded; preserve all artifacts without expansion'
                history=export_full_history(run,binding)
                write_new(run/'COMPLETED.json',dict(completed=True,stage='F',final_checkpoint=checkpoint,
                    actual_output_allocated_bytes=size,full_history=history,
                    final_marker=dict(path=str(run/'policy/FINAL.json'),sha256=sha(run/'policy/FINAL.json')),
                    evaluation_marker=dict(path=str(run/'policy/EVALUATION-COMPLETE.json'),sha256=sha(run/'policy/EVALUATION-COMPLETE.json')),
                    human_action_authority=binding['human_action_authority'],resume_checkpoint=binding['resume_checkpoint']))
                write_new(job/'annotation-shutdown.json',dict(completed=True))
                write_new(dispatch/'shutdown.json',dict(completed=True))
                break
        time.sleep(1)


signal.signal(signal.SIGTERM,halt)
signal.signal(signal.SIGINT,halt)
try:main()
except BaseException as error:
    fail(job/'errors'/f'node-{rank:02d}.json',f'{type(error).__name__}: {error}')
    if rank==0:
        fail(job/'annotation-shutdown.json',str(error));fail(dispatch/'shutdown.json',str(error))
    raise
finally:
    for child in ([controller] if controller else [])+services:
        if child.poll() is None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
    for child in ([controller] if controller else [])+services:
        try:child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait()
    for log in streams:log.close()
