"""Resource admission and two sequential official loading/smoke subprocesses."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);out=Path(os.environ['TASK_OUT']);source=Path(os.environ['TASK_WT']);inputs=Path(os.environ['TASK_INPUT']);deadline=int(os.environ['TASK_STARTED'])+3420
assert socket.gethostname().startswith('job-')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args):
 remaining=deadline-time.time()-30
 if remaining<30:raise TimeoutError('Original loading/smoke deadline exhausted')
 subprocess.run(args,check=True,timeout=remaining)
assert subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()==''
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_SOURCE']
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=int(Path('/sys/fs/cgroup/memory.max').read_text())
assert int(cpu[0])/int(cpu[1])==16 and memory==96<<30
import torch,rlinf,fluffy_bassoon
assert torch.cuda.device_count()==1 and torch.__version__=='2.11.0+cu128'
original_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
# Preserve every nonempty scheduler namespace. The exclusive one-card container
# can omit this optional variable; the original NVIDIA UUID still gates probe.
if not original_visibility:os.environ['CUDA_VISIBLE_DEVICES']='0'
assert Path(rlinf.__file__).resolve().is_relative_to(Path(os.environ['TASK_RLINF']))
attestation=dict(hostname=socket.gethostname(),pid=os.getpid(),source_commit=os.environ['TASK_SOURCE'],python=sys.version,torch=torch.__version__,torch_origin=torch.__file__,rlinf_origin=rlinf.__file__,fluffy_origin=fluffy_bassoon.__file__,cpu_max=cpu,memory_bytes=memory,actual_gpu_count=1,original_CUDA_VISIBLE_DEVICES=original_visibility,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],NVIDIA_VISIBLE_DEVICES=os.environ.get('NVIDIA_VISIBLE_DEVICES'),absolute_inner_deadline=deadline)
(out/'runtime-attestation.json').write_text(json.dumps(attestation,indent=2)+'\n')
run([sys.executable,'-B',str(source/'scripts/[train]gpu_probe.py'),'--expected-count','1','--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')])
cpu_ref=inputs/'PPO-READY40-CPU-QUALIFICATION.json';qual=json.loads(cpu_ref.read_text());assert qual['completed'] and qual['tests']==67
artifact_ref=inputs/'READY40-INITIALIZER-ARTIFACT.json';artifact=json.loads(artifact_ref.read_text());assert artifact['completed'] and artifact['revision']=='45ccfcc4e28634f1576ebf78cab0fbe2fd82432d'
profile=dict(status='ACTUAL_ALLOCATED_SMOKE_ADMISSION',source_commit=os.environ['TASK_SOURCE'],phase='smoke',arm='S',world_size=1,allocation_id=socket.gethostname(),allocated_gpu_count=1,scheduler_wall_seconds=3540,allocated_start_unix=int(os.environ['TASK_STARTED']),deadline_unix=deadline,
 actual_gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),initializer_artifact_receipt=dict(path=str(artifact_ref),sha256=sha(artifact_ref)),cpu_qualification=dict(path=str(cpu_ref),sha256=sha(cpu_ref)),
 loading_qualification=None,baseline_validation=None,cost_status_path=str(inputs/'CURRENT-QUALIFICATION-COST-VIEW.json'),common_actions_attribution='one planned native smoke, at most415 including15 settling; common baseline bucket',estimated_iteration_seconds=None,closure_reserve_seconds=180)
(out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n');os.environ['FLORA_PPO_PROFILE_SHA']=sha(out/'profile.json')
run([sys.executable,'-B',str(inputs/'READY40-LOAD-QUALIFICATION.py')])
loading=json.loads((out/'loading-qualification.json').read_text());assert loading['completed'] and loading['backbone_missing_keys']==[] and loading['unexpected_keys']==[]
print(json.dumps(dict(event='single_native_smoke_begin',loading_qualification_sha256=sha(out/'loading-qualification.json'),remaining_seconds=deadline-time.time(),maximum_episodes=1,maximum_actions=415)),flush=True)
run([sys.executable,'-B',str(source/'scripts/diagnostics/rlinf_flora_ppo_entry.py'),'--phase','smoke','--arm','S','--world','1','--profile',str(out/'profile.json'),'--model',os.environ['FLORA_PPO_MODEL'],'--norm',os.environ['FLORA_PPO_NORM'],'--output',os.environ['FLORA_PPO_OUTPUT'],'--exchange',os.environ['FLORA_PPO_EXCHANGE'],'--monitoring',os.environ['FLORA_PPO_MONITORING'],'--accounting',os.environ['FLORA_PPO_ACCOUNTING'],'--attempt',os.environ['TASK_RUN']])
eval_path=Path(os.environ['FLORA_PPO_EXCHANGE'])/'S/smoke/iteration-0000/evaluation.json';evaluation=json.loads(eval_path.read_text());assert evaluation['completed'] and evaluation['episodes']==1 and evaluation['physical_actions']<=415
receipt=dict(completed=True,source_commit=os.environ['TASK_SOURCE'],loading_qualification=dict(path=str(out/'loading-qualification.json'),sha256=sha(out/'loading-qualification.json')),native_smoke=dict(path=str(eval_path),sha256=sha(eval_path)),simulator_actions=evaluation['physical_actions'],episodes=1,optimizer_steps=0,remaining_seconds=deadline-time.time(),same_original_deadline=deadline)
(out/'summary.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
