"""Actual allocated native headroom, no learner update or scorer in this job."""
import hashlib,json,os,pathlib,socket,subprocess,sys,time
import torch
r=pathlib.Path(os.environ['TASK_ROOT']);out=pathlib.Path(os.environ['TASK_OUT']);inputs=pathlib.Path(os.environ['TASK_INPUT']);code=pathlib.Path(os.environ['TASK_META']);deadline=int(os.environ['TASK_STARTED'])+6420
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def run(args):
 remaining=deadline-time.time()-180
 if remaining<30:raise TimeoutError('Finite native qualification deadline reached')
 subprocess.run(list(map(str,args)),check=True,timeout=remaining)
assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==1
original_visibility=os.environ.get('CUDA_VISIBLE_DEVICES')
if not original_visibility:os.environ['CUDA_VISIBLE_DEVICES']='0'
cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text());assert int(cpu[0])/int(cpu[1])==32 and memory==128<<30
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
assert subprocess.check_output(['git','-C',os.environ['TASK_RLINF'],'rev-parse','HEAD'],text=True).strip()=='8a1233c3fae6f5905adfda7684c455842b0ecfb8'
stat=os.statvfs(out);assert stat.f_bavail*stat.f_frsize>=32<<30
(out/'allocated-resource.json').write_text(json.dumps(dict(hostname=socket.gethostname(),pid=os.getpid(),actual_gpu_count=1,cpus=32,memory_bytes=memory,free_bytes=stat.f_bavail*stat.f_frsize,quota='UNVERIFIED',deadline_unix=deadline),indent=2))
run([sys.executable,'-B',code/'scripts/[train]gpu_probe.py','--expected-count','1','--purpose','evaluation','--baige-container','--output',out/'gpu-probe.json'])
summary=json.loads((inputs/'POLICY-SUMMARY.json').read_text());loading=inputs/'STRICT-LOADING.json'
assert summary['completed'] and summary['model_sha256']=='661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
model=pathlib.Path(os.environ['ROBOCASA_PI05_TORCH_MODEL']);norm=pathlib.Path(os.environ['ROBOCASA_PI05_NATIVE_NORM'])
assert sha(norm)==summary['normalization_sha256']
cases=json.loads((inputs/'CASES.json').read_text());selected=None
for task in ('OpenDrawer','ScrubCuttingBoard'):
 profile=dict(source_commit=os.environ['TASK_CODE_PIN'],rlinf_execution_commit='8a1233c3fae6f5905adfda7684c455842b0ecfb8',mode='headroom',arm='S',task=task,scheduler_seconds=6720,closure_seconds=180,deadline_unix=deadline,allocated_start_unix=int(os.environ['TASK_STARTED']),actual_gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),loading_qualification=dict(path=str(loading),sha256=sha(loading)),core_qualification=[],initializer=dict(repository='robocasa/robocasa365_checkpoints',revision='c484448aba1a9b60a04c9b0ca117241518ea69f3',prefix='pi05_pretrain_human300/multitask_learning/75000',model_directory=str(model),model_sha256=summary['model_sha256'],normalization_sha256=summary['normalization_sha256'],norm_path=str(norm)),output=str(out/task),monitoring_root=str(out/'monitoring'/task),accounting_root=str(r/'outputs/robocasa-native-residual-headroom-20261101-r1/actions'),attempt=os.environ['TASK_RUN']+'-'+task,maximum_all_in_actions=17648,motor_seed=610100,residual_seed=42,update_seed=43,evaluation_motor_seeds=dict(baseline=610100),headroom_cases=cases[task],ppo=dict(feature_dim=2088,residual_dim=6,width=256,layers=2,initial_std=.5,learning_rate=3e-4,epochs=4,minibatch_size=64,clip=.2,gamma=1.,gae_lambda=.95,entropy_coef=0.,value_coef=.5,max_grad_norm=1.))
 p=out/(task+'-profile.json');p.write_text(json.dumps(profile,indent=2)+'\n')
 run([sys.executable,'-B',code/'scripts/diagnostics/robocasa_residual_ppo.py','--profile',p])
 result=json.loads((out/task/'headroom.json').read_text())
 if result['native_headroom']:
  selected=dict(task=task,result=result,episodes=dict(path=str(out/task/'episodes.json'),sha256=sha(out/task/'episodes.json')));break
terminal=dict(completed=True,selected=selected,Flora_AUROC='NOT_MEASURED',scientific_optimizer_updates=0,selection='prospective native-only first passing candidate; no Flora score used')
(out/'summary.json').write_text(json.dumps(terminal,indent=2)+'\n');print(json.dumps(terminal),flush=True)
