"""Allocated pipeline binding; no model/array import on the development host."""
import json, os, pathlib, socket, subprocess, sys, time
import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank

r=pathlib.Path(os.environ['TASK_ROOT']);out=pathlib.Path(os.environ['TASK_OUT'])
inputs=pathlib.Path(os.environ['TASK_INPUT']);code=pathlib.Path(os.environ['TASK_META'])
role=os.environ['TASK_ROLE'];gpus=2 if role=='F' else 1
start=int(os.environ['TASK_STARTED']);deadline=start+(27300 if role=='F' else 27540)
assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==gpus
if not os.environ.get('CUDA_VISIBLE_DEVICES'):os.environ['CUDA_VISIBLE_DEVICES']=','.join(map(str,range(gpus)))
cpu=pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split();mem=int(pathlib.Path('/sys/fs/cgroup/memory.max').read_text())
assert int(cpu[0])/int(cpu[1])==(64 if role=='F' else 32) and mem==(256 if role=='F' else 128)<<30
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()
free=os.statvfs(out);assert free.f_bavail*free.f_frsize>=256<<30
write_new(out/'resource.json',dict(hostname=socket.gethostname(),gpu_count=gpus,cpus=int(cpu[0])/int(cpu[1]),memory_bytes=mem,free_bytes=free.f_bavail*free.f_frsize,quota='UNVERIFIED',deadline_unix=deadline))
subprocess.run([sys.executable,'-B',str(code/'scripts/[train]gpu_probe.py'),'--expected-count',str(gpus),'--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')],check=True,timeout=120)
mapping=json.loads((out/'gpu-probe.json').read_text())
selected=mapping['devices']
assert len(selected)==gpus
root=r/'outputs/robocasa-residual-integrated-20261101-r3';exchange=root/'exchange'
native=r/'outputs/robocasa-native-residual-headroom-q4-20261101-r1'
episodes=native/'OpenDrawer/episodes.json'
assert sha(episodes)=='f912ddb229bd836f53ca37b78815bae3c8fb1b457a6dbd39db318277ee6a2999'
assert json.loads((native/'summary.json').read_text())['selected']['result']['successes']==3
assert (native/'exit-code.txt').read_text().strip()=='0'
summary=json.loads((inputs/'POLICY-SUMMARY.json').read_text());loading=inputs/'STRICT-LOADING.json'
probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json'))
initializer=dict(repository='robocasa/robocasa365_checkpoints',revision='c484448aba1a9b60a04c9b0ca117241518ea69f3',prefix='pi05_pretrain_human300/multitask_learning/75000',model_directory=str(r/'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'),model_sha256=summary['model_sha256'],normalization_sha256=summary['normalization_sha256'],norm_path=str(r/'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'))
assert initializer['model_sha256']=='661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
base=dict(source_commit=os.environ['TASK_CODE_PIN'],rlinf_execution_commit='8a1233c3fae6f5905adfda7684c455842b0ecfb8',task='OpenDrawer',scheduler_seconds=(27660 if role=='F' else 27900),closure_seconds=180,deadline_unix=deadline,allocated_start_unix=start,actual_gpu_probe=probe,loading_qualification=dict(path=str(loading),sha256=sha(loading)),core_qualification=[],initializer=initializer,motor_seed=710100,residual_seed=42,update_seed=43,evaluation_motor_seeds=dict(initial_DEV=810100,DEV=810100,before_ID=910100,ID=910100,before_OOD=1010100,OOD=1010100),ppo=dict(feature_dim=2088,residual_dim=6,width=256,layers=2,initial_std=.5,learning_rate=3e-4,epochs=4,minibatch_size=64,clip=.2,gamma=1.,gae_lambda=.95,entropy_coef=0.,value_coef=.5,max_grad_norm=1.),first_iteration_wall_ceiling_seconds=7200,dev_evaluation_wall_ceiling_seconds=1200,final_evaluation_wall_reserve_seconds=2400)
common=dict(base,mode='common_before',arm='S',output=str(root/'common-before'),monitoring_root=str(root/'monitoring/common-before'),accounting_root=str(root/'common-actions'),attempt='common-before',maximum_all_in_actions=22272)
training=dict(base,mode='train',arm=role,output=str(root/role),monitoring_root=str(root/'monitoring'/role),accounting_root=str(root/(role+'-actions')),attempt=role+'-train',maximum_all_in_actions=971600,paired_stop_root=str(root/'paired-stop'),training_cases=[case for iteration in range(1,11) for round_index in range(4) for case in case_bank('OpenDrawer','train',iteration=iteration,round_index=round_index)])
receipt=r/'inputs/flora-rewarder-031299e4-20261101-r1-receipt/READY.json'
scorer=dict(mode='integrated',task='OpenDrawer',scheduler_seconds=(27660 if role=='F' else 27900),deadline_unix=deadline,closure_seconds=180,actual_gpu_probe=probe,model=str(r/'inputs/flora-rewarder-031299e4-20261101-r1'),model_receipt=dict(path=str(receipt),sha256=sha(receipt)),official_code=str(r/'deps/nvidia-flora-129a2ada-20261101-r1/repo'),output=str(root/'scorer'),F_output=str(root/'F'),frozen_native_episodes_sha256=sha(episodes),maximum_batches=10)
native_env=dict(CUDA_VISIBLE_DEVICES=selected[0]['visibility_token'],PYTHONPATH=str(code/'src')+':'+os.environ['TASK_RLINF'],RLINF_PATH=os.environ['TASK_RLINF'],ROBOCASA_ASSETS_PATH=str(r/'setups/robocasa365-native-20261101-r1/assets'),OPENPI_DATA_HOME=str(r/'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0')
native_env.update(ROBOCASA_NATIVE_OUTPUT=str(out),ROBOCASA_PI05_TORCH_MODEL=initializer['model_directory'],ROBOCASA_PI05_NATIVE_NORM=initializer['norm_path'])
flora_python=str(r/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python')
toolkit=r/'outputs/flora-ppo-official-inference-q7-20261101-r1/toolkit-site/nvidia/cu13'
scorer_env=dict(CUDA_VISIBLE_DEVICES=selected[-1]['visibility_token'],PYTHONPATH=str(code/'src')+':'+str(r/'deps/nvidia-flora-129a2ada-20261101-r1/repo'),CUDA_HOME=str(toolkit),CUDA_PATH=str(toolkit),CUDACXX=str(toolkit/'bin/nvcc'),PATH=str(toolkit/'bin')+':'+str(pathlib.Path(flora_python).parent)+':'+os.environ['PATH'],LD_LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),LIBRARY_PATH=str(toolkit/'lib')+':'+os.environ.get('LIBRARY_PATH',''),FLASHINFER_WORKSPACE_BASE=str(out/'cache/flashinfer'),VLLM_CACHE_ROOT=str(out/'cache/vllm'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MAX_JOBS='8')
profile=dict(arm=role,gpus=gpus,deadline_unix=deadline,closure_seconds=180,exchange=str(exchange),policy_python=sys.executable,scorer_python=flora_python,policy_env=native_env,scorer_env=scorer_env,common_before=common,training=training,scorer=scorer,arm_outputs={key:str(root/key) for key in ('S','F')},native_episodes=str(episodes),native_monitor=str(native/'monitoring/OpenDrawer/baseline-000'))
for environment, device in [(native_env,selected[0]),(scorer_env,selected[-1])]:
 environment.update(LYCO_BAIGE_GPU_MAP=str(out/'gpu-probe.json'),LYCO_BAIGE_ISOLATED_UUID=device['uuid'])
if role=='F':
 core=inputs/'PERSISTENT-CORE.xml';profile['persistent_core_qualification']=dict(path=str(core),sha256=sha(core))
write_new(out/'pipeline-profile.json',profile)
print(json.dumps(dict(event='integrated_residual_pipeline_start',arm=role,gpus=gpus,native_successes=3,scientific_optimizer_updates=0,Flora_AUROC='NOT_MEASURED')),flush=True)
subprocess.run([sys.executable,'-B',str(code/'scripts/diagnostics/robocasa_residual_pipeline.py'),'--profile',str(out/'pipeline-profile.json')],check=True,timeout=max(1,deadline-time.time()))
