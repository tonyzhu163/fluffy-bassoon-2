"""Finite resource-only public checkpoint staging/conversion/load; allocated CPU only."""
import concurrent.futures,hashlib,importlib.metadata,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path,PurePosixPath
root=Path(os.environ['TASK_ROOT']);out=Path(os.environ['TASK_OUT']);inputs=Path(os.environ['TASK_INPUT']);source=Path(os.environ['TASK_RLINF']);meta=Path(os.environ['TASK_META']);deadline=int(os.environ['TASK_DEADLINE']);py=Path(os.environ['TASK_NATIVE_PY'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
def run(args,env=None):
 remaining=deadline-time.time()-45
 if remaining<30:raise TimeoutError('Original policy CPU allocation deadline exhausted')
 print(json.dumps(dict(event='subprocess',command=list(map(str,args)),deadline_unix=deadline)),flush=True)
 subprocess.run(list(map(str,args)),check=True,timeout=remaining,env=env)
assert socket.gethostname().startswith('job-')
import torch
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split();mem=int(Path('/sys/fs/cgroup/memory.max').read_text())
assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized() and int(cpu[0])/int(cpu[1])==8 and mem==128<<30
save('zero-gpu-attestation.json',dict(hostname=socket.gethostname(),pid=os.getpid(),python=sys.version,torch=torch.__version__,torch_origin=torch.__file__,actual_gpu_count_before_mask=0,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),NVIDIA_VISIBLE_DEVICES=os.environ.get('NVIDIA_VISIBLE_DEVICES'),cpu_max=cpu,memory_bytes=mem,original_deadline_unix=deadline))
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()=='0035eef7d4d7c3b05698a8c636f4ba0ba74622fe'
assert subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()==''
assert subprocess.check_output(['git','-C',str(meta),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_CODE_PIN']
assert subprocess.check_output(['git','-C',str(meta),'status','--porcelain'],text=True).strip()==''
profile=json.loads((inputs/'CPU-PROFILE.json').read_text())
ready=Path(profile['native_setup_summary']['path']);saved=json.loads(ready.read_text());assert saved['completed'] and sha(ready)==profile['native_setup_summary']['sha256']
assert (ready.parent/'exit-code.txt').read_text().strip()=='0'
import rlinf,openpi
assert Path(rlinf.__file__).resolve().is_relative_to(source.resolve())
save('native-runtime-origins.json',dict(python=sys.executable,rlinf_origin=rlinf.__file__,openpi_origin=openpi.__file__,Torch_origin=torch.__file__,versions={name:importlib.metadata.version(name) for name in ['torch','jax','orbax-checkpoint','numpy','transformers','robocasa','robosuite']},native_setup_summary_sha256=sha(ready)))
assert importlib.metadata.version('jax')=='0.5.3' and importlib.metadata.version('orbax-checkpoint')=='0.11.13'
tokenizer=Path(os.environ['TOKENIZER_DIR'])/'big_vision/paligemma_tokenizer.model';assert tokenizer.is_file() and sha(tokenizer)=='8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6'
save('bound-tokenizer.json',dict(path=str(tokenizer),bytes=tokenizer.stat().st_size,sha256=sha(tokenizer),source='Originalnativeinstaller publicHFdownload LFSmetadata8986; actualhashverifiedinsideCPU'))
manifest=json.loads((inputs/'PUBLIC-PI05-PARAMS-MANIFEST.json').read_text());assert manifest['revision']=='c484448aba1a9b60a04c9b0ca117241518ea69f3'
rows=[r for group in manifest['included'] for r in group['files']];assert len(rows)==19 and sum(r['bytes'] for r in rows)==12440863506
payload=Path(os.environ['TASK_PAYLOAD']);converted=Path(os.environ['TASK_CONVERTED']);assert payload.is_dir() and not converted.exists()
prior=root/'outputs/robocasa365-public-policy-cpu-q1-20261101-r1/PUBLIC-PARAMS-READY.json'
saved_payload=json.loads(prior.read_text());assert saved_payload['completed'] and saved_payload['revision']==manifest['revision'] and saved_payload['total_bytes']==12440863506
for row in saved_payload['files']:
 assert Path(row['path']).is_relative_to(payload) and Path(row['path']).stat().st_size==row['bytes']
save('REUSED-ORIGINAL-PARAMS.json',dict(completed=True,prior_receipt=str(prior),prior_receipt_sha256=sha(prior),files=len(saved_payload['files']),bytes=saved_payload['total_bytes'],method='Prior actual byte/hash verification plus unchanged original files/stat; no re-download or large rehash'))
test=meta/'tests/test_robocasa_pi05_preprocessing.py'
run([py,'-B','-m','pytest','-q',str(test)+'::test_action_mean_std_output_keeps_native_eef_gripper_base_order',str(meta/'tests/test_robocasa_residual.py')+'::test_full_native_horizon_and_video_are_truth_isolated',meta/'tests/test_robocasa_residual_rewards.py','--junitxml='+str(out/'binding-junit.xml')])
norm=payload/'assets/norm_stats.json';assert norm.is_file()
run([py,'-B','-m','rlinf.utils.ckpt_convertor.openpi.convert','--mode','jax_to_openpi','--input-model',payload,'--input-norm-stats',norm,'--output-model',converted,'--output-norm-stats',converted/'robocasa365/norm_stats.json','--action-dim','32','--action-horizon','50','--max-token-len','200'])
assert sha(norm)==sha(converted/'robocasa365/norm_stats.json') and (converted/'model.safetensors').stat().st_size<=40<<30
run([py,'-B',inputs/'STRICT-LOAD.py'])
new_paths=[payload,converted,out,inputs,source,meta];total=0;measure={}
for path in new_paths:
 value=int(subprocess.check_output(['du','-sb',str(path)],text=True).split()[0]);measure[str(path)]=value;total+=value
assert total<=64<<30
save('summary.json',dict(completed=True,source_commit=os.environ['TASK_CODE_PIN'],rlinf_execution_commit='0035eef7d4d7c3b05698a8c636f4ba0ba74622fe',model_sha256=sha(converted/'model.safetensors'),normalization_sha256=sha(norm),native_setup_receipt_sha256=sha(ready),new_storage_bytes=total,storage_paths=measure,physical_actions=0,optimizer_updates=0,model_forward_calls=0,remaining_seconds=deadline-time.time()))
