"""One archived native video through the unchanged pinned official Flora CLI."""
import hashlib,importlib.metadata,json,os,socket,subprocess,sys,time
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);out=Path(os.environ['TASK_OUT']);source=Path(os.environ['TASK_WT']);inputs=Path(os.environ['TASK_INPUT']);deadline=int(os.environ['TASK_STARTED'])+2820
assert socket.gethostname().startswith('job-')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(args,**kwargs):
 remaining=deadline-time.time()-30
 if remaining<30:raise TimeoutError('Original Flora allocation deadline exhausted')
 subprocess.run(args,check=True,timeout=remaining,**kwargs)
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==os.environ['TASK_SOURCE']
assert subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()==''
admission=json.loads(Path(os.environ['TASK_TICKET']).read_text());assert admission['physical_actions']==0 and admission['maximum_inclusive_gpu_seconds']==3600
assert sha(inputs/'COMPARISON-GRANT-FOR-FLORA-Q1.json')==admission['grant_sha256']
code=root/'deps/nvidia-flora-129a2ada-20261101-r1/repo';model=root/'inputs/flora-rewarder-031299e4-20261101-r1';ready=root/'inputs/flora-rewarder-031299e4-20261101-r1-receipt/READY.json'
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==admission['official_code']
assert subprocess.check_output(['git','-C',str(code),'status','--porcelain'],text=True).strip()==''
assert sha(ready)==admission['model_ready_sha256']
manifest=json.loads(ready.read_text());assert manifest['completed'] and manifest['model_revision']==admission['model_revision']
for row in manifest['files']:assert Path(row['path']).stat().st_size==row['bytes']
episodes=Path(admission['fixture_episodes']['path']);assert sha(episodes)==admission['fixture_episodes']['sha256']
rows=json.loads(episodes.read_text());assert len(rows)==1 and rows[0]['video']==admission['fixture_video'] and sha(rows[0]['video']['path'])==rows[0]['video']['sha256']
cpu=Path('/sys/fs/cgroup/cpu.max').read_text().split();memory=int(Path('/sys/fs/cgroup/memory.max').read_text());assert int(cpu[0])/int(cpu[1])==16 and memory==96<<30
import torch
original_visibility=os.environ.get('CUDA_VISIBLE_DEVICES');assert torch.cuda.device_count()==1
if not original_visibility:os.environ['CUDA_VISIBLE_DEVICES']='0'
attestation=dict(hostname=socket.gethostname(),pid=os.getpid(),source_commit=os.environ['TASK_SOURCE'],python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,torch_origin=torch.__file__,cpu_max=cpu,memory_bytes=memory,actual_gpu_count=1,original_CUDA_VISIBLE_DEVICES=original_visibility,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],NVIDIA_VISIBLE_DEVICES=os.environ.get('NVIDIA_VISIBLE_DEVICES'),absolute_inner_deadline=deadline,packages={name:importlib.metadata.version(name) for name in ['vllm','transformers','decord','qwen-vl-utils']},script_sha256=sha(__file__),ticket_sha256=sha(os.environ['TASK_TICKET']))
(out/'runtime-attestation.json').write_text(json.dumps(attestation,indent=2)+'\n')
run([sys.executable,'-B',str(source/'scripts/[train]gpu_probe.py'),'--expected-count','1','--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')])

# These are real allocation/compiler observations, not configured positive flags.
assert os.environ.get('FLASHINFER_CUDA_ARCH_LIST') is None, 'No forced FlashInfer target architecture is admitted'
expected_toolkit=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/lib/python3.12/site-packages/nvidia/cu13'
nvcc=Path(subprocess.check_output(['which','nvcc'],text=True).strip()).resolve()
assert nvcc==(expected_toolkit/'bin/nvcc').resolve() and Path(os.environ['CUDA_HOME']).resolve()==expected_toolkit.resolve() and os.environ['CUDA_PATH']==os.environ['CUDA_HOME']
version_text=subprocess.check_output([str(nvcc),'--version'],text=True)
from flashinfer.jit.cpp_ext import get_cuda_path,get_cuda_version,is_cuda_version_at_least
from flashinfer.jit.core import current_compilation_context
selected_toolkit=Path(get_cuda_path()).resolve();selected_version=str(get_cuda_version());targets=sorted(current_compilation_context.TARGET_CUDA_ARCHS)
capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
assert selected_toolkit==expected_toolkit.resolve() and is_cuda_version_at_least('12.9')
assert targets and all(any(major==cap[0] and int(''.join(c for c in str(minor) if c.isdigit()))==cap[1] for major,minor in targets) for cap in capabilities)
toolchain=dict(completed=True,nvcc_path=str(nvcc),nvcc_bytes=nvcc.stat().st_size,nvcc_sha256=sha(nvcc),nvcc_version_output=version_text,CUDA_HOME=os.environ['CUDA_HOME'],CUDA_PATH=os.environ['CUDA_PATH'],CUDACXX=os.environ['CUDACXX'],FlashInfer_selected_toolkit=str(selected_toolkit),FlashInfer_selected_version=selected_version,TARGET_CUDA_ARCHS=[list(t) for t in targets],actual_device_capabilities=capabilities,Torch_cuda=torch.version.cuda,forced_architecture=None,policy_environment_mutated=False)
(out/'toolchain-admission.json').write_text(json.dumps(toolchain,indent=2)+'\n')
print(json.dumps(dict(event='actual_SM_toolkit_gate_passed',toolchain=toolchain)),flush=True)

profile=dict(status='ACTUAL_ALLOCATED_TOOLCHAIN_REPAIRED_FIXTURE_ADMISSION',source_commit=os.environ['TASK_SOURCE'],allocation_id=socket.gethostname(),resource=dict(gpus=1,cpus=16,ram_gib=96),deadline_unix=deadline,ticket=dict(path=os.environ['TASK_TICKET'],sha256=sha(os.environ['TASK_TICKET'])),actual_gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),archived_native_fixture=admission['fixture_episodes'],video=rows[0]['video'],model_receipt=dict(path=str(ready),sha256=sha(ready)),physical_actions=0,optimizer_steps=0)
profile.update(execution_admission=dict(path=str(inputs/'FLORA-TOOLCHAIN-R2-ADMISSION.json'),sha256=sha(inputs/'FLORA-TOOLCHAIN-R2-ADMISSION.json')),toolchain_admission=dict(path=str(out/'toolchain-admission.json'),sha256=sha(out/'toolchain-admission.json')),same_original_clock_root=os.environ['TASK_CLOCK'],prior_closed_gpu_seconds=446)
(out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
print(json.dumps(dict(event='official_flora_archived_fixture_begin',deadline_unix=deadline,remaining_seconds=deadline-time.time(),physical_actions=0)),flush=True)
run([sys.executable,'-B',str(source/'scripts/diagnostics/rlinf_flora_ppo_score.py'),'--mode','smoke','--code',str(code),'--model',str(model),'--model-receipt',str(ready),'--episodes',str(episodes),'--output',str(out/'score'),'--deadline-unix',str(deadline)])
receipt=out/'score/terminal.receipt.json';assert json.loads(receipt.read_text())['completed']
summary=dict(completed=True,official_cli_receipt=dict(path=str(receipt),sha256=sha(receipt)),physical_actions=0,optimizer_steps=0,fixture_episodes=admission['fixture_episodes'],actual_gpu_probe=profile['actual_gpu_probe'],remaining_seconds=deadline-time.time())
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
