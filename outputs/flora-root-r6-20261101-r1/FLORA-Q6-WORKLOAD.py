"""One archived native video through the unchanged pinned official Flora CLI."""
import hashlib,importlib.metadata,json,os,socket,subprocess,sys,time,shutil,urllib.request,re
from pathlib import Path
root=Path(os.environ['TASK_ROOT']);out=Path(os.environ['TASK_OUT']);source=Path(os.environ['TASK_WT']);inputs=Path(os.environ['TASK_INPUT']);deadline=int(os.environ['TASK_STARTED'])+1920
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
# All exact official wheels are staged and installed INSIDE this originalGPU clock.
wheel_spec=json.loads((inputs/'FLORA-Q6-WHEEL-BINDING.json').read_text());wheels=out/'toolkit-wheels';wheels.mkdir();wheel_files=[]
for row in wheel_spec['wheels']:
 local=wheels/row['filename'];h=hashlib.sha256();count=0
 with urllib.request.urlopen(row['url'],timeout=25) as response,local.open('xb') as stream:
  while True:
   if time.time()>deadline-120:raise TimeoutError('OriginalFloraGPUdeadline exhausted while staging pinned toolkit')
   b=response.read(4<<20)
   if not b:break
   h.update(b);count+=len(b);stream.write(b)
  stream.flush();os.fsync(stream.fileno())
 assert count==row['bytes'] and h.hexdigest()==row['sha256'];wheel_files.append(str(local))
uv=root/'setups/robocasa365-native-20261101-r1/bootstrap/bin/uv';assert uv.is_file()
run([str(uv),'pip','install','--python',sys.executable,'--target',str(out/'toolkit-site'),'--no-deps',*wheel_files])
(out/'toolkit-wheel-receipt.json').write_text(json.dumps(dict(completed=True,wheels=wheel_spec['wheels'],installed_target=str(out/'toolkit-site'),installer_path=str(uv),installer_sha256=sha(uv),retained_flora_environment_modified=False),indent=2)+'\n')
# Resolve actual build executables before any55GB model load.
ninja=shutil.which('ninja');cxx=shutil.which(os.environ.get('CXX','c++'))
assert ninja and cxx, 'Native ninja/C++ build tools must resolve in the actual job PATH'
ninja=Path(ninja).resolve();cxx=Path(cxx).resolve()
assert ninja==(root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/ninja').resolve()
assert sha(ninja)=='08639e194fffa7f08b259fc4abfa4803aff66b64de52549cee42ec527d55cea6'
ninja_process=subprocess.run([str(ninja),'--version'],text=True,capture_output=True,check=True);ninja_version=ninja_process.stdout.strip()
cxx_version=subprocess.check_output([str(cxx),'--version'],text=True)
build_tools=dict(completed=True,PATH=os.environ['PATH'],ninja_path=str(ninja),ninja_sha256=sha(ninja),ninja_version=ninja_version,ninja_command_exit=ninja_process.returncode,cxx_variable=os.environ.get('CXX'),cxx_path=str(cxx),cxx_sha256=sha(cxx),cxx_version=cxx_version)
(out/'build-tools-admission.json').write_text(json.dumps(build_tools,indent=2)+'\n')
print(json.dumps(dict(event='actual_build_tools_gate_passed',build_tools=build_tools)),flush=True)

import torch
original_visibility=os.environ.get('CUDA_VISIBLE_DEVICES');assert torch.cuda.device_count()==1
if not original_visibility:os.environ['CUDA_VISIBLE_DEVICES']='0'
attestation=dict(hostname=socket.gethostname(),pid=os.getpid(),source_commit=os.environ['TASK_SOURCE'],python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,torch_origin=torch.__file__,cpu_max=cpu,memory_bytes=memory,actual_gpu_count=1,original_CUDA_VISIBLE_DEVICES=original_visibility,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],NVIDIA_VISIBLE_DEVICES=os.environ.get('NVIDIA_VISIBLE_DEVICES'),absolute_inner_deadline=deadline,packages={name:importlib.metadata.version(name) for name in ['vllm','transformers','decord','qwen-vl-utils']},script_sha256=sha(__file__),ticket_sha256=sha(os.environ['TASK_TICKET']))
(out/'runtime-attestation.json').write_text(json.dumps(attestation,indent=2)+'\n')
run([sys.executable,'-B',str(source/'scripts/[train]gpu_probe.py'),'--expected-count','1','--purpose','evaluation','--baige-container','--output',str(out/'gpu-probe.json')])

# These are real allocation/compiler observations, not configured positive flags.
assert os.environ.get('FLASHINFER_CUDA_ARCH_LIST') is None, 'No forced FlashInfer target architecture is admitted'
expected_toolkit=out/'toolkit-site/nvidia/cu13'
nvcc=Path(subprocess.check_output(['which','nvcc'],text=True).strip()).resolve()
assert nvcc==(expected_toolkit/'bin/nvcc').resolve() and Path(os.environ['CUDA_HOME']).resolve()==expected_toolkit.resolve() and os.environ['CUDA_PATH']==os.environ['CUDA_HOME']
version_text=subprocess.check_output([str(nvcc),'--version'],text=True)
# Effective13.0 runtime headers must agree with actual13.0 compiler; no disabledCCCLguard.
header=expected_toolkit/'include/cuda_runtime_api.h';driver_header=expected_toolkit/'include/cuda.h'
header_version=int(re.search(r'#define\s+CUDART_VERSION\s+(\d+)',header.read_text()).group(1));driver_version=int(re.search(r'#define\s+CUDA_VERSION\s+(\d+)',driver_header.read_text()).group(1))
assert header_version==13000 and driver_version==13000 and 'release 13.0' in version_text
cccl=root/'envs/flora-rewarder-129a2ada-py312-20261101-r1/lib/python3.12/site-packages/flashinfer/data/cccl/libcudacxx/include'
probe_source=out/'toolkit-compatibility.cu';probe_source.write_text('#include <cuda_runtime.h>\n#include <cuda/std/__cccl/cuda_toolkit.h>\nstatic_assert(CUDART_VERSION/1000 == __CUDACC_VER_MAJOR__);\nstatic_assert((CUDART_VERSION%1000)/10 == __CUDACC_VER_MINOR__);\nint toolkit_probe(){return CUDART_VERSION;}\n')
compile_command=[str(nvcc),'-std=c++17','-I'+str(cccl),'-c',str(probe_source),'-o',str(out/'toolkit-compatibility.o')]
run(compile_command)
(out/'toolkit-compatibility.json').write_text(json.dumps(dict(completed=True,command=compile_command,compiler_sha256=sha(nvcc),runtime_header_sha256=sha(header),driver_header_sha256=sha(driver_header),CUDART_VERSION=header_version,CUDA_VERSION=driver_version,source_sha256=sha(probe_source),object_sha256=sha(out/'toolkit-compatibility.o'),compatibility_guard_disabled=False,FlashInfer_architecture_override=None),indent=2)+'\n')
print(json.dumps(dict(event='actual_matching13_0_compiler_header_compile_passed',nvcc_version=version_text,CUDART_VERSION=header_version)),flush=True)
from flashinfer.jit.cpp_ext import get_cuda_path,get_cuda_version,is_cuda_version_at_least
from flashinfer.jit.core import current_compilation_context
selected_toolkit=Path(get_cuda_path()).resolve();selected_version=str(get_cuda_version());targets=sorted(current_compilation_context.TARGET_CUDA_ARCHS)
capabilities=[list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
assert selected_toolkit==expected_toolkit.resolve() and is_cuda_version_at_least('12.9')
assert targets and all(any(major==cap[0] and int(''.join(c for c in str(minor) if c.isdigit()))==cap[1] for major,minor in targets) for cap in capabilities)
toolchain=dict(completed=True,nvcc_path=str(nvcc),nvcc_bytes=nvcc.stat().st_size,nvcc_sha256=sha(nvcc),nvcc_version_output=version_text,CUDA_HOME=os.environ['CUDA_HOME'],CUDA_PATH=os.environ['CUDA_PATH'],CUDACXX=os.environ['CUDACXX'],FlashInfer_selected_toolkit=str(selected_toolkit),FlashInfer_selected_version=selected_version,TARGET_CUDA_ARCHS=[list(t) for t in targets],actual_device_capabilities=capabilities,Torch_cuda=torch.version.cuda,forced_architecture=None,policy_environment_mutated=False)
(out/'toolchain-admission.json').write_text(json.dumps(toolchain,indent=2)+'\n')
print(json.dumps(dict(event='actual_SM_toolkit_gate_passed',toolchain=toolchain)),flush=True)

# Compile/load the actual affected FlashInfer sampling extension before model loading.
# This makes no sampling call, random draw, policy/model constructor or action.
sampling_code = """import hashlib,json,os,time
from pathlib import Path
from flashinfer.jit.sampling import gen_sampling_module
started=time.time()
module=gen_sampling_module().build_and_load()
value=dict(completed=True,api='flashinfer.jit.sampling.gen_sampling_module().build_and_load()',elapsed_seconds=time.time()-started,sampling_calls=0,model_constructors=0,physical_actions=0,guard_disabled=False,module_type=type(module).__module__+'.'+type(module).__name__)
(Path(os.environ['TASK_OUT'])/'sampling-module-admission.json').write_text(json.dumps(value,indent=2)+chr(10))
print(json.dumps(dict(event='actual_flashinfer_sampling_module_build_passed',**value)),flush=True)
"""
run([sys.executable,'-B','-c',sampling_code])
sampling_receipt=out/'sampling-module-admission.json'
assert json.loads(sampling_receipt.read_text())['completed']
print(json.dumps(dict(event='sampling_module_qualified_before_model',remaining_seconds=deadline-time.time())),flush=True)

profile=dict(status='ACTUAL_ALLOCATED_TOOLCHAIN_REPAIRED_FIXTURE_ADMISSION',source_commit=os.environ['TASK_SOURCE'],allocation_id=socket.gethostname(),resource=dict(gpus=1,cpus=16,ram_gib=96),deadline_unix=deadline,ticket=dict(path=os.environ['TASK_TICKET'],sha256=sha(os.environ['TASK_TICKET'])),actual_gpu_probe=dict(path=str(out/'gpu-probe.json'),sha256=sha(out/'gpu-probe.json')),archived_native_fixture=admission['fixture_episodes'],video=rows[0]['video'],model_receipt=dict(path=str(ready),sha256=sha(ready)),physical_actions=0,optimizer_steps=0)
profile.update(build_tools_admission=dict(path=str(out/'build-tools-admission.json'),sha256=sha(out/'build-tools-admission.json')),execution_admission=dict(path=str(inputs/'FLORA-CURAND-R6-ADMISSION.json'),sha256=sha(inputs/'FLORA-CURAND-R6-ADMISSION.json')),toolchain_admission=dict(path=str(out/'toolchain-admission.json'),sha256=sha(out/'toolchain-admission.json')),same_original_clock_root=os.environ['TASK_CLOCK'],prior_closed_gpu_seconds=1372)
profile['sampling_module_admission']=dict(path=str(sampling_receipt),sha256=sha(sampling_receipt))
(out/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
print(json.dumps(dict(event='official_flora_archived_fixture_begin',deadline_unix=deadline,remaining_seconds=deadline-time.time(),physical_actions=0)),flush=True)
run([sys.executable,'-B',str(source/'scripts/diagnostics/rlinf_flora_ppo_score.py'),'--mode','smoke','--code',str(code),'--model',str(model),'--model-receipt',str(ready),'--episodes',str(episodes),'--output',str(out/'score'),'--deadline-unix',str(deadline)])
receipt=out/'score/terminal.receipt.json';assert json.loads(receipt.read_text())['completed']
summary=dict(completed=True,official_cli_receipt=dict(path=str(receipt),sha256=sha(receipt)),physical_actions=0,optimizer_steps=0,fixture_episodes=admission['fixture_episodes'],actual_gpu_probe=profile['actual_gpu_probe'],remaining_seconds=deadline-time.time())
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
