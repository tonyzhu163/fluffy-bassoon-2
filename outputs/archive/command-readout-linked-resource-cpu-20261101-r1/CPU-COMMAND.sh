bash <<'READOUT_RESOURCE_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=949c88fef18090ecb82378027f836bc932f88063
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-command-readout-linked-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/command-readout-linked-resource-cpu-20261101-r1/cpu-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export PYTHONPATH=$TASK_CODE/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+115-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+105
code=Path.cwd();required_passed=False;components={}
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='command_readout_linked_CPU_clock_v1',phase_id='command-readout-linked-resource-cpu-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,status='started',attempts=1,all_old_CPU_balances_untouched=True,GPU_seconds=None)
write('cpu-worker-clock.json',clock);rc,zero,error=1,False,None
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 grant=root.parent/'GRANT.json';assert sha(grant)=='373aa37b782f51c653c16834b05570875242e6fe8fa1086c786e0106f02cc937'
 write('allocation.json',dict(format='command_readout_resource_CPU_ticket_v1',source_commit=source,scientific_source='235fcc00f0ddf7a07590604daba9e79c9a4c3c6d',hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,scope='ONLY54new/affected resourcecases first; optionalpre-staged all90tinywidehead analysis <=30s if >=45s remain; noold57/VLA/simulator/CUDArestore'))
 def run(name,args,env,until):
  with (root/(name+'.log')).open('x') as log:
   try:
    result=subprocess.run(args,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,until-time.time()),check=False);ret=result.returncode;detail=None
   except subprocess.TimeoutExpired:ret,detail=124,'Original shared deadline; partialdata retained, no retry'
  components[name]=dict(return_code=ret,error=detail,arguments=args);write('components.json',components);return ret
 payload=code/'outputs/robotwin-command-readout-linked-20261101-r1/CPU-PAYLOAD.sh'
 env=dict(os.environ,READOUT_SOURCE=str(code),READOUT_PYTHON=sys.executable,READOUT_CPU_OUTPUT=str(root/'required'),OWNER_CPU_RESOURCE_RECEIPT=str(root/'allocation.json'))
 required_rc=run('required54',['bash',str(payload)],env,deadline-20)
 if (root/'required/junit.xml').exists():
  suite=next(ET.parse(root/'required/junit.xml').iter('testsuite'));counts=dict(suite.attrib);components['required54']['JUnit']=counts
  required_passed=required_rc==0 and counts['tests']=='54' and all(counts[k]=='0' for k in ('errors','failures','skipped'))
 script=root.parent/'OPTIONAL-WIDE5000-ANALYSIS.py';optional_rc=0
 if required_passed and script.exists() and sha(script)=='4084144f9871b3de3c248d486093455e301e139015bdf9f1b88654435a3ebb88' and deadline-time.time()>=45:
  optional_deadline=min(deadline-15,time.time()+30)
  args=[sys.executable,str(script),'--endpoint','/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/reactive-curve-v1/action/command-diagnostic-fit-action-linked-r1/endpoint-005000.pt','--endpoint-sha256','19ed635cbff5427a184c1969fa6955551f8f5f33a260c1747507617a93deb200','--cache','/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cache-readout-correction-cpu-20261101-r1/cpu-r1/complete-diagnostic-cache.json','--cache-sha256','65c708bf3b2c85509ed6f35150768149569558def96dd981e5b548f10a673cf8','--output',str(root/'wide5000-analysis.json'),'--deadline',str(optional_deadline),'--visible-gpu-count','0']
  optional_rc=run('optional-wide5000',args,dict(os.environ),optional_deadline+1)
 else:components['optional-wide5000']=dict(status='skipped_required_failed_or_original_clock_insufficient_or_not_prestaged')
 write('components.json',components)
 rc=0 if required_passed and optional_rc==0 else 1
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>180:rc,error=1,(error or '')+' Inclusive180CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,180-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='command_readout_linked_CPU_qualification_v1',source_commit=source,passed=rc==0,required_passed=required_passed,components=components,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=180,GPU_seconds=0 if zero else None,new_actions=0,scope='required54resourcecases +conditionallyoptional all90CPUwidehead analysis; noVLA/CUDArestore/physics',all_old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
READOUT_RESOURCE_CPU
