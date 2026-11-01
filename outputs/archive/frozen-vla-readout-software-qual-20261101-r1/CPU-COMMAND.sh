bash <<'READOUT_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=f3d0d52ba040a4bcdf3aee3ffd0c39644006f500
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-command-readout-cpu-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-readout-software-qual-20261101-r1/cpu-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export PYTHONPATH=$TASK_CODE/src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+57-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=3s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
schema_component={}
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+52
code=Path.cwd()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='linked_readout_CPU_clock_v1',phase_id='frozen-vla-readout-software-qual-20261101-r1',source_commit=source,started_unix=started,deadline_unix=deadline,maximum_CPU_worker_seconds=120,prior_CPU_charge=0,old_original1775_and_linked98_untouched=True,interrupted_charge_CPU_worker_seconds=120,status='started',attempts=1,GPU_seconds=None)
write('cpu-worker-clock.json',clock);rc,zero,error=1,False,None;component={}
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 grant=root.parent/'GRANT.json';g=json.loads(grant.read_text());assert g['maximum_aggregate_CPU_worker_seconds']==120 and g['required_source_commit']==source
 write('allocation.json',dict(format='linked_readout_CPU_ticket_v1',source_commit=source,phase_id=clock['phase_id'],hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),schema_addendum_ref=dict(path=str(root.parent/'SCHEMA-INSPECTION-ADDENDUM.json'),sha256=sha(root.parent/'SCHEMA-INSPECTION-ADDENDUM.json')),inspection_script_sha256=sha(root.parent/'SCHEMA-INSPECT.py'),inspection_inputs_sha256=sha(root.parent/'SCHEMA-INSPECTION-INPUTS.json'),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=120,prior_CPU_charge=0,old_unused_seconds_not_borrowed=True,scope='Required57 readout tests FIRST then scoped three saved-row CPU schema inspection within20seconds if originaldeadline permits; no old33/84/5,VLAweights,simulator,actions'
))
 with (root/'readout.log').open('x') as stream:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_command_diagnostic_readout.py','--junitxml='+str(root/'readout.junit.xml')],stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-4),check=False)
 component=dict(source_commit=source,tests=['tests/test_robotwin_command_diagnostic_readout.py'],return_code=result.returncode,status='passed' if result.returncode==0 else 'failed');rc=result.returncode
 if deadline-time.time()>8:
  inspect=root.parent/'SCHEMA-INSPECT.py';ins=root.parent/'SCHEMA-INSPECTION-INPUTS.json';inspection_deadline=min(deadline-3,time.time()+20)
  with (root/'schema-inspection.log').open('x') as stream:
   schema_run=subprocess.run([sys.executable,str(inspect),'--inputs',str(ins),'--inputs-sha256',sha(ins),'--output',str(root/'schema-inspection.json'),'--deadline',str(inspection_deadline)],stdout=stream,stderr=subprocess.STDOUT,timeout=max(1,inspection_deadline-time.time()),check=False)
  schema_component=dict(return_code=schema_run.returncode,status='reported' if schema_run.returncode==0 else 'failed',maximum_inspection_seconds=20,script_sha256=sha(inspect),inputs_sha256=sha(ins))
  if schema_run.returncode:rc=rc or schema_run.returncode
 else:schema_component=dict(status='not_run_insufficient_original_clock')
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:rc,error=124,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>120:rc,error=1,(error or '')+' Inclusive120CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,120-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='linked_readout_CPU_qualification_v1',source_commit=source,passed=rc==0,exit_code=rc,error=error,component=component,schema_inspection=schema_component,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=120,GPU_seconds=0 if zero else None,new_actions=0,no_VLA_model_or_simulator=True,actual_saved_cache_schema_inspection_allowed=True,old_original1775_and_linked98_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
READOUT_CPU
