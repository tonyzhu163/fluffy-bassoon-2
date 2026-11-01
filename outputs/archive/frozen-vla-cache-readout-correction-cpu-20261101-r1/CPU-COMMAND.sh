bash <<'CORRECTION_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=400eed69ead8b8672686c62f33ed5531f6f60558
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-cache-state-abi-cpu-20261101-r1
TASK_READOUT=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-command-readout-cpu-20261101-r2
TASK_BRIDGE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-fitted-physical-cpu-20261101-r3
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cache-readout-correction-cpu-20261101-r1/cpu-r1
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
# Whole300: absolute235s process stop +5s forced cleanup +60s startup allowance.
TASK_REMAINING=$((TASK_STARTED+235-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" "$TASK_READOUT" "$TASK_BRIDGE" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started,readout,bridge=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]),Path(sys.argv[4]),Path(sys.argv[5]);deadline=started+225
code=Path.cwd();readout_source='235fcc00f0ddf7a07590604daba9e79c9a4c3c6d';bridge_source='52022ca02e6ae7f2803bec0e082faf4da21cfef1'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
def clean(w,pin):
 return subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip()==pin and not subprocess.check_output(['git','-C',str(w),'status','--porcelain'],text=True).strip()
clock=dict(format='linked_cache_readout_correction_CPU_clock_v1',phase_id='frozen-vla-cache-readout-correction-cpu-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=300,prior_phase_CPU_charge=0,all_old_unused25_22_51_untouched=True,interrupted_charge_CPU_worker_seconds=300,status='started',attempts=1,GPU_seconds=None)
write('cpu-worker-clock.json',clock);rc,zero,error=1,False,None;components={}
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 assert clean(readout,readout_source)
 grant=root.parent/'GRANT.json';g=json.loads(grant.read_text());assert g['maximum_CPU_worker_seconds']==300 and g['required_cache_source']==source
 write('allocation.json',dict(format='linked_correction_CPU_ticket_v1',source_commit=source,readout_source=readout_source,phase_id=clock['phase_id'],hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=300,old_CPU_budgets_untouched=True,scope='Cache12 +oneaffectedreadout +optional3affectedbridge thenindependentallocatedCPU full90seal; noold84/33/56/5,VLAweights,SGD,simulator,GPU,reencode,payloadwrites'))
 def run(name,w,pin,targets,reserve=10,env_override=None):
  if deadline-time.time()<=reserve+2:
   components[name]=dict(status='not_run_original_clock',source_commit=pin);write('components.json',components);return 124
  env=dict(os.environ,PYTHONPATH=str(w/'src'))
  if env_override:env.update(env_override)
  components[name]=dict(status='started',source_commit=pin,arguments=targets);write('components.json',components)
  try:
   with (root/(name+'.log')).open('x') as stream:result=subprocess.run(targets,stdout=stream,stderr=subprocess.STDOUT,cwd=w,env=env,timeout=max(1,deadline-time.time()-reserve),check=False)
   result_code=result.returncode;detail=None
  except subprocess.TimeoutExpired:result_code,detail=124,'Original shared absolute deadline; no new allowance'
  components[name].update(status='passed' if result_code==0 else 'failed',return_code=result_code,error=detail);write('components.json',components);return result_code
 cache_rc=run('cache-tests',code,source,[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_cache_state_abi.py','tests/test_robotwin_teacher_bank.py','-k','state_abi or full_twelve_cache_join or full_cache_join_rejects or two_different_anchors','--junitxml='+str(root/'cache-tests.junit.xml')],reserve=30)
 readout_rc=run('readout-test',readout,readout_source,[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_command_diagnostic_readout.py::test_common_loop_natural400_and_full_episode_admission_use_existing_canonical_ledger','--junitxml='+str(root/'readout-test.junit.xml')],reserve=20)
 if cache_rc==0:
  suite=next(ET.parse(root/'cache-tests.junit.xml').iter('testsuite'))
  if suite.attrib['tests']=='12' and all(suite.attrib[k]=='0' for k in ('errors','failures','skipped')):
   request=code/'outputs/robotwin-cache-state-abi-20261101-r1/SEAL-REQUEST.json';assert sha(request)=='ba3aa2d3ba42d39f468c445a3cbf7992f5562a80ba81c33051e3482e3e6ded7c'
   payload=code/'outputs/robotwin-command-diagnostic-20261101-r1/DATA-PAYLOAD.sh'
   seal_rc=run('seal',code,source,['bash',str(payload)],reserve=10,env_override=dict(DATA_OPERATION='seal',DATA_REQUEST=str(request),DATA_REQUEST_SHA=sha(request),DATA_OUTPUT=str(root/'complete-diagnostic-cache.json'),OWNER_DATA_DEADLINE=str(deadline-10)))
   if seal_rc==0:
    value=json.loads((root/'complete-diagnostic-cache.json').read_text());assert value['complete'] and value['diagnostic12_fit_eligible'] and not value['fixed12_fit_eligible'] and not value['source_eligible'] and not value['source_student_admitted'] and not value['resource_fit_admitted']
    write('seal-completion.json',dict(format=value['format'],records=len(value['records']),teacher_ids=sorted({r['context']['teacher_id'] for r in value['records']}),complete_cache_ref=dict(path=str(root/'complete-diagnostic-cache.json'),sha256=sha(root/'complete-diagnostic-cache.json')),failed04_still_false=value['bank']['outcome_evidence']['phone-04']['success'] is False,source_eligible=False,fit_resources_admitted=False,new_robot_actions=0,cache_payloads_rewritten=False))
  else:components['seal']=dict(status='not_run_cache_count_or_skip_mismatch');write('components.json',components)
 else:components['seal']=dict(status='not_run_cache_test_gate_failed');write('components.json',components)
 if deadline-time.time()>8 and bridge.exists() and clean(bridge,bridge_source):
  run('bridge-tests',bridge,bridge_source,[sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_fitted_physical.py::test_zero_is_fitted_baseline_raw_native_is_separate_and_nonzero_preserves_frozen_hook[kv]','tests/test_robotwin_fitted_physical.py::test_zero_is_fitted_baseline_raw_native_is_separate_and_nonzero_preserves_frozen_hook[action]','tests/test_robotwin_fitted_physical.py::test_recorded_request_and_public2076_are_after_frontend_physical_before_hidden_calibration','--junitxml='+str(root/'bridge-tests.junit.xml')],reserve=4)
 else:components['bridge-tests']=dict(status='optional_not_run_not_ready_or_original_clock');write('components.json',components)
 rc=0 if all(components.get(k,{}).get('return_code')==0 for k in ('cache-tests','readout-test','seal')) and components.get('bridge-tests',{}).get('return_code',0)==0 else 1
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>300:rc,error=1,(error or '')+' Inclusive300CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,300-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='linked_cache_readout_correction_CPU_qualification_v1',source_commit=source,passed=rc==0,exit_code=rc,error=error,components=components,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=300,GPU_seconds=0 if zero else None,new_actions=0,no_VLA_modelweights_SGD_simulator_or_reencode=True,old_unused25_22_51_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
CORRECTION_CPU
