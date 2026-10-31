bash <<'PRESERVATION_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=d4425999d4c4ef1c572cc28abe9003f012aaeedf
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-cache04-recovery-20261101-r2
TASK_BRIDGE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-fitted-physical-cpu-20261101-r2
TASK_READOUT=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-command-readout-cpu-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-preservation-software-qual-20261101-r1/cpu-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
# Absolute57s stop +3s forced cleanup +60s startup allowance =120 inclusive.
TASK_REMAINING=$((TASK_STARTED+57-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=3s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" "$TASK_BRIDGE" "$TASK_READOUT" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
root,source,started,bridge,readout=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]),Path(sys.argv[4]),Path(sys.argv[5]);deadline=started+52
code=Path.cwd();bridge_source='008fffa0282c21ebdfab387872a46603f0779c42';readout_source='f3d0d52ba040a4bcdf3aee3ffd0c39644006f500'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='linked_preservation_CPU_clock_v1',phase_id='frozen-vla-preservation-software-qual-20261101-r1',source_commit=source,started_unix=started,deadline_unix=deadline,maximum_cpu_worker_seconds=120,prior_linked_cpu_charge=0,old_phase_cpu_charge=1775,old_phase_cpu_ceiling=1800,old_25_seconds_untouched=True,interrupted_charge_cpu_worker_seconds=120,status='started',attempts=1,charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc,zero_gpu,error=1,False,None
components={}
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-')
 assert sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count()
 zero_gpu=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current']
 raw={p:Path(p).read_text().strip() for p in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero_gpu,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,python=sys.version,raw_cgroup=raw,actual_cpu_cores=cores,actual_memory_bytes=memory,required_cpu_cores=2,required_memory_bytes=16*1024**3))
 assert zero_gpu and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 for w,pin in ((bridge,bridge_source),(readout,readout_source)):
  assert subprocess.check_output(['git','-C',str(w),'rev-parse','HEAD'],text=True).strip()==pin
  assert not subprocess.check_output(['git','-C',str(w),'status','--porcelain'],text=True).strip()
 grant=root.parent/'GRANT.json';addendum=root.parent/'OPTIONAL-READOUT-ADDENDUM.json'
 assert json.loads(grant.read_text())['maximum_aggregate_CPU_worker_seconds']==120
 write('allocation.json',dict(grant_ref=dict(path=str(grant),sha256=sha(grant)),optional_addendum_ref=dict(path=str(addendum),sha256=sha(addendum)),format='linked_preservation_CPU_ticket_v1',phase_id=clock['phase_id'],hostname=socket.gethostname(),gpu_count=0,source_commit=source,bridge_source=bridge_source,readout_source=readout_source,output=str(root),maximum_cpu_worker_seconds=120,old_phase_cpu_charge=1775,old_phase_25_seconds_untouched=True,actual_cpu_cores=cores,actual_memory_bytes=memory,runtime=sys.executable,zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,deadline_unix=deadline,scope='Only33newcache recovery tests +3failedbridge fixture nodes +optional57newreadout tests; no old84/old5 or actual model/simulator/bank actions'))
 def run(name,w,pin,targets):
  components[name]=dict(source_commit=pin,worktree=str(w),status='started',tests=targets)
  write('components.json',components)
  with (root/(name+'.log')).open('x') as log:
   subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*targets,'--junitxml='+str(root/(name+'.junit.xml'))],stdout=log,stderr=subprocess.STDOUT,cwd=w,env=dict(os.environ,PYTHONPATH=str(w/'src')),timeout=max(1,deadline-time.time()-4),check=True)
  components[name]['status']='passed';write('components.json',components)
 run('cache-recovery',code,source,['tests/test_robotwin_cache04_recovery.py'])
 run('bridge-fixture',bridge,bridge_source,['tests/test_robotwin_fitted_physical.py::test_zero_is_fitted_baseline_raw_native_is_separate_and_nonzero_preserves_frozen_hook[kv]','tests/test_robotwin_fitted_physical.py::test_zero_is_fitted_baseline_raw_native_is_separate_and_nonzero_preserves_frozen_hook[action]','tests/test_robotwin_fitted_physical.py::test_recorded_request_and_public2076_are_after_frontend_physical_before_hidden_calibration'])
 if deadline-time.time()>8:run('readout',readout,readout_source,['tests/test_robotwin_command_diagnostic_readout.py'])
 else:components['readout']=dict(source_commit=readout_source,status='not_run_insufficient_same_clock');write('components.json',components)
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:rc,error=124,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>120:rc,error=1,(error or '')+' Inclusive120s CPU envelope exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,120-charged),charged_gpu_seconds=0 if zero_gpu else None)
 write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='linked_preservation_CPU_qualification_v1',source_commit=source,phase_id=clock['phase_id'],passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,components=components,no_actual_bank=True,no_vla_model=True,no_simulator=True,robot_actions=0,charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=120,charged_gpu_seconds=0 if zero_gpu else None,old_phase_25_seconds_untouched=True))
 (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
PRESERVATION_CPU
