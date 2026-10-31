bash <<'DIAGNOSTIC_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=06b1425607783082df071f4dc21414a096a6f5d8
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-command-diagnostic-cpu-20261101-r2
TASK_BRIDGE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-fitted-physical-cpu-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/robotwin-command-diagnostic-cpu-20261101-r1
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
# Absolute47s process stop +3s forced cleanup +60s startup allowance =110inclusive.
TASK_REMAINING=$((TASK_STARTED+47-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=3s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" "$TASK_BRIDGE" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
root,source,started,bridge=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]),Path(sys.argv[4]);deadline=started+42
code=Path.cwd();bridge_source='bbb200ef804dbef33ae5e7156a93803d8b2cef3c'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='frozen_pi05_endtoend_cpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',source_commit=source,started_unix=started,deadline_unix=deadline,maximum_cpu_worker_seconds=110,prior_phase_cpu_charge=1682,maximum_phase_cpu_worker_seconds=1800,interrupted_charge_cpu_worker_seconds=110,status='started',attempts=1,charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc,zero_gpu,error=1,False,None
bridge_status='not_run'
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-')
 assert sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count()
 zero_gpu=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current']
 raw={p:Path(p).read_text().strip() for p in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero_gpu,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,python=sys.version,raw_cgroup=raw,actual_cpu_cores=cores,actual_memory_bytes=memory,required_cpu_cores=2,required_memory_bytes=16*1024**3))
 assert zero_gpu and cores==2 and memory==16*1024**3
 assert str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 assert subprocess.check_output(['git','-C',str(bridge),'rev-parse','HEAD'],text=True).strip()==bridge_source
 assert not subprocess.check_output(['git','-C',str(bridge),'status','--porcelain'],text=True).strip()
 write('allocation.json',dict(format='frozen_pi05_command_diagnostic_cpu_ticket_v1',phase_id='frozen-pi05-endtoend-20261031-r1',hostname=socket.gethostname(),gpu_count=0,source_commit=source,bridge_source=bridge_source,output=str(root),maximum_cpu_worker_seconds=110,prior_phase_cpu_charge=1682,maximum_phase_cpu_worker_seconds=1800,actual_cpu_cores=cores,actual_memory_bytes=memory,runtime=sys.executable,zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,deadline_unix=deadline,scope='Only42newdiagnostic+42unqualifiedcurve+optional8bridge syntheticCPU cases;no actual bank/cache/model/simulator/robot actions'))
 with (root/'diagnostic-curve-tests.log').open('x') as log:
  subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_command_diagnostic.py','tests/test_robotwin_reactive_curve.py','--junitxml='+str(root/'diagnostic-curve.junit.xml')],stdout=log,stderr=subprocess.STDOUT,cwd=code,env=dict(os.environ),timeout=max(1,deadline-time.time()-4),check=True)
 if deadline-time.time()>8:
  bridge_status='started'
  with (root/'bridge-tests.log').open('x') as log:
   subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_fitted_physical.py','--junitxml='+str(root/'bridge.junit.xml')],stdout=log,stderr=subprocess.STDOUT,cwd=bridge,env=dict(os.environ,PYTHONPATH=str(bridge/'src')),timeout=max(1,deadline-time.time()-3),check=True)
  bridge_status='completed'
 else:bridge_status='not_run_insufficient_same_clock'
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:rc,error=124,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>110:rc,error=1,(error or '')+' Inclusive110s CPU envelope exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,110-charged),charged_gpu_seconds=0 if zero_gpu else None)
 write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='frozen_pi05_command_diagnostic_cpu_qualification_v1',source_commit=source,bridge_source=bridge_source,phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,tests=['tests/test_robotwin_command_diagnostic.py','tests/test_robotwin_reactive_curve.py'],bridge_status=bridge_status,no_actual_bank=True,no_vla_model=True,no_simulator=True,robot_actions=0,optimizer_training=False,charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=110,charged_gpu_seconds=0 if zero_gpu else None))
 (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
DIAGNOSTIC_CPU
