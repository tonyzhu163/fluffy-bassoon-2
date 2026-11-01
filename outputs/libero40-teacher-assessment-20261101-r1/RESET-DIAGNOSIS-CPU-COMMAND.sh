bash <<'RESET_DIAGNOSIS_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/libero40-assessment-reset-diagnosis-cpu-20261101-r1
TASK_SOURCE=aab7513d96e88d90ce43ab15689547fdac82e095
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1/reset-diagnosis-cpu-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export PYTHONPATH=$TASK_CODE/src:/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+35-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=2s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+30;code=Path.cwd()
rc,zero,error,counts,analysis_rc=1,False,None,None,None
prior=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1/cpu-r1/cpu-worker-clock.json')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='assessment_residual100_CPU_clock_v1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,outer_TERM_unix=started+35,outer_KILL_unix=started+37,maximum_this_attempt_CPU_worker_seconds=100,original_CPU_worker_cap=180,prior_CPU_worker_seconds=80,prior_clock=ref(prior),scheduler_startup_allowance_seconds=60,maximum_wall_seconds_before_allowance=40,accounting='One2core CPU worker: inclusive elapsedwall plus same60second scheduler allowance; original80 charge retained, no reset/borrowing',status='started',interrupted_charge_CPU_worker_seconds=100)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={p:Path(p).read_text().strip() for p in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 assert json.loads(prior.read_text())['charged_CPU_worker_seconds']==80
 profile_path=code/'outputs/libero40-teacher-assessment-reset-diagnosis-code-20261101-r1/PROFILE.json';profile=json.loads(profile_path.read_text());admission=code/profile['admission']['path'];assert sha(admission)==profile['admission']['sha256']
 files=['scripts/diagnostics/libero_assessment_reset_pair_diagnosis.py','tests/test_libero_assessment_reset_pair_diagnosis.py','outputs/libero40-teacher-assessment-reset-diagnosis-code-20261101-r1/PROFILE.json']
 write('allocation.json',dict(source_commit=source,hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,absolute_deadline_unix=deadline,admission=ref(admission),source_files=[ref(code/p) for p in files],prior_clock=ref(prior),maximum_remaining_CPU_worker_seconds=100,original_cap=180,prior_charge=80,scope='ONLY4newtests then2bound savedreset records; no actor/checkpoint/VLA/JAXconstructor/simulator/SGD/action or oldsuite'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_libero_assessment_reset_pair_diagnosis.py','--junitxml='+str(root/'junit.xml')],cwd=code,stdout=log,stderr=subprocess.STDOUT,timeout=max(0.1,deadline-time.time()),check=False)
 counts=dict(next(ET.parse(root/'junit.xml').iter('testsuite')).attrib)
 assert result.returncode==0 and counts['tests']=='4' and all(counts[k]=='0' for k in ('errors','failures','skipped'))
 command=[sys.executable,*profile['CLI'][1:]]
 command=[str(deadline) if x=='OWNER_BOUND_ABSOLUTE_DEADLINE' else str(root/'reset-diagnosis.json') if x=='/ABS/FRESH_CPU_OUTPUT/reset-diagnosis.json' else x for x in command]
 write('analysis-command.json',dict(argv=command,profile=ref(profile_path),original_absolute_deadline_unix=deadline))
 with (root/'analysis.log').open('x') as log:analysis_rc=subprocess.run(command,cwd=code,stdout=log,stderr=subprocess.STDOUT,timeout=max(0.1,deadline-time.time()),check=False).returncode
 assert analysis_rc==0 and (root/'reset-diagnosis.json').is_file()
 assert all(x['unchanged'] for x in json.loads((root/'reset-diagnosis.json').read_text())['read_only_proof'].values())
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>100:rc,error=1,(error or '')+' Inclusive100CPU remainder exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,original_cumulative_CPU_worker_seconds=80+charged,original_remaining_CPU_worker_seconds=max(0,180-80-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(source_commit=source,passed=rc==0,JUnit=counts,analysis_exit_code=analysis_rc,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,prior_CPU_worker_seconds=80,original_cumulative_CPU_worker_seconds=80+charged,original_CPU_cap=180,GPU_seconds=0 if zero else None,new_actions=0,new_SGD=0,VLA_constructed=False,simulator_constructed=False,JAX_constructor=False,GPU_retry_admitted=False));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
RESET_DIAGNOSIS_CPU
