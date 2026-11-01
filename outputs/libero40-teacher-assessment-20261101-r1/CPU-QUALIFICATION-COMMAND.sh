bash <<'TEACHER_ASSESSMENT_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=7ef8a62058773201e85c8b2ec26bf3847ff15026
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/libero40-teacher-assessment-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1/cpu-r1
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
TASK_REMAINING=$((TASK_STARTED+110-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+100;code=Path.cwd()
rc,zero,error,counts=1,False,None,None
selectors=['tests/test_vla_source40_teacher_assessment_profile.py','tests/test_vla_source40_teacher_assessment_runtime.py']
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='all40_teacher_assessment_CPU_clock_v1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,status='started',attempts=1,interrupted_charge_CPU_worker_seconds=180,all_old_CPU_balances_untouched=True)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 grant=code/'outputs/libero40-teacher-assessment-20261101-r1/GRANT.json';assert sha(grant)=='5e2384e028a023fbac2f497b1664ddfb1bd2ed4a5376e964aff6c45fea1d3436'
 assert all((code/p).is_file() for p in selectors)
 write('allocation.json',dict(format='all40_teacher_assessment_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,tests=selectors,scope='ONLY59newsynthetic cases; no realactor/checkpoint/VLA/JAXconstructor/simulator/SGD/physics, no oldsuite repeat'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*selectors,'--junitxml='+str(root/'junit.xml')],cwd=code,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib)
 rc=0 if result.returncode==0 and counts['tests']=='59' and all(counts[k]=='0' for k in ('errors','failures','skipped')) else 1
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>180:rc,error=1,(error or '')+' Inclusive180CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,180-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='all40_teacher_assessment_CPU_qualification_v1',source_commit=source,passed=rc==0,JUnit=counts,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=180,GPU_seconds=0 if zero else None,new_actions=0,new_SGD=0,real_checkpoint_loaded=False,JAX_constructor=False,VLA_constructed=False,simulator_constructed=False,actual_GPU_runtime_proof=False,all_old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
TEACHER_ASSESSMENT_CPU
