bash <<'RESTORE_REPAIR_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=d2369fb94a024315c7ef41d7021bebe171c8b515
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi05-headroom-recovery-r3-20261101-r1
TASK_PHASE=/mowenjie/fluffy-bassoon-2/outputs/libero-headroom-restore-repair-cpu-20261101-r1
TASK_OUT=$TASK_PHASE/cpu-r1
TASK_PROBE=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+110-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PROBE" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" "$TASK_MODEL" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started,model_python=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]),sys.argv[4]
code=Path.cwd();deadline=started+100
rc,zero,error,counts=1,False,None,None
components={}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='headroom_restore_repair_CPU_clock_v1',phase_id='libero-headroom-restore-repair-cpu-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,status='started',attempts=1,interrupted_charge_CPU_worker_seconds=180,old_CPU_balances_untouched=True)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==8 and memory==64*1024**3 and str(torch.__version__).startswith('2.11.')
 phase=root.parent;profile=phase/'GUARD-PROFILE.json';grant=phase/'GRANT.json'
 assert True  # Fresh independently funded180CPU phase; all oldunused remain closed
 assert sha(profile)=='31374ffd99bbdb2c7a01555c62b076be0869ea317c5011fcc1a7d0f36f8cdb49'
 assert sha(grant)=='d84c4597aa065c6076aeea9dd897670dea79f1c0b6703c272bf2e5d667051b1f'
 write('allocation.json',dict(format='headroom_restore_repair_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),actual_GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,cuda_initialized=False,grant_ref=dict(path=str(grant),sha256=sha(grant)),profile_ref=dict(path=str(profile),sha256=sha(profile)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=180,scope='ONLY20new restore/migration cases+1affectedflow selector, then ORIGINAL431MB productionguard CPUdryrun; no old4/VLA/sample/update/simulator/actions'))
 env=dict(os.environ);env['PYTHONPATH']=str(code/'src')+':/mowenjie/fluffy-bassoon-2/envs/dsrl-pi0-overlay-20261029-r1:/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1'
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([model_python,'-m','pytest','-q','-p','no:cacheprovider','tests/test_vla_headroom_restore_guard.py','tests/test_vla_target_online_dsrl_headroom_learning.py::test_fresh_native_readmission_aborts_before_any_sgd_and_cannot_resume_past_it','--junitxml='+str(root/'junit.xml')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-10),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib);components['tests']=result.returncode
 assert result.returncode==0 and counts['tests']=='21' and all(counts[k]=='0' for k in ('errors','failures','skipped'))
 with (root/'guard-dryrun.log').open('x') as log:
  result=subprocess.run([model_python,'-m','fluffy_bassoon.meta_rl.vla_headroom_restore_guard','--profile',str(profile),'--profile-sha256',sha(profile),'--output',str(root/'real-guard.json')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-10),check=False)
 components['production_guard_dryrun']=result.returncode
 assert result.returncode==0
 proof=json.loads((root/'real-guard.json').read_text());assert proof['passed'] and proof['input_SHA_before']==proof['input_SHA_after'] and proof['new_SGD']==0 and proof['new_robot_actions']==0

 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>180:rc,error=1,(error or '')+' Inclusive180CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,180-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='headroom_restore_repair_CPU_result_v1',source_commit=source,passed=rc==0,JUnit=counts,component_return_codes=components,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=180,GPU_seconds=0 if zero else None,new_actions=0,new_SGD=0,qualification_not_GPU_resource_admission=True,old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
RESTORE_REPAIR_CPU
