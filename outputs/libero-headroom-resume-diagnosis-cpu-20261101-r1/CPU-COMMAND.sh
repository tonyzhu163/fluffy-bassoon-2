bash <<'RESUME_DIAGNOSIS_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=00e131a39eca9ef24e5529ec36b665baf9ff3bad
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/libero-headroom-resume-diagnosis-20261101-r1
TASK_PHASE=/mowenjie/fluffy-bassoon-2/outputs/libero-headroom-resume-diagnosis-cpu-20261101-r1
TASK_OUT=$TASK_PHASE/cpu-r1
TASK_PROBE=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+230-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PROBE" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" "$TASK_MODEL" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started,model_python=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]),sys.argv[4]
code=Path.cwd();deadline=started+220
rc,zero,error,counts=1,False,None,None
components={}
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='headroom_resume_diagnosis_CPU_clock_v1',phase_id='libero-headroom-resume-diagnosis-cpu-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=300,status='started',attempts=1,interrupted_charge_CPU_worker_seconds=300,old_CPU_balances_untouched=True)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,12)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==8 and memory==64*1024**3 and str(torch.__version__).startswith('2.11.')
 phase=root.parent;profile=phase/'PROFILE.json';grant=phase/'GRANT.json'
 assert sha(profile)=='0812f6c8751cef8f0c92a03a523eb25ed8a043e771c32971835a902c4bc9430e'
 assert sha(grant)=='b89a420776c7112d9ce9a1335b28c34e755cc21cc69e72e9bf99365947d10480'
 write('allocation.json',dict(format='headroom_resume_diagnosis_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),actual_GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,cuda_initialized=False,grant_ref=dict(path=str(grant),sha256=sha(grant)),profile_ref=dict(path=str(profile),sha256=sha(profile)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=300,scope='Four scoped synthetic checks then original small pixelSAC CPU initialize/restore/exact comparisons; no VLA/sample/update/simulator/actions'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_libero_headroom_resume_diagnosis.py','--junitxml='+str(root/'junit.xml')],cwd=code,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-10),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib);components['tests']=result.returncode
 assert result.returncode==0 and counts['tests']=='4' and all(counts[k]=='0' for k in ('errors','failures','skipped'))
 env=dict(os.environ);env['PYTHONPATH']='/mowenjie/fluffy-bassoon-2/envs/dsrl-pi0-overlay-20261029-r1:/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi05-headroom-learning-20261101-r1/src'
 with (root/'inspection.log').open('x') as log:
  result=subprocess.run([model_python,str(code/'scripts/diagnostics/libero_headroom_resume_diagnosis.py'),'--profile',str(profile),'--profile-sha256',sha(profile),'--ticket',str(root/'allocation.json'),'--output',str(root/'diagnosis.json')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()-10),check=False)
 components['inspection']=result.returncode
 assert result.returncode==0
 diagnosis=json.loads((root/'diagnosis.json').read_text());assert diagnosis['all_inputs_unchanged'] and diagnosis['new_SGD']==0 and diagnosis['new_robot_actions']==0
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>300:rc,error=1,(error or '')+' Inclusive300CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,300-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='headroom_resume_diagnosis_CPU_result_v1',source_commit=source,passed=rc==0,JUnit=counts,component_return_codes=components,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=300,GPU_seconds=0 if zero else None,new_actions=0,new_SGD=0,classification_is_diagnosis_not_production_admission=True,old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
RESUME_DIAGNOSIS_CPU
