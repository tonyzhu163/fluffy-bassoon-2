bash <<'PROVENANCE_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=5470090165f89a114afe544758ab89d5d956fe76
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-readout-provenance-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/command-readout-provenance-cpu-20261101-r1/cpu-r1
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
TASK_REMAINING=$((TASK_STARTED+55-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=5s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+45
code=Path.cwd();rc,zero,error,counts=1,False,None,None
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='command_readout_provenance_CPU_clock_v1',phase_id='command-readout-provenance-cpu-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=120,status='started',attempts=1,all_old_CPU_balances_untouched=True,GPU_seconds=None)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 grant=root.parent/'GRANT.json';assert sha(grant)=='4d8e245be8a2333fa225e3a5e59859b9970cd006135ec2f35106172a94534871'
 write('allocation.json',dict(format='command_readout_provenance_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=120,scope='ONLY4provenance regressions; noold54/models/simulator/ledger'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_robotwin_readout_provenance.py','--junitxml='+str(root/'junit.xml')],cwd=code,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib)
 rc=0 if result.returncode==0 and counts['tests']=='4' and all(counts[k]=='0' for k in ('errors','failures','skipped')) else 1
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>120:rc,error=1,(error or '')+' Inclusive120CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,120-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='command_readout_provenance_CPU_qualification_v1',source_commit=source,passed=rc==0,JUnit=counts,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=120,GPU_seconds=0 if zero else None,new_actions=0,scope='ONLY4provenance cases; noVLA/physics/ledger',all_old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
PROVENANCE_CPU
