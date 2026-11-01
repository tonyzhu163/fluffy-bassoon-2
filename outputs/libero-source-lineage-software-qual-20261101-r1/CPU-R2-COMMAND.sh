bash <<'LINEAGE_SYNTHETIC_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=87fa769beb9d926e9d0216550c6f9286848df6f9
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/libero-source-lineage-software-qual-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/libero-source-lineage-software-qual-20261101-r1/cpu-r2
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export PYTHONPATH=$TASK_CODE/src:/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+38-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=4s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+33
code=Path.cwd();rc,zero,error,counts=1,False,None,None
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='source_lineage_software_CPU_clock_v1',phase_id='libero-source-lineage-software-qual-20261101-r1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=105,prior_CPU_worker_seconds=75,whole_phase_CPU_worker_seconds=180,status='started',attempts=1,all_old_CPU_balances_untouched=True,GPU_seconds=None)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 names=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in names};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 import jax,flax
 assert all(d.platform=='cpu' for d in jax.devices()) and not torch.cuda.is_initialized()
 write('JAX-CPU-runtime.json',dict(jax_version=jax.__version__,jax_file=jax.__file__,flax_version=flax.__version__,flax_file=flax.__file__,devices=[str(d) for d in jax.devices()],VLA_constructed=False,real_teacher_constructed=False,SGD=0,robot_actions=0))
 prior=json.loads((root.parent/'cpu-r1/cpu-worker-clock.json').read_text());assert prior['status']=='finished' and prior['charged_CPU_worker_seconds']==75
 grant=root.parent/'GRANT.json';assert sha(grant)=='68e1b72992b8fc61e3b3871c35da7a38032ac04d1033a21f8e1225323cc0fa77'
 write('allocation.json',dict(format='source_lineage_software_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=105,prior_CPU_worker_seconds=75,whole_phase_CPU_worker_seconds=180,scope='ONLY32adapter+17producer syntheticcases; no realteacher/VLA/checkpoint/SGD/sim/features/originaldata assembly'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_vla_source40_lineage_adapter.py','tests/test_vla_source40_compact_evidence.py','--junitxml='+str(root/'junit.xml')],cwd=code,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib)
 rc=0 if result.returncode==0 and counts['tests']=='49' and all(counts[k]=='0' for k in ('errors','failures','skipped')) else 1
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>105:rc,error=1,(error or '')+' Inclusive remaining105CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,105-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='source_lineage_software_CPU_qualification_v1',source_commit=source,passed=rc==0,all_passed=rc==0,JUnit=counts,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=105,prior_CPU_worker_seconds=75,whole_phase_CPU_worker_seconds=180,GPU_seconds=0 if zero else None,new_actions=0,scope='ONLY49newsyntheticcases; SOFTWARE ONLY, no actual source/runtime/collection/fit admission',all_old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
LINEAGE_SYNTHETIC_CPU
