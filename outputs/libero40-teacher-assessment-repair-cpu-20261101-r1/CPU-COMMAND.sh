bash <<'ASSESSMENT_REPAIR_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=9d29a805cae2c129ffc088af4eed81b3a38d2769
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/libero40-teacher-assessment-repair-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-repair-cpu-20261101-r1/cpu-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir -p "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export PYTHONPATH=$TASK_CODE/src:/mowenjie/fluffy-bassoon-2/envs/pytest-overlay-20261029-r1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
TASK_REMAINING=$((TASK_STARTED+52-$(date +%s)))
test "$TASK_REMAINING" -gt 0
set +e
timeout --signal=INT --kill-after=3s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+45;code=Path.cwd()
rc,zero,error,counts=1,False,None,None
selectors=['tests/test_vla_source40_teacher_repair.py','tests/test_vla_source40_teacher_assessment_runtime.py::test_exact_public_pair_refusal_happens_before_next_steering_action','tests/test_vla_source40_teacher_assessment_runtime.py::test_actual_feature_value_drift_fails_even_when_all_hash_labels_match','tests/test_vla_source40_teacher_assessment_runtime.py::test_closed_resume_refuses_fresh_sibling_clock_before_server_or_reset','tests/test_vla_source40_teacher_assessment_runtime.py::test_closed_resume_does_not_replay_controls_or_qualification']
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='assessment_repair_CPU_clock_v1',source_commit=source,started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=120,status='started',attempts=1,interrupted_charge_CPU_worker_seconds=120,all_old_CPU_balances_untouched=True)
write('cpu-worker-clock.json',clock)
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-') and sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count();zero=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective'];raw={n:Path(n).read_text().strip() for n in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,raw_cgroup=raw,actual_CPU_cores=cores,actual_memory_bytes=memory,scheduler_GPU_visibility_preserved=True))
 assert zero and cores==2 and memory==16*1024**3 and str(torch.__version__).startswith('2.11.')
 grant=code/'outputs/libero40-teacher-assessment-repair-cpu-20261101-r1/GRANT.json';assert sha(grant)=='22dd73a9b75873e1302e6ffd99e73d2e5d2a803bbf88e222c4cd5c1f4d3a8e42'
 assert all((code/p.split('::')[0]).is_file() for p in selectors)
 write('allocation.json',dict(format='all40_teacher_assessment_CPU_ticket_v1',source_commit=source,hostname=socket.gethostname(),GPU_count=0,actual_CPU_cores=cores,actual_memory_bytes=memory,grant_ref=dict(path=str(grant),sha256=sha(grant)),zero_gpu_attestation_sha256=sha(root/'zero-gpu-attestation.json'),started_unix=started,absolute_deadline_unix=deadline,maximum_CPU_worker_seconds=120,tests=selectors,scope='ONLY26new+4affected cases then exact62745B progress dryrun; no actorweights/VLA/JAXconstructor/simulator/SGD/physics, no59repeat; original154of180untouched'))
 with (root/'pytest.log').open('x') as log:
  result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',*selectors,'--junitxml='+str(root/'junit.xml')],cwd=code,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=max(1,deadline-time.time()),check=False)
 suite=next(ET.parse(root/'junit.xml').iter('testsuite'));counts=dict(suite.attrib)
 assert result.returncode==0 and counts['tests']=='30' and all(counts[k]=='0' for k in ('errors','failures','skipped'))
 packet=code/'outputs/libero40-teacher-assessment-repair-code-20261101-r1'
 assert sha(packet/'PROFILE.json')=='222e98b6dd8762944d504b9af033cbda52b7b1d1e3625b8c0b12fda3ceb508a0'
 env=dict(os.environ,REPAIR_CPU_PYTHON=sys.executable,REPAIR_TINY_DRYRUN_JSON=str(root/'tiny-dryrun.json'))
 with (root/'tiny-dryrun.log').open('x') as log:
  dryresult=subprocess.run(['bash',str(packet/'TINY-DRYRUN-COMMAND.sh')],cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.time()),check=False)
 assert dryresult.returncode==0
 tiny=json.loads((root/'tiny-dryrun.json').read_text());assert tiny['numerical_before']==tiny['numerical_after'] and tiny['old_receipts']==tiny['new_receipts']==161 and tiny['status']=='TINY_METADATA_MIGRATION_DRYRUN_PASS_NOT_ADMISSION'
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>120:rc,error=1,(error or '')+' Inclusive120CPU cap exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_CPU_worker_seconds=charged,remaining_CPU_worker_seconds=max(0,120-charged),GPU_seconds=0 if zero else None);write('cpu-worker-clock.json',clock)
 write('summary.json',dict(evidence={k:dict(path=str(root/v),sha256=sha(root/v)) for k,v in [('junit','junit.xml'),('attestation','zero-gpu-attestation.json'),('cost','cpu-worker-clock.json'),('tiny_dryrun','tiny-dryrun.json')] if (root/v).is_file()},format='assessment_repair_CPU_qualification_v1',source_commit=source,passed=rc==0,JUnit=counts,exit_code=rc,error=error,charged_CPU_worker_seconds=charged,maximum_CPU_worker_seconds=120,GPU_seconds=0 if zero else None,new_actions=0,new_SGD=0,real_actor_checkpoint_loaded=False,tiny_progress_loaded=(root/'tiny-dryrun.json').is_file(),JAX_constructor=False,VLA_constructed=False,simulator_constructed=False,actual_GPU_runtime_proof=False,all_old_CPU_balances_untouched=True));(root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TASK_EXIT=$?
test -f "$TASK_OUT/exit-code.txt" || echo "$TASK_EXIT" > "$TASK_OUT/exit-code.txt"
exit "$TASK_EXIT"
ASSESSMENT_REPAIR_CPU
