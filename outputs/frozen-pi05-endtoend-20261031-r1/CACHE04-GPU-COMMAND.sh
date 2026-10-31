bash <<'CACHE04_GPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=d4425999d4c4ef1c572cc28abe9003f012aaeedf
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-cache04-recovery-20261101-r2
TASK_HELPER=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_ALLOC=$TASK_PHASE/command-diagnostic-cache04-allocation-r1
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_ALLOC"
mkdir "$TASK_ALLOC"
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_HELPER/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
TASK_REMAINING=$((TASK_STARTED+530-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal,xml.etree.ElementTree as ET
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='d4425999d4c4ef1c572cc28abe9003f012aaeedf';lineage='fixed12-command-diagnostic/cache04'
run='command-diagnostic-cache04-r1';cache_parent=phase/'reactive-curve-v1/cache04';output=cache_parent/run
admission=phase/'admission/command-diagnostic-cache04-r1'
scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256=scope_sha,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def run_process(args,log,env,deadline):
 with Path(log).open('x') as stream:
  proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,cwd=code,env=env)
  try:return proc.wait(timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:
   proc.send_signal(signal.SIGTERM)
   try:proc.wait(timeout=8)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   raise TimeoutError('Original cache04 clock; preserve every partial/recovery and charge, no automatic retry')
attempt=begin(phase/'gpu-clock',contract,slice_seconds=540,worker_seconds=28800,resume=True,now=started,allow_short_slice=True,lineage=lineage,lineage_seconds=600)
rc,error=1,None
try:
 assert sha(phase/'SCOPE.json')==scope_sha
 authority=admission/'COMMAND-DIAGNOSTIC-CACHE04-CHILD-AUTHORITY.json';migration=admission/'CACHE04-CODE-MIGRATION.json'
 assert sha(authority)=='59322c918af9bd975945c70db37089327fe3badb56deae1aa2178e05172e5eb9'
 m=json.loads(migration.read_text());assert m['successor_source']==source and m['maximum_aggregate_GPU_seconds']==600 and m['cache04_lineage']==lineage
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-preservation-software-qual-20261101-r1/cpu-r1/cache-recovery.junit.xml')
 assert sha(cpu)=='58d2ad1e978b46b08e3957fc8fd3aef5326f2a01a91d58aa313bfae6605d0f17'
 suite=next(ET.parse(cpu).iter('testsuite'));assert suite.attrib['tests']=='33' and suite.attrib['failures']=='0' and suite.attrib['errors']=='0'
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert sum(r['charged_seconds'] for r in clock['attempts'] if r['status']=='finished')==3452
 rows=[r for r in clock['attempts'] if r.get('lineage')==lineage];assert len(rows)==1 and rows[0]['path']==str(attempt) and rows[0]['reserved_seconds']==600
 env=dict(os.environ,PYTHONPATH=str(code/'src')+':'+os.environ['REPO_PATH']+':'+os.environ['ROBOTWIN_PATH'])
 probe=allocation/'gpu-probe.json'
 assert run_process([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],allocation/'gpu-probe.log',env,min(started+120,started+410))==0
 env['LYCO_BAIGE_GPU_MAP']=str(probe)
 # All real bank assembly and array checks are inside THIS existing funded GPU clock.
 payload=code/'outputs/robotwin-command-diagnostic-20261101-r1/DATA-PAYLOAD.sh';inputs=payload.with_name('INPUTS.json');bank=allocation/'bank.json'
 data_env=dict(env,DATA_OPERATION='assemble',DATA_REQUEST=str(inputs),DATA_REQUEST_SHA=sha(inputs),DATA_OUTPUT=str(bank),OWNER_DATA_DEADLINE=str(started+410))
 assert run_process(['bash',str(payload)],allocation/'assemble.log',data_env,started+410)==0
 bank_value=json.loads(bank.read_text());assert len(bank_value['entries'])==12 and bank_value['outcome_evidence']['phone-04']['success'] is False and bank_value['source_eligible'] is False
 partial=phase/'partial11-cache-r1/frozen-pi05-partial11-cache-20261101-r1/partial-cache.json'
 cursor=partial.with_name('cursor.json');close=phase/'partial11-cache-allocation-r1/allocation-closeout.json'
 assert sha(partial)=='ebee3741b51099df9c41cdf42ed108058b447e5831a15b13285f201f1808c8ba'
 assert sha(cursor)=='0edbb0512094bd87ab3b0ddb746a2b4aa8bc2a66f689b9bd5ee44a9e3e3387d7' and sha(close)=='20195ec978bf5172cfafbf46575172cd7bc7b9fcf4e44d4f73efcc8e084d266e'
 request=json.loads((payload.with_name('CACHE04-REQUEST-TEMPLATE.json')).read_text());request.update(status='OWNER_BOUND_ALLOCATED_CACHE04_ONLY',bank=ref(bank),cache_receipts=[ref(partial)],partial11_cursor=ref(cursor),partial11_closeout=ref(close),cache04_resume=None)
 request['wide_configuration']['path']=str(code/request['wide_configuration']['path'])
 request_path=allocation/'request.json';write(request_path,request)
 # Pre-model numeric provenance/RNG admission; no backend/model construction here.
 preparation_code="import json,sys;from pathlib import Path;from fluffy_bassoon.meta_rl.robotwin_cache04_recovery import partial11_binding,motor_anchors;from fluffy_bassoon.meta_rl.robotwin_replay_diagnosis import Audit;r=json.loads(Path(sys.argv[1]).read_text());a=Audit(float(sys.argv[2]));b=a.json(r['bank']);v=dict(partial11=partial11_binding(r,b,deadline=a.deadline),motor_anchors=motor_anchors(b,deadline=a.deadline),model_loaded=False,robot_actions=0);Path(sys.argv[3]).write_text(json.dumps(v,indent=2)+chr(10))"
 assert run_process([sys.executable,'-c',preparation_code,str(request_path),str(started+410),str(allocation/'data-admission.json')],allocation/'data-admission.log',env,started+410)==0
 storage=admission/'STORAGE.json';cost=admission/'FACTORIAL-PARTIAL11-CLOSEOUT.json'
 plan=dict(format='owner_finite_cache04_resource_plan_v1',diagnostic_purpose='fixed12_original_plan_command_capacity_diagnostic_only',source_eligible=False,admitted=True,clock_root=str(phase/'gpu-clock'),clock_contract=contract,stage_gpu_limits=dict(cache04=600,fit=0),fit_head_gpu_limits=dict(action=0,kv=0),fit_topology='separate_one_gpu_physical_allocations',other_fit_ticket_receipts=[],prior_cost_receipts=[ref(cost)],storage_admission=ref(storage),code_migration=ref(migration),no_fit_or_readout_funding=True)
 plan_path=allocation/'resource-plan.json';write(plan_path,plan)
 ticket=dict(format=request['profile']['format'],admitted=True,operation='cache04',kind='cache04',source_commit=source,request_sha256=sha(request_path),hostname=socket.gethostname(),gpu_count=1,private_motor_context=True,maximum_robot_actions=0,output=str(output),phase_output_root=str(phase),clock_root=str(phase/'gpu-clock'),clock_contract=contract,resource_attempt=str(attempt),driver_deadline_unix=started+470,authority_receipt=ref(authority),resource_plan_receipt=ref(plan_path),new_recovery_CPU_ref=ref(cpu),migration_ref=ref(migration),reserved_gpu_seconds=600,started_unix=started)
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1');model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1')
 args=[sys.executable,'scripts/diagnostics/robotwin_command_diagnostic_cache04.py','--request',str(request_path),'--request-sha256',sha(request_path),'--ticket',str(ticket_path),'--ticket-sha256',sha(ticket_path),'--gpu-probe',str(probe),'--model-path',str(model),'--norm-stats',str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'--robotwin-root',str(native/'RoboTwin'),'--rlinf-root',str(native/'RLinf'),'--output',str(output),'--kind','cache04']
 write(allocation/'invocation.json',dict(source_commit=source,arguments=args,original_started_unix=started))
 assert run_process(args,allocation/'console.log',env,started+480)==0
 # Reuse paid11 by reference only; no re-encoding, copying or fitting.
 partial04=output/'partial-cache.json';seal=allocation/'seal-request.json';write(seal,dict(bank=ref(bank),cache_receipts=[ref(partial),ref(partial04)]))
 seal_env=dict(env,DATA_OPERATION='seal',DATA_REQUEST=str(seal),DATA_REQUEST_SHA=sha(seal),DATA_OUTPUT=str(allocation/'complete-diagnostic-cache.json'),OWNER_DATA_DEADLINE=str(started+510))
 assert run_process(['bash',str(payload)],allocation/'seal.log',seal_env,started+510)==0
 complete=json.loads((allocation/'complete-diagnostic-cache.json').read_text());assert complete['complete'] and complete['diagnostic12_fit_eligible'] and not complete['source_eligible'] and not complete['fixed12_fit_eligible']
 size=sum(p.stat().st_size for parent in (allocation,cache_parent) for p in parent.rglob('*') if p.is_file());assert size<=2147483648
 write(allocation/'completion.json',dict(source_commit=source,status='complete_diagnostic12_cache_only',complete_cache=ref(allocation/'complete-diagnostic-cache.json'),new04_cache=ref(partial04),new04_cursor=ref(output/'cursor.json'),paid11_unchanged=sha(partial)=='ebee3741b51099df9c41cdf42ed108058b447e5831a15b13285f201f1808c8ba',aggregate_new_output_bytes=size,new_robot_actions=0,failed04_still_false=complete['bank']['outcome_evidence']['phone-04']['success'] is False,source_eligible=False,fit_resource_admitted=False,readout_admitted=False))
 rc=0
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(allocation/'allocation-closeout.json',dict(source_commit=source,exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],cache_lineage=lineage,maximum_cache_lineage_seconds=600,maximum_new_actions=0,partial_orphans_preserved=True))
 (allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
CACHE04_GPU
