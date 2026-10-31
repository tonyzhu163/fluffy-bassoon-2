bash <<'COMMAND_FIT_GPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=400eed69ead8b8672686c62f33ed5531f6f60558
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-cache-state-abi-cpu-20261101-r1
TASK_HELPER=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_ALLOC=$TASK_PHASE/command-diagnostic-fit-action-allocation-r1
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
TASK_REMAINING=$((TASK_STARTED+820-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal,xml.etree.ElementTree as ET
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='400eed69ead8b8672686c62f33ed5531f6f60558';kind='action';lineage='fixed12-command-diagnostic/fit-'+kind
run='command-diagnostic-fit-'+kind+'-r1';parent=phase/'reactive-curve-v1'/kind;output=parent/run
admission=phase/'admission/command-diagnostic-fits-r1'
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
   raise TimeoutError('Original900-second head clock; preserve every complete recovery, no automatic retry')
attempt=begin(phase/'gpu-clock',contract,slice_seconds=840,worker_seconds=28800,resume=True,now=started,lineage=lineage,lineage_seconds=900)
rc,error=1,None
try:
 assert sha(phase/'SCOPE.json')==scope_sha
 authority=admission/'COMMAND-DIAGNOSTIC-FIT-CHILD-AUTHORITY.json'
 assert sha(authority)=='4f649dd1394ca9f5359c28f37a48c6bc5a0449024cc4bca4630a382ce2aafb3d'
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cache-readout-correction-cpu-20261101-r1/cpu-r1')
 summary=json.loads((cpu/'summary.json').read_text());assert summary['passed'] and summary['components']['cache-tests']['return_code']==0 and summary['components']['seal']['return_code']==0
 suite=next(ET.parse(cpu/'cache-tests.junit.xml').iter('testsuite'));assert suite.attrib['tests']=='12' and suite.attrib['failures']=='0' and suite.attrib['errors']=='0'
 complete_path=cpu/'complete-diagnostic-cache.json';assert sha(complete_path)=='65c708bf3b2c85509ed6f35150768149569558def96dd981e5b548f10a673cf8'
 complete=json.loads(complete_path.read_text());canonical_sha=hashlib.sha256(json.dumps(complete,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert canonical_sha=='5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd'
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert sum(r['charged_seconds'] for r in clock['attempts'] if r['status']=='finished')>=3625
 env=dict(os.environ,PYTHONPATH=str(code/'src')+':'+os.environ['REPO_PATH']+':'+os.environ['ROBOTWIN_PATH'])
 probe=allocation/'gpu-probe.json'
 assert run_process([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],allocation/'gpu-probe.log',env,started+120)==0
 env['LYCO_BAIGE_GPU_MAP']=str(probe)
 request=json.loads((code/'outputs/robotwin-command-diagnostic-20261101-r1/REQUEST-TEMPLATE.json').read_text())
 bank=phase/'command-diagnostic-cache04-allocation-r1/bank.json';assert sha(bank)=='5087dc5a230cb5bb9c635c5bc0f75bddcf2d15a7b080674fa4ee8a653f38cc0e'
 request.update(status='OWNER_BOUND_MODEL_ONLY_FIT',bank=ref(bank),cache_receipts=complete['cache_receipts'],complete_cache_sha256=canonical_sha,resume=None)
 request['wide_configuration']['path']=str(code/request['wide_configuration']['path'])
 request_path=allocation/'request.json';write(request_path,request)
 others=[]
 if kind=='kv':
  marker=phase/'reactive-curve-v1/action/command-diagnostic-fit-action-r1-resource-admission.json';m=json.loads(marker.read_text());assert m['status']=='resource_ticket_admitted_before_model_load' and m['kind']=='action' and m['source_commit']==source
  others=[m['ticket']];assert sha(others[0]['path'])==others[0]['sha256']
 storage=admission/'STORAGE.json';cost=phase/'command-diagnostic-cache04-allocation-r1/allocation-closeout.json'
 plan=dict(format='owner_finite_command_fit_resource_plan_v1',diagnostic_purpose='fixed12_original_plan_command_capacity_diagnostic_only',source_eligible=False,admitted=True,clock_root=str(phase/'gpu-clock'),clock_contract=contract,stage_gpu_limits=dict(cache04=600,fit=1800),fit_head_gpu_limits=dict(action=900,kv=900),fit_topology='separate_one_gpu_physical_allocations',other_fit_ticket_receipts=others,prior_cost_receipts=[ref(cost),ref(admission/'PRIOR-COSTS.json')],storage_admission=ref(storage),complete_cache_file=ref(complete_path),required_CPU_evidence=ref(cpu/'summary.json'))
 plan_path=allocation/'resource-plan.json';write(plan_path,plan)
 ticket=dict(format=request['profile']['format'],admitted=True,operation='fit',kind=kind,source_commit=source,request_sha256=sha(request_path),hostname=socket.gethostname(),gpu_count=1,private_motor_context=True,maximum_robot_actions=0,output=str(output),phase_output_root=str(phase),clock_root=str(phase/'gpu-clock'),clock_contract=contract,resource_attempt=str(attempt),driver_deadline_unix=started+780,authority_receipt=ref(authority),resource_plan_receipt=ref(plan_path),reserved_gpu_seconds=900,started_unix=started)
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1');model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1')
 args=[sys.executable,'scripts/diagnostics/robotwin_command_diagnostic_curve.py','--request',str(request_path),'--request-sha256',sha(request_path),'--ticket',str(ticket_path),'--ticket-sha256',sha(ticket_path),'--gpu-probe',str(probe),'--model-path',str(model),'--norm-stats',str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'--robotwin-root',str(native/'RoboTwin'),'--rlinf-root',str(native/'RLinf'),'--output',str(output),'--kind',kind]
 write(allocation/'invocation.json',dict(source_commit=source,arguments=args,original_started_unix=started,complete_cache_file=ref(complete_path),complete_cache_canonical_sha256=canonical_sha))
 rc=run_process(args,allocation/'console.log',env,started+800)
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(allocation/'allocation-closeout.json',dict(source_commit=source,kind=kind,exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],fit_lineage=lineage,maximum_head_lineage_seconds=900,maximum_new_actions=0,all_full_recoveries_and_prefixes_preserved=True))
 (allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
COMMAND_FIT_GPU
