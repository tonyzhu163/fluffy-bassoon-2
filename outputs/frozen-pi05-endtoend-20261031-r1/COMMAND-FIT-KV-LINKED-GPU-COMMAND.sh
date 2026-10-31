bash <<'COMMAND_FIT_GPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=3d35b3201a4d1131876c9251b6583464bac04982
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-command-fit-linked-20261101-r1
TASK_HELPER=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_ALLOC=$TASK_PHASE/command-diagnostic-fit-kv-linked-allocation-r1
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
TASK_REMAINING=$((TASK_STARTED+1730-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal,xml.etree.ElementTree as ET
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='3d35b3201a4d1131876c9251b6583464bac04982';kind='kv';lineage='fixed12-command-diagnostic/fit-'+kind+'-linked-v1'
scientific_source='400eed69ead8b8672686c62f33ed5531f6f60558'
run='command-diagnostic-fit-'+kind+'-linked-r1';parent=phase/'reactive-curve-v1'/kind;output=parent/run
admission=phase/'admission/command-fit-linked-r1'
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
   raise TimeoutError('Original1800-second head clock; preserve every complete recovery, no automatic retry')
attempt=begin(phase/'gpu-clock',contract,slice_seconds=1740,worker_seconds=28800,resume=True,now=started,lineage=lineage,lineage_seconds=1800)
rc,error=1,None
try:
 assert sha(phase/'SCOPE.json')==scope_sha
 authority=admission/'LINKED-AUTHORITY.json'
 assert sha(authority)=='2ca8a32f0dd57b274e957b39f421c04e83675f2676c52f002bf11c3ba905d751'
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/command-fit-linked-resource-cpu-20261101-r1/cpu-r1')
 summary=json.loads((cpu/'summary.json').read_text());assert summary['passed'] and summary['source_commit']==source
 assert json.loads((cpu/'predecessor-inspection.json').read_text())['passed']
 suite=next(ET.parse(cpu/'linked-fit.junit.xml').iter('testsuite'));assert suite.attrib['tests']=='29' and all(suite.attrib[k]=='0' for k in ('errors','failures','skipped'))
 cache_cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cache-readout-correction-cpu-20261101-r1/cpu-r1')
 complete_path=cache_cpu/'complete-diagnostic-cache.json';assert sha(complete_path)=='65c708bf3b2c85509ed6f35150768149569558def96dd981e5b548f10a673cf8'
 complete=json.loads(complete_path.read_text());canonical_sha=hashlib.sha256(json.dumps(complete,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert canonical_sha=='5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd'
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert sum(r['charged_seconds'] for r in clock['attempts'] if r['status']=='finished')>=5189
 env=dict(os.environ,PYTHONPATH=str(code/'src')+':'+os.environ['REPO_PATH']+':'+os.environ['ROBOTWIN_PATH'])
 probe=allocation/'gpu-probe.json'
 assert run_process([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],allocation/'gpu-probe.log',env,started+120)==0
 env['LYCO_BAIGE_GPU_MAP']=str(probe)
 op=json.loads(authority.read_text());previous=op['predecessors'][kind]
 assert sha(previous['request']['path'])==previous['request']['sha256']=='9514b69420a1389eb76a377d2c44a32fc2bbc7e14f8d708a5cd6c9382ef2341d'
 request=json.loads(Path(previous['request']['path']).read_text())
 progress=json.loads(Path(previous['progress']['path']).read_text());assert sha(previous['progress']['path'])==previous['progress']['sha256']
 request['resume']={k:progress['recovery'][k] for k in ('path','sha256')}
 request['runtime_blob_sha256']=json.loads((code/'outputs/robotwin-command-fit-linked-20261101-r1/EXECUTION-RUNTIME-BLOBS.json').read_text())
 request['linked_fit_continuation']=ref(authority)
 request_path=allocation/'request.json';write(request_path,request)
 others=[]
 if kind=='kv':
  marker=phase/'reactive-curve-v1/action/command-diagnostic-fit-action-linked-r1-resource-admission.json';m=json.loads(marker.read_text());assert m['status']=='resource_ticket_admitted_before_model_load' and m['kind']=='action' and m['source_commit']==source
  others=[m['ticket']];assert sha(others[0]['path'])==others[0]['sha256']
 storage=admission/'STORAGE.json';cost=admission/'closed-parent-clock-5189.json'
 plan=dict(format='owner_finite_command_fit_resource_plan_v1',diagnostic_purpose='fixed12_original_plan_command_capacity_diagnostic_only',source_eligible=False,admitted=True,clock_root=str(phase/'gpu-clock'),clock_contract=contract,stage_gpu_limits=dict(cache04=600,fit=3600),fit_head_gpu_limits=dict(action=1800,kv=1800),fit_topology='separate_one_gpu_physical_allocations',other_fit_ticket_receipts=others,prior_cost_receipts=[ref(cost)],linked_fit_continuation=ref(authority),storage_admission=ref(storage),complete_cache_file=ref(complete_path),required_CPU_evidence=ref(cpu/'summary.json'))
 plan_path=allocation/'resource-plan.json';write(plan_path,plan)
 ticket=dict(format=request['profile']['format'],admitted=True,operation='fit',kind=kind,source_commit=source,request_sha256=sha(request_path),hostname=socket.gethostname(),gpu_count=1,private_motor_context=True,maximum_robot_actions=0,output=str(output),phase_output_root=str(phase),clock_root=str(phase/'gpu-clock'),clock_contract=contract,resource_attempt=str(attempt),driver_deadline_unix=started+1680,authority_receipt=ref(authority),resource_plan_receipt=ref(plan_path),reserved_gpu_seconds=1800,started_unix=started)
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1');model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1')
 args=[sys.executable,'scripts/diagnostics/robotwin_command_diagnostic_curve.py','--request',str(request_path),'--request-sha256',sha(request_path),'--ticket',str(ticket_path),'--ticket-sha256',sha(ticket_path),'--gpu-probe',str(probe),'--model-path',str(model),'--norm-stats',str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'--robotwin-root',str(native/'RoboTwin'),'--rlinf-root',str(native/'RLinf'),'--output',str(output),'--kind',kind]
 write(allocation/'invocation.json',dict(source_commit=source,arguments=args,original_started_unix=started,complete_cache_file=ref(complete_path),complete_cache_canonical_sha256=canonical_sha))
 rc=run_process(args,allocation/'console.log',env,started+1700)
 if rc==0:
  result_path=output/'result.json';result=json.loads(result_path.read_text())
  assert result['complete'] and result['iteration']==5000 and result['contract']['source_commit']==scientific_source
  assert set(result['milestones'])=={'0','100','500','2000','5000'} and result['parity']['actual_cached_rows']==90 and result['parity']['all_native_base_bits_equal']
  write(allocation/'completion.json',dict(execution_source_commit=source,scientific_source_commit=scientific_source,iteration=5000,complete=True,result=ref(result_path),full_recovery=json.loads((output/'progress.json').read_text())['recovery'],primary_endpoint=result['primary_endpoint'],producer_actual_after_parameter_signature_checked='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41',signature_method='Successful source _qualified_curve postfit frozen_fingerprint comparison, before complete result publication',new_robot_actions=0,new_source_eligible=False))
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(allocation/'allocation-closeout.json',dict(source_commit=source,kind=kind,exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],fit_lineage=lineage,maximum_head_lineage_seconds=1800,maximum_new_actions=0,all_full_recoveries_and_prefixes_preserved=True))
 (allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
COMMAND_FIT_GPU
