bash <<'COMMAND_READOUT_GPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=5470090165f89a114afe544758ab89d5d956fe76
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-readout-provenance-20261101-r1
TASK_HELPER=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_ALLOC=$TASK_PHASE/command-diagnostic-readout-r2
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
export FB2_ROBOTWIN_RENDER_RECORDS="$TASK_ALLOC/renderer-bindings"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
 ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
TASK_REMAINING=$((TASK_STARTED+1355-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal,xml.etree.ElementTree as ET
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='5470090165f89a114afe544758ab89d5d956fe76';lineage='fixed12-command-diagnostic/readout';run='frozen-pi05-command-readout-20261101-r2';output=allocation/'experiment';admission=phase/'admission/command-readout-r2'
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447',maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def run_process(args,log,env,deadline,cwd=code):
 with Path(log).open('x') as stream:
  proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,cwd=cwd,env=env,start_new_session=True)
  write(allocation/(Path(log).stem+'-process.json'),dict(pid=proc.pid,parent_pid=os.getpid(),hostname=socket.gethostname(),source_commit=source))
  try:return proc.wait(timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise TimeoutError('Original1500-second readoutclock; preserve allpartial/cursor/physicalintents, no automatic retry')
  finally:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   else:
    until=time.monotonic()+2
    while time.monotonic()<until:
     try:os.killpg(proc.pid,0)
     except ProcessLookupError:break
     time.sleep(0.05)
    else:
     try:os.killpg(proc.pid,signal.SIGKILL)
     except ProcessLookupError:pass
prior_clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert all(r['status']=='finished' for r in prior_clock['attempts']) and sum(r['charged_seconds'] for r in prior_clock['attempts'])==7057
attempt=begin(phase/'gpu-clock',contract,slice_seconds=1365,worker_seconds=28800,resume=True,now=started,lineage=lineage,lineage_seconds=1500)
rc,error=1,None
try:
 authority=admission/'COMMAND-DIAGNOSTIC-READOUT-CHILD-AUTHORITY.json';assert sha(authority)=='4c2e2d331b26076a143a6d1b5eb2fddece637830013ae1229f816ad118729ef0'
 assert sha(admission/'CODE-MIGRATION.json')=='06ff8ff1069ed332bb6a94d44d67f20a93f3164b3dd5ff658920b0a4bb32ec89'
 assert sha(admission/'STORAGE.json')=='f45b47e59444bd7461a7f8365ffae2f72463a06a4a885c3cb815f84856acf66a'
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/command-readout-provenance-cpu-20261101-r1/cpu-r1');summary=json.loads((cpu/'summary.json').read_text());assert sha(cpu/'summary.json')=='10b944b9012c7da24d7125b6366370b533d00c2e7bb9072f1b1ac7519367f864';assert summary['passed'] and summary['source_commit']==source
 suite=next(ET.parse(cpu/'junit.xml').iter('testsuite'));assert suite.attrib['tests']=='4' and all(suite.attrib[k]=='0' for k in ('errors','failures','skipped'))
 old_cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/command-readout-linked-resource-cpu-20261101-r1/cpu-r1/summary.json');assert json.loads(old_cpu.read_text())['required_passed']
 env=dict(os.environ,PYTHONPATH=str(code/'src')+':'+os.environ['REPO_PATH']+':'+os.environ['ROBOTWIN_PATH'])
 probe=allocation/'gpu-probe.json';assert run_process([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],allocation/'gpu-probe.log',env,started+120)==0;env['LYCO_BAIGE_GPU_MAP']=str(probe)
 # Reuse the immutable canonical DATA receipt already produced in the paid failed attempt.
 request_input=admission/'REQUEST-INPUT.json';assert sha(request_input)=='8ab606e416d40a38c79adfe3b6331fec497159c6ff74f9d7e9f6043bc70188a4'
 request=json.loads(request_input.read_text());assert request['resume'] is None
 assert sha(request['canonical_bindings']['path'])==request['canonical_bindings']['sha256']=='e8c49c56d55853f14d0e2fb3a753c6395a1e3267a1097f4ae3b7c2836423a2a9'
 request_path=allocation/'request.json';write(request_path,request)
 for k,result_ref in request['fit_results'].items():
  assert sha(result_ref['path'])==result_ref['sha256'];result=json.loads(Path(result_ref['path']).read_text());assert result['complete'] and result['iteration']==5000 and result['contract']['source_commit']=='400eed69ead8b8672686c62f33ed5531f6f60558'
  assert result['contract']['complete_cache_sha256']=='5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd' and sha(result['primary_endpoint']['path'])==result['primary_endpoint']['sha256']
 plan_path=admission/'RESOURCE-PLAN-INPUT.json';assert sha(plan_path)=='8de9111d8e887ea47651b9c0dcf9ade469898b23326237f8c8c441707fe12cdb'
 head=json.loads(Path(request['current_head']['path']).read_text());assert head['actions']['actual_actions']==head['actions']['charged_actions']==23922 and head.get('pending') is None
 ticket=dict(format=request['format'],admitted=True,source_commit=source,request_sha256=sha(request_path),hostname=socket.gethostname(),gpu_count=1,maximum_robot_actions=6000,run_id=run,output=str(output),clock_root=str(phase/'gpu-clock'),clock_contract=contract,resource_attempt=str(attempt),resource_plan_receipt=ref(plan_path),authority_receipt=ref(authority),predecessor_terminal_receipt=ref(admission/'FACTORIAL-TERMINAL-RECEIPT.json'),phase_output_root=str(phase),action_ledger=head['action_ledger'],driver_deadline_unix=started+1305,cleanup_seconds=60,minimum_episode_admission_seconds=90,aggregate_output_root=str(allocation),maximum_output_bytes=8*1024**3,code_migration=ref(admission/'CODE-MIGRATION.json'),required_CPU_ref=ref(cpu/'summary.json'),carried_resource_CPU_ref=ref(old_cpu),prior_failed_readout_closeout=ref(phase/'command-diagnostic-readout-r1/allocation-closeout.json'),prior_actual_global_GPU_seconds=7057)
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1');model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1')
 args=[sys.executable,'scripts/diagnostics/robotwin_command_diagnostic_readout.py','--request',str(request_path),'--request-sha256',sha(request_path),'--ticket',str(ticket_path),'--ticket-sha256',sha(ticket_path),'--gpu-probe',str(probe),'--model-path',str(model),'--norm-stats',str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'--robotwin-root',str(native/'RoboTwin'),'--rlinf-root',str(native/'RLinf'),'--output',str(output)]
 write(allocation/'invocation.json',dict(source_commit=source,scientific_reference='235fcc00f0ddf7a07590604daba9e79c9a4c3c6d',arguments=args,original_started_unix=started,maximum_GPU_seconds=1425,whole_readout_lineage_GPU_seconds=1500,prior_failed_GPU_seconds=75,maximum_actions=6000,maximum_episodes=15))
 rc=run_process(args,allocation/'console.log',env,started+1325)
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(allocation/'allocation-closeout.json',dict(source_commit=source,exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],readout_lineage=lineage,maximum_readout_GPU_seconds=1500,maximum_new_actions=6000,all_full_recoveries_and_raw_preserved=True))
 (allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
COMMAND_READOUT_GPU
