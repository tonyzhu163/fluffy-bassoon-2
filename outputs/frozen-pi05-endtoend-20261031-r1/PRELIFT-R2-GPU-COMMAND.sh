bash <<'FROZEN_PRELIFT_GPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=374ee8958f7cabcd54d4401057a604c6cc6ae51a
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-prelift-probe-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_CHILD=$TASK_PHASE/prelift-pair-r2
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_CHILD"
mkdir "$TASK_CHILD"
export PATH="$(dirname "$TASK_PYTHON"):$PATH"
export PYTHONPATH="$TASK_CODE/src:$TASK_RLINF:$TASK_ROBOTWIN"
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 FLUFFY_GPU_EXECUTION=baige
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mowenjie/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/inputs/rlinf-openpi-tokenizer-befaa248-20261028-r1
export ROBOT_PLATFORM=ALOHA ASSETS_PATH="$TASK_ROBOTWIN" ROBOTWIN_PATH="$TASK_ROBOTWIN"
export EMBODIED_PATH="$TASK_RLINF/examples/embodiment" REPO_PATH="$TASK_RLINF"
export FB2_ROBOTWIN_RENDER_RECORDS="$TASK_CHILD/renderer-bindings"
mkdir -p /root/.sapien/physx
if [ ! -e /root/.sapien/physx/105.1-physx-5.3.1.patch0 ]; then
  ln -s /mowenjie/fluffy-bassoon/inputs/maniskill-scene-assets-20261028-r1/physx/105.1-physx-5.3.1.patch0 /root/.sapien/physx/105.1-physx-5.3.1.patch0
fi
timeout --signal=TERM --kill-after=5s 417 "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_CHILD" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
helper=Path('/mowenjie/fluffy-bassoon-2/worktrees/robotwin-prelift-resource-r2-20261101')
assert subprocess.check_output(['git','-C',str(helper),'rev-parse','HEAD'],text=True).strip()=='6397b1dc005c08389e511db19a115c273580fbb6'
assert not subprocess.check_output(['git','-C',str(helper),'status','--porcelain'],text=True).strip()
sys.path.insert(0,str(helper/'src'))
from fluffy_bassoon.meta_rl import vla_source40_ppo_worker as resource_worker
assert Path(resource_worker.__file__).resolve()==helper/'src/fluffy_bassoon/meta_rl/vla_source40_ppo_worker.py'
begin,finish=resource_worker.begin,resource_worker.finish
sys.path.pop(0)
phase,child,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='374ee8958f7cabcd54d4401057a604c6cc6ae51a'
run='frozen-pi05-prelift-pair-20261101-r2'
scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
model='/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1'
norm=model+'/assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
native='/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1'
admission=phase/'admission/prelift-pair-r2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256=scope_sha,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
# SAME600-second child: failed103 retained +retry497 (=437slice+60); B historical2652 unchanged.
attempt=begin(phase/'gpu-clock',contract,slice_seconds=437,worker_seconds=28800,resume=True,now=started,allow_short_slice=True)
rc=1;error=None
try:
 grant_path=phase/'GRANT.json';grant=json.loads(grant_path.read_text())
 assert sha(grant_path)=='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4'
 assert grant['approved'] and grant['maximum_gpu_seconds']==28800 and grant['maximum_total_actions']==300000 and grant['maximum_output_bytes']==64*1024**3
 assert sha(phase/'SCOPE.json')==scope_sha
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-prelift-cpu-20261101-r1/summary.json');authority_path=admission/'PRELIFT-R2-CHILD-AUTHORITY.json'
 assert sha(cpu)=='953a5332b9f92721e76adfaae4cee9433b368a2542b4412800c94f96043e6a92'
 assert json.loads(cpu.read_text())['passed']
 resource_cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-prelift-resource-cpu-r2-20261101/summary.json')
 assert sha(resource_cpu)=='62a67a11a3105bd509b7ea4db749cc0d25551a56a7960d534bd45dece61d4779' and json.loads(resource_cpu.read_text())['passed']
 assert sha(cpu.with_name('preflight04.json'))=='a1ee5e4f524055eecbde733e6afb63e1aa9a3883f0795043868e902ff068fa7a'
 assert sha(authority_path)=='a42a6e90a6e712c25d4ede27abf4b55d817be883f9a2c9415118dd0641e00f61'
 assert json.loads(authority_path.read_text())['maximum_new_GPU_seconds']==497
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert len(clock['attempts'])==7 and [r.get('charged_seconds') for r in clock['attempts'][:6]]==[632,1138,266,298,318,103] and clock['attempts'][-1]['reserved_seconds']==497
 probe=child/'gpu-probe.json'
 with (child/'gpu-probe.log').open('x') as log:
  subprocess.run([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 os.environ['LYCO_BAIGE_GPU_MAP']=str(probe)
 interface=phase/'admission/A1-INTERFACE.json';joints=phase/'admission/JOINT-CONVENTIONS.json';demos=phase/'admission/capacity-r1/demos.json'
 assert sha(interface)=='c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb'
 assert sha(joints)=='8a0ff1e70fd776cf8cdb9e254966422e1dee8227584c388a262e62068ebf9bc4'
 assert sha(demos)=='a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4'
 inputs=Path('/mowenjie/fluffy-bassoon-2/inputs/robotwin-prelift-closure-probe-20261101-r1');profile=inputs/'PROFILE.json';context=inputs/'EXECUTION-CONTEXT.json'
 assert sha(profile)=='0072b8c2c8239fb28d2cee7999f6614c770cd4c11ade6404c6fd285751a77ba1'
 assert sha(context)=='50fd8a601de9239be64ba1eb0013dfdd9d6ea25719dbe0456aab799fb622e655'
 head=phase/'event-suffix-r1/experiment/result.json';assert sha(head)=='73bd6483cd5abf8093fc46d8eb5fb116b745d2010e1e922b9f7d658d4625cea6'
 ledger=json.loads(head.read_text())['action_ledger']
 assert ledger['source_caps']==grant['canonical_action_accounting']['source_caps'] and ledger['phase_caps']==grant['canonical_action_accounting']['phase_caps']
 ticket=dict(scope_sha256=scope_sha,source_commit=source,hostname=socket.gethostname(),gpu_count=1,output=str(child/'experiment'),phase_output_root=str(phase),driver_deadline_unix=started+397,reserved_gpu_seconds=600,prior_phase_gpu_seconds=2652,other_reserved_phase_gpu_seconds=0,prior_a1_gpu_seconds=2652,action_ledger=ledger,grant_sha256=sha(grant_path),cpu_summary_sha256=sha(cpu),resource_attempt=str(attempt),child_authority_sha256=sha(authority_path),expected_prior_motor_parameter_sha256='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41',actual_current_attempt_reserved_gpu_seconds=497,failed_child_gpu_seconds=103,current_actual_prior_phase_gpu_seconds=2755,current_actual_prior_a1_gpu_seconds=2755,reservation_scope='Original whole-child600 includes failed103; current parent clock/deadline only497. Historical prior2652 binds immutable B head.',resource_helper_source='6397b1dc005c08389e511db19a115c273580fbb6',mode='robotwin_phone04_prelift_closure_pair_v1',run_id=run,capacity_action_bound=800,scheduler_start_unix=started)
 ticket_path=child/'allocation.json';write(ticket_path,ticket)
 args=[sys.executable,'scripts/diagnostics/robotwin_prelift_probe.py','--scope',str(phase/'SCOPE.json'),'--scope-sha256',scope_sha,'--allocation',str(ticket_path),'--allocation-sha256',sha(ticket_path),'--interface',str(interface),'--interface-sha256',sha(interface),'--joint-conventions',str(joints),'--joint-conventions-sha256',sha(joints),'--demos',str(demos),'--demos-sha256',sha(demos),'--profile',str(profile),'--profile-sha256',sha(profile),'--execution-context',str(context),'--execution-context-sha256',sha(context),'--source-commit',source,'--run-id',run,'--output',str(child/'experiment'),'--model-path',model,'--norm-stats',norm,'--rlinf-root',native+'/RLinf','--robotwin-root',native+'/RoboTwin','--gpu-probe',str(probe)]
 write(child/'invocation.json',dict(arguments=args,started_unix=started,allocation_sha256=sha(ticket_path),interface_sha256=sha(interface),grant_sha256=sha(grant_path)))
 with (child/'console.log').open('x') as log:
  process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
  write(child/'driver-process.json',dict(pid=process.pid,parent_pid=os.getpid(),hostname=socket.gethostname(),source_commit=source))
  try:rc=process.wait(timeout=max(1,started+397-time.time()))
  except subprocess.TimeoutExpired:
   process.send_signal(signal.SIGTERM)
   try:rc=process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();rc=process.wait()
   error='Driver exceeded original graceful deadline; partial raw/recovery retained, no automatic retry'
except BaseException:
 error=traceback.format_exc();rc=1
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(child/'allocation-closeout.json',dict(exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],whole_child_cumulative_gpu_seconds=103+row['charged_seconds'],phase_id=contract['phase_id'],source_commit=source))
 (child/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
FROZEN_PRELIFT_GPU_JOB
