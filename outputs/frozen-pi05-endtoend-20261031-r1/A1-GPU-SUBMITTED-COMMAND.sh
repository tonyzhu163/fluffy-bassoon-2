bash <<'FROZEN_A1_GPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=8248b1cefe8757b5e43f2a1b6b503e5b93831659
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/frozen-pi05-endtoend-a1-20261031-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_CHILD=$TASK_PHASE/a1-r1
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
timeout --signal=TERM --kill-after=20s 7100 "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_CHILD" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,child,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='8248b1cefe8757b5e43f2a1b6b503e5b93831659'
scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
model='/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1'
norm=model+'/assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
native='/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256=scope_sha,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
attempt=begin(phase/'gpu-clock',contract,slice_seconds=7140,worker_seconds=28800,now=started)
rc=1;error=None
try:
 grant_path=phase/'GRANT.json';grant=json.loads(grant_path.read_text())
 assert sha(grant_path)=='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4'
 assert grant['approved'] and grant['phase_id']==contract['phase_id'] and grant['source_commit']==source
 assert grant['maximum_gpu_seconds']==28800 and grant['maximum_total_actions']==300000 and grant['maximum_output_bytes']==64*1024**3
 assert sha(phase/'SCOPE.json')==scope_sha
 assert sha(phase/'admission/cpu-summary.json')=='dd280f102abcf73969d618f6c226af9e6294e3c20440a36bbc8e0deacb7760f2'
 assert json.loads((phase/'admission/cpu-summary.json').read_text())['passed']
 assert json.loads((phase/'admission/cpu-summary.json').read_text())['source_commit']==source
 probe=child/'gpu-probe.json'
 with (child/'gpu-probe.log').open('x') as log:
  subprocess.run([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 os.environ['LYCO_BAIGE_GPU_MAP']=str(probe)
 source_caps=grant['canonical_action_accounting']['source_caps'];caps=grant['canonical_action_accounting']['phase_caps']
 assert len(source_caps)==6 and set(source_caps.values())=={19200}
 interface=phase/'admission/A1-INTERFACE.json';joints=phase/'admission/JOINT-CONVENTIONS.json'
 assert sha(interface)=='c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb' and sha(joints)=='8a0ff1e70fd776cf8cdb9e254966422e1dee8227584c388a262e62068ebf9bc4'
 interface_data=json.loads(interface.read_text())
 ledger=dict(root=str(phase/'action-budget'),campaign_id=contract['phase_id'],total_actions=300000,source_caps=source_caps,phase_caps=caps,contract=dict(phase_id=contract['phase_id'],scope_sha256=scope_sha,motor_sha256=interface_data['motor_sha256'],motor_parameter_sha256='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41',normalization_sha256='1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb',candidate_tasks=['place_phone_stand','place_object_stand']))
 ticket=dict(scope_sha256=scope_sha,source_commit=source,hostname=socket.gethostname(),gpu_count=1,output=str(child/'experiment'),phase_output_root=str(phase),driver_deadline_unix=started+6960,initial_episode_seconds=180,reserved_gpu_seconds=7200,prior_phase_gpu_seconds=0,other_reserved_phase_gpu_seconds=0,prior_a1_gpu_seconds=0,action_ledger=ledger,grant_sha256=sha(grant_path),cpu_summary_sha256=sha(phase/'admission/cpu-summary.json'),resource_attempt=str(attempt),mapping_predeclaration='fixed engineering bounds before robot outcomes; no scale sweep',expected_prior_motor_parameter_sha256='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41')
 ticket_path=child/'allocation.json';write(ticket_path,ticket)
 args=[sys.executable,'scripts/diagnostics/robotwin_steering_screen.py','--scope',str(phase/'SCOPE.json'),'--scope-sha256',scope_sha,'--allocation',str(ticket_path),'--allocation-sha256',sha(ticket_path),'--interface',str(interface),'--interface-sha256',sha(interface),'--joint-conventions',str(joints),'--joint-conventions-sha256',sha(joints),'--source-commit',source,'--run-id','frozen-pi05-endtoend-a1-20261031-r1','--output',str(child/'experiment'),'--model-path',model,'--norm-stats',norm,'--rlinf-root',native+'/RLinf','--robotwin-root',native+'/RoboTwin','--gpu-probe',str(probe)]
 write(child/'invocation.json',dict(arguments=args,started_unix=started,allocation_sha256=sha(ticket_path),interface_sha256=sha(interface),grant_sha256=sha(grant_path)))
 with (child/'console.log').open('x') as log:
  process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
  try:rc=process.wait(timeout=max(1,started+7040-time.time()))
  except subprocess.TimeoutExpired:
   process.send_signal(signal.SIGTERM)
   try:rc=process.wait(timeout=30)
   except subprocess.TimeoutExpired:process.kill();rc=process.wait()
   error='Driver exceeded original graceful deadline; partial raw/recovery retained, no automatic retry'
except BaseException:
 error=traceback.format_exc();rc=1
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(child/'allocation-closeout.json',dict(exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],phase_id=contract['phase_id'],source_commit=source))
 (child/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
FROZEN_A1_GPU_JOB
