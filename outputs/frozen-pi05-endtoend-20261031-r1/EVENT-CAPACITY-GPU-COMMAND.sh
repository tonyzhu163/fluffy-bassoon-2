bash <<'FROZEN_EVENT_GPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=be247676c1217d9e046091c52cac2b7d0f2d3014
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/frozen-pi05-event-capacity-20261031-r2
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_CHILD=$TASK_PHASE/capacity-event-r1
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
timeout --signal=TERM --kill-after=5s 5090 "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_CHILD" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,signal,socket,subprocess,sys,time,traceback
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,child,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='be247676c1217d9e046091c52cac2b7d0f2d3014'
run='frozen-pi05-event-capacity-20261031-r1'
scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
model='/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1'
norm=model+'/assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
native='/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1'
admission=phase/'admission/event-r1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256=scope_sha,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
# One 5164-second all-in reservation: 5104 slice + existing 60 scheduler allowance.
attempt=begin(phase/'gpu-clock',contract,slice_seconds=5104,worker_seconds=28800,resume=True,now=started)
rc=1;error=None
try:
 grant_path=phase/'GRANT.json';grant=json.loads(grant_path.read_text())
 assert sha(grant_path)=='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4'
 assert grant['approved'] and grant['maximum_gpu_seconds']==28800 and grant['maximum_total_actions']==300000 and grant['maximum_output_bytes']==64*1024**3
 assert sha(phase/'SCOPE.json')==scope_sha
 rollup=admission/'EVENT-CAPACITY-CPU-ROLLUP.json';authority_path=admission/'EVENT-CAPACITY-CHILD-AUTHORITY.json'
 assert sha(rollup)=='2f67915587448aef3bb9d2704973a362fe1df8b53227991d43e036d1ac99155f'
 assert sha(authority_path)=='571f955a43903bb6a49f3489e7082c96f5cc246e8f1cc389975385c4cca21506'
 qualified=json.loads(rollup.read_text());assert qualified['passed'] and qualified['carried_actual_suite_cases']==24 and qualified['successor_affected_cases_passed']==2 and qualified['all12_schedules_admitted'] and qualified['actual_gpu_count']==0
 authority=json.loads(authority_path.read_text());assert authority['approved'] and authority['source_commit']==source and authority['maximum_new_gpu_seconds']==5164 and authority['parent_grant_sha256']==sha(grant_path)
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());assert len(clock['attempts'])==4 and [r.get('charged_seconds') for r in clock['attempts'][:3]]==[632,1138,266] and clock['attempts'][-1]['reserved_seconds']==5164
 probe=child/'gpu-probe.json'
 with (child/'gpu-probe.log').open('x') as log:
  subprocess.run([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 os.environ['LYCO_BAIGE_GPU_MAP']=str(probe)
 interface=phase/'admission/A1-INTERFACE.json';joints=phase/'admission/JOINT-CONVENTIONS.json';demos=phase/'admission/capacity-r1/demos.json';original=phase/'a1-r2/experiment/result.json'
 assert sha(interface)=='c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb'
 assert sha(joints)=='8a0ff1e70fd776cf8cdb9e254966422e1dee8227584c388a262e62068ebf9bc4'
 assert sha(demos)=='a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4'
 assert sha(original)=='aece67fdfd33af52653dd74587424450571288a26e03be18a0f1d99721fe3538'
 predecessor=phase/'capacity-r1/experiment/result.json';resume=phase/'capacity-r1/experiment/recovery.pt';migration=admission/'migration-be247.json'
 assert sha(predecessor)=='e9035c734ff5f7aa688d93e5a476794e452f1b19c2dedc813e5af262f0acd05b' and sha(resume)=='c59c91799938c928bdd683a6ce681cfa3f9853fe0c829c05fe6657fe30af5d5b'
 assert sha(migration)=='f91319f0a999817f8502f2a9eb1f5a254ab73980357b916442f1854affb3dfd3'
 ledger=json.loads(original.read_text())['action_ledger']
 assert ledger['source_caps']==grant['canonical_action_accounting']['source_caps'] and ledger['phase_caps']==grant['canonical_action_accounting']['phase_caps']
 config=dict(seed=20261031,rank=4,kv_scale=.1,gripper_bound=0.,updates=100,global_batch=2,learning_rate=1e-3,validate_every=25)
 ticket=dict(scope_sha256=scope_sha,source_commit=source,hostname=socket.gethostname(),gpu_count=1,output=str(child/'experiment'),phase_output_root=str(phase),driver_deadline_unix=started+4924,reserved_gpu_seconds=5164,prior_phase_gpu_seconds=2036,other_reserved_phase_gpu_seconds=0,prior_a1_gpu_seconds=2036,action_ledger=ledger,grant_sha256=sha(grant_path),cpu_rollup_sha256=sha(rollup),resource_attempt=str(attempt),child_authority_sha256=sha(authority_path),expected_prior_motor_parameter_sha256='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41',mode='original_a1_event_aligned_capacity_v1',run_id=run,capacity_action_bound=10000,capacity_fit_config=config,scheduler_start_unix=started)
 ticket_path=child/'allocation.json';write(ticket_path,ticket)
 args=[sys.executable,'scripts/diagnostics/robotwin_event_capacity.py','--scope',str(phase/'SCOPE.json'),'--scope-sha256',scope_sha,'--allocation',str(ticket_path),'--allocation-sha256',sha(ticket_path),'--interface',str(interface),'--interface-sha256',sha(interface),'--joint-conventions',str(joints),'--joint-conventions-sha256',sha(joints),'--demos',str(demos),'--demos-sha256',sha(demos),'--original-result',str(original),'--original-result-sha256',sha(original),'--predecessor-result',str(predecessor),'--predecessor-result-sha256',sha(predecessor),'--migration',str(migration),'--migration-sha256',sha(migration),'--resume',str(resume),'--resume-sha256',sha(resume),'--source-commit',source,'--run-id',run,'--output',str(child/'experiment'),'--model-path',model,'--norm-stats',norm,'--rlinf-root',native+'/RLinf','--robotwin-root',native+'/RoboTwin','--gpu-probe',str(probe)]
 write(child/'invocation.json',dict(arguments=args,started_unix=started,allocation_sha256=sha(ticket_path),interface_sha256=sha(interface),grant_sha256=sha(grant_path)))
 with (child/'console.log').open('x') as log:
  process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
  write(child/'driver-process.json',dict(pid=process.pid,parent_pid=os.getpid(),hostname=socket.gethostname(),source_commit=source))
  try:rc=process.wait(timeout=max(1,started+5064-time.time()))
  except subprocess.TimeoutExpired:
   process.send_signal(signal.SIGTERM)
   try:rc=process.wait(timeout=20)
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
FROZEN_EVENT_GPU_JOB
