bash <<'PARTIAL11_GPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=7eef998223aa93c91bc16f9599979aa3dc7f522d
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-endtoend-20261031-r1
TASK_ALLOC=$TASK_PHASE/partial11-cache-allocation-r1
TASK_PYTHON=$TASK_ROOT/envs/robotwin-vlrla-20261029-r1/bin/python
TASK_RLINF=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RLinf
TASK_ROBOTWIN=$TASK_ROOT/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_ALLOC"
mkdir "$TASK_ALLOC"
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
timeout --signal=TERM --kill-after=10s 1120 "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='7eef998223aa93c91bc16f9599979aa3dc7f522d';run='frozen-pi05-partial11-cache-20261101-r1'
lineage='frozen-pi05-endtoend-20261031-r1/partial11-model-cache'
cache=phase/'partial11-cache-r1';output=cache/run
scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
grant_sha='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4'
contract=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',scope_sha256=scope_sha,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
attempt=begin(phase/'gpu-clock',contract,slice_seconds=1140,worker_seconds=28800,resume=True,now=started,lineage=lineage,lineage_seconds=1200)
rc,error=1,None
try:
 assert sha(phase/'GRANT.json')==grant_sha and sha(phase/'SCOPE.json')==scope_sha
 cpu=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-partial11-factorial-cpu-20261101-r1/summary.json')
 assert sha(cpu)=='db6a4642e1a0cd524b5aa2c51c5906e8a24a2c104ba76494626d1a9b770b52fb' and json.loads(cpu.read_text())['passed']
 authority=phase/'admission/partial11-cache-r1/PARTIAL11-CHILD-AUTHORITY.json'
 assert sha(authority)=='5b67d67403baf716969054432e6008f08f32a75abe713db1eb03c80c236dffb6' and json.loads(authority.read_text())['maximum_new_GPU_seconds']==1200
 profile=code/'outputs/robotwin-partial11-cache-20261101-r1/PROFILE.json';inputs=code/'outputs/robotwin-fixed-teacher-bank-20261101-r1/INPUTS.json'
 assert sha(profile)=='4f2351d0218ff145423afebe6bb3bc662891acd9da41324f3a07fb01b720d46f'
 assert sha(inputs)=='2670a58da8e3228537e36bc6f2ff38ee29ad64dbc61065439378e7da5d32f480'
 pair_ticket_path=phase/'prelift-factorial-r1/allocation.json';pair_ticket=json.loads(pair_ticket_path.read_text())
 assert pair_ticket['source_commit']=='8632e421412388e8bb9c933b2b10cf5940dc7156' and pair_ticket['reserved_gpu_seconds']==900 and pair_ticket['mode']=='robotwin_phone04_prelift_dose_grip_factorial_v1'
 clock=json.loads((phase/'gpu-clock/worker-time.json').read_text());rows=[x for x in clock['attempts'] if x.get('lineage')==lineage]
 assert len(rows)==1 and rows[0]['path']==str(attempt) and rows[0]['reserved_seconds']==1200 and rows[0]['lineage_seconds']==1200
 assert any(x['path']==pair_ticket['resource_attempt'] and x['reserved_seconds']==900 for x in clock['attempts'])
 assert not cache.exists()
 probe=allocation/'gpu-probe.json'
 with (allocation/'gpu-probe.log').open('x') as log:
  subprocess.run([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 os.environ['LYCO_BAIGE_GPU_MAP']=str(probe)
 ticket=dict(mode='robotwin_partial11_model_cache_v1',source_commit=source,run_id=run,hostname=socket.gethostname(),gpu_count=1,output=str(output),cache_root=str(cache),phase_output_root=str(phase),clock_root=str(phase/'gpu-clock'),resource_attempt=str(attempt),pair_resource_attempt=pair_ticket['resource_attempt'],scope_sha256=scope_sha,profile_sha256=sha(profile),prior_cache_gpu_seconds=0,maximum_robot_actions=0,reserved_gpu_seconds=1200,scheduler_start_unix=started,driver_deadline_unix=started+1000,CACHE_LINEAGE=lineage,lineage_seconds=1200,original_pair_ticket_ref=dict(path=str(pair_ticket_path),sha256=sha(pair_ticket_path)),CPU_summary_sha256=sha(cpu),child_authority_sha256=sha(authority))
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native='/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1';model='/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1'
 args=[sys.executable,'scripts/diagnostics/robotwin_partial11_cache.py','--profile',str(profile),'--profile-sha256',sha(profile),'--inputs',str(inputs),'--inputs-sha256',sha(inputs),'--scope',str(phase/'SCOPE.json'),'--scope-sha256',scope_sha,'--grant',str(phase/'GRANT.json'),'--grant-sha256',grant_sha,'--allocation',str(ticket_path),'--allocation-sha256',sha(ticket_path),'--model-path',model,'--norm-stats',model+'/assets/pi0.5_clean_randomize_joint_training/norm_stats.json','--robotwin-root',native+'/RoboTwin','--rlinf-root',native+'/RLinf','--gpu-probe',str(probe),'--output',str(output),'--cache-root',str(cache),'--source-commit',source,'--run-id',run]
 write(allocation/'invocation.json',dict(arguments=args,source_commit=source,started_unix=started))
 with (allocation/'console.log').open('x') as log:
  process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ));write(allocation/'driver-process.json',dict(pid=process.pid,hostname=socket.gethostname()))
  try:rc=process.wait(timeout=max(1,started+1000-time.time()))
  except subprocess.TimeoutExpired:
   process.send_signal(signal.SIGTERM)
   try:rc=process.wait(timeout=10)
   except subprocess.TimeoutExpired:process.kill();rc=process.wait()
   error='Original cumulative cache deadline; preserve partial cursor/RNG and actual cost; no automatic retry'
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc)
 write(allocation/'allocation-closeout.json',dict(source_commit=source,exit_code=rc,error=error,attempt=row,charged_gpu_seconds=row['charged_seconds'],cache_lineage=lineage,maximum_cache_lineage_seconds=1200,maximum_new_actions=0))
 (allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
PARTIAL11_GPU_JOB
