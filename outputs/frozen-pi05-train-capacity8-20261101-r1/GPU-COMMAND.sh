bash <<'TRAIN_CAPACITY8_GPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=86f4a962c8f7fc83cfcfa39b466dafcde9f2b25b
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_CODE=$TASK_ROOT/worktrees/robotwin-train-capacity8-20261101-r1
TASK_HELPER=$TASK_ROOT/worktrees/robotwin-partial11-cache-20261101-r1
TASK_PHASE=$TASK_ROOT/outputs/frozen-pi05-train-capacity8-20261101-r1
TASK_ALLOC=$TASK_PHASE/gpu-r1
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
TASK_REMAINING=$((TASK_STARTED+1130-$(date +%s)))
test "$TASK_REMAINING" -gt 0
timeout --signal=TERM --kill-after=10s "$TASK_REMAINING" "$TASK_PYTHON" - "$TASK_PHASE" "$TASK_ALLOC" "$TASK_CODE" "$TASK_STARTED" <<'PY'
import hashlib,json,os,socket,subprocess,sys,time,traceback,signal,shutil
from pathlib import Path
from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
phase,allocation,code,started=Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4])
source='86f4a962c8f7fc83cfcfa39b466dafcde9f2b25b';phase_id='frozen-pi05-train-capacity8-20261101-r1';output=phase/'experiment';admission=phase/'admission'
contract=dict(format='frozen_pi05_train_capacity8_GPU_clock_v1',phase_id=phase_id,maximum_GPU_seconds=1200,maximum_actions=3200,maximum_output_bytes=4*1024**3,authority_sha256='cf36a7d955065fb03c2a8eab381f74e768c6ff784684b9dc8ad89b5bb7287b37',protocol_sha256='c2fd3076f0e3a54b8df054b761a0dd61531023ae48b739c19cab1d3464f56807',grant_sha256='8248afafbc5e6f8a268dcddd17513c589b641291f161696f9054b223ae7343f8',source_commit=source,motor_file_sha256='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p))
def write(p,v):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def run_process(args,log,env,deadline):
 with Path(log).open('x') as stream:
  proc=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,cwd=code,env=env,start_new_session=True)
  write(allocation/(Path(log).stem+'-process.json'),dict(pid=proc.pid,parent_pid=os.getpid(),hostname=socket.gethostname(),source_commit=source))
  try:return proc.wait(timeout=max(1,deadline-time.time()))
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise TimeoutError('Original1200phase deadline; preserve partial/cursor and no automatic replay')
  finally:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   else:
    until=time.monotonic()+2
    while time.monotonic()<until:
     try:os.killpg(proc.pid,0)
     except ProcessLookupError:break
     time.sleep(.05)
    else:
     try:os.killpg(proc.pid,signal.SIGKILL)
     except ProcessLookupError:pass
attempt=begin(phase/'gpu-clock',contract,slice_seconds=1140,worker_seconds=1200,resume=False,now=started,lineage=phase_id,lineage_seconds=1200)
rc,error=1,None
try:
 assert socket.gethostname().startswith('job-')
 grant=phase/'GRANT.json';assert sha(grant)==contract['grant_sha256'];grant_data=json.loads(grant.read_text())
 cpu=phase/'cpu-r1/summary.json';assert sha(cpu)=='c8380898c07be4111bd1cf47748a4f58b6554ab4e31d4bc1fee6ce1e98be7279';summary=json.loads(cpu.read_text());assert summary['source_commit']==source and summary['all_passed'] and summary['JUnit']['tests']=='11'
 request=admission/'REQUEST.json';assert sha(request)=='f679d245fbcb3e9dcf7bfeb2320fcf5bba265c6bda1ef3f98153f241624ed7c4';request_data=json.loads(request.read_text())
 storage=admission/'STORAGE.json';assert sha(storage)=='a956df353ed71c0bb08e9f60808843f8448635af03c7b7fe05501480da708666';assert shutil.disk_usage(phase).free>500*1000**3+128*1024**3+4*1024**3
 review=admission/'LATEST-MAIN-REVIEW.json';assert sha(review)=='2ab71c76046feaf6b726691437e0c5b44ad47a71926aaa1e65715abb25b80658'
 env=dict(os.environ,PYTHONPATH=str(code/'src')+':'+os.environ['REPO_PATH']+':'+os.environ['ROBOTWIN_PATH'])
 probe=allocation/'gpu-probe.json';assert run_process([sys.executable,'scripts/[train]gpu_probe.py','--expected-count','1','--output',str(probe),'--purpose','training','--baige-container','--baige-exclusive-training'],allocation/'gpu-probe.log',env,started+110)==0;env['LYCO_BAIGE_GPU_MAP']=str(probe)
 raw={n:Path(n).read_text().strip() for n in ['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max']};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();assert int(quota)/int(period)==16 and int(raw['/sys/fs/cgroup/memory.max'])==96*1024**3
 job_id=socket.gethostname().removesuffix('-master-0');actual=allocation/'actual-allocation.json';write(actual,dict(job_id=job_id,source_commit=source,resources=grant_data['GPU_resources'],actual_cpu_cores=16,actual_memory_bytes=96*1024**3,raw_cgroup=raw,actual_GPU_probe=ref(probe)))
 clock_snapshot=allocation/'clock-at-start.json';write(clock_snapshot,json.loads((phase/'gpu-clock/worker-time.json').read_text()))
 ledger=dict(root=str(phase/'action-budget'),campaign_id=phase_id,total_actions=3200,source_caps={},phase_caps={'train_capacity8':3200},contract=dict(phase_id=phase_id,authority_sha256=request_data['authority']['sha256'],protocol_sha256=request_data['protocol']['sha256'],grant_sha256=request_data['grant']['sha256'],prior_closed_actions=29922,origin_recovery_sha256='aa44ce2d60c0352265c1138b078cba8106a86a69f875f7577e691a9705cee1c4',source_eligible=False))
 assert not Path(ledger['root']).exists() and not output.exists()
 ticket=dict(format='train_capacity8_actual_owner_ticket_v1',admitted=True,phase_id=phase_id,scope='first8_TRAIN_controls_only',source_commit=source,request_sha256=sha(request),hostname=socket.gethostname(),job_id=job_id,gpu_count=1,output=str(output),aggregate_output_root=str(phase),run_id=phase_id+'-gpu-r1',automatic_retry=False,maximum_gpu_seconds=1200,maximum_output_bytes=4*1024**3,driver_deadline_unix=started+1080,cleanup_seconds=60,minimum_episode_admission_seconds=90,current_worker_clock=ref(clock_snapshot),clock_contract=contract,resource_attempt=str(attempt),actual_allocation=ref(actual),focused_CPU_qualification=ref(cpu),latest_main_review=ref(review),storage_admission=ref(storage),action_ledger=ledger,resume=None)
 ticket_path=allocation/'allocation.json';write(ticket_path,ticket)
 native=Path('/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1');model=Path('/mowenjie/fluffy-bassoon-2/cache/robotwin/pi05-motus-effc6e1a-r1')
 args=[sys.executable,'-m','fluffy_bassoon.meta_rl.robotwin_train_capacity8','--request',str(request),'--request-sha256',sha(request),'--ticket',str(ticket_path),'--ticket-sha256',sha(ticket_path),'--gpu-probe',str(probe),'--model-path',str(model),'--norm-stats',str(model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'),'--robotwin-root',str(native/'RoboTwin'),'--rlinf-root',str(native/'RLinf'),'--output',str(output)]
 write(allocation/'invocation.json',dict(source_commit=source,arguments=args,maximum_GPU_seconds=1200,maximum_controller_actions=3200,maximum_episodes=8,source_eligible=False))
 rc=run_process(args,allocation/'console.log',env,started+1100)
except BaseException:rc,error=1,traceback.format_exc()
finally:
 row=finish(phase/'gpu-clock',attempt,rc);write(allocation/'allocation-closeout.json',dict(source_commit=source,exit_code=rc,error=error,attempt=row,charged_GPU_seconds=row['charged_seconds'],maximum_phase_GPU_seconds=1200,maximum_new_actions=3200,all_full_recoveries_and_raw_preserved=True));(allocation/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
TRAIN_CAPACITY8_GPU
