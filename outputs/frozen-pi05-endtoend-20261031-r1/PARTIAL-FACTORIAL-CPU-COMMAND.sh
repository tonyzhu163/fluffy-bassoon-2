bash <<'PARTIAL_FACTORIAL_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=7eef998223aa93c91bc16f9599979aa3dc7f522d
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-partial11-cache-20261101-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/robotwin-partial11-factorial-cpu-20261101-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
# One all-in180s envelope:110s outer+5s forced cleanup+60s allowance;5s setup margin.
timeout --signal=INT --kill-after=5s 110 "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+90
phase=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1')
code=Path.cwd()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='frozen_pi05_endtoend_cpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',source_commit=source,started_unix=started,deadline_unix=deadline,maximum_cpu_worker_seconds=180,prior_phase_cpu_charge=1594,maximum_phase_cpu_worker_seconds=1800,interrupted_charge_cpu_worker_seconds=180,status='started',attempts=1,charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc,zero_gpu,error=1,False,None
try:
 assert sys.platform=='linux' and socket.gethostname().startswith('job-')
 assert sys.version_info[:2]==(3,11)
 import torch
 initialized=torch.cuda.is_initialized();visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')};count=torch.cuda.device_count()
 zero_gpu=count==0 and not initialized and not torch.cuda.is_initialized()
 paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current']
 raw={p:Path(p).read_text().strip() for p in paths};quota,period=raw['/sys/fs/cgroup/cpu.max'].split();cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
 write('zero-gpu-attestation.json',dict(passed=zero_gpu,hostname=socket.gethostname(),device_count_before_mask=count,cuda_initialized_before_mask=initialized,environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,python=sys.version,raw_cgroup=raw,actual_cpu_cores=cores,actual_memory_bytes=memory,required_cpu_cores=2,required_memory_bytes=16*1024**3))
 assert zero_gpu and cores==2 and memory==16*1024**3
 assert str(torch.__version__).startswith('2.11.')
 os.environ['CUDA_VISIBLE_DEVICES']=''
 proof=root/'zero-gpu-attestation.json'
 write('allocation.json',dict(format='frozen_pi05_partial_factorial_cpu_ticket_v1',phase_id='frozen-pi05-endtoend-20261031-r1',hostname=socket.gethostname(),gpu_count=0,source_commit=source,output=str(root),maximum_cpu_worker_seconds=180,prior_phase_cpu_charge=1594,maximum_phase_cpu_worker_seconds=1800,actual_cpu_cores=cores,actual_memory_bytes=memory,runtime=sys.executable,zero_gpu_attestation_sha256=sha(proof),started_unix=started,deadline_unix=deadline,governing_grant_sha256='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4',scope='Only29newpartial11 and33newfactorial cases plus actual04preflight; no model/simulator/ledger/actions'))
 factorial=Path('/mowenjie/fluffy-bassoon-2/worktrees/robotwin-prelift-factorial-20261101-r1')
 assert subprocess.check_output(['git','-C',str(factorial),'rev-parse','HEAD'],text=True).strip()=='8632e421412388e8bb9c933b2b10cf5940dc7156'
 assert not subprocess.check_output(['git','-C',str(factorial),'status','--porcelain'],text=True).strip()
 profile=factorial/'outputs/robotwin-prelift-dose-grip-20261101-r1/PROFILE.json'
 assert sha(profile)=='4f51314865605578851e7912d34f0984d019f21bfa82870733d36ab188ed6daa'
 write('component-sources.json',dict(partial11=dict(source_commit=source,worktree=str(code)),factorial=dict(source_commit='8632e421412388e8bb9c933b2b10cf5940dc7156',worktree=str(factorial)),one_original_deadline=deadline))
 with (root/'partial11-tests.log').open('x') as log:
  subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_partial11_cache.py','--junitxml='+str(root/'partial11-wrapper.junit.xml')],stdout=log,stderr=subprocess.STDOUT,cwd=code,env=dict(os.environ),timeout=max(1,deadline-time.time()-5),check=True)
 env=dict(os.environ,PYTHONPATH=str(factorial/'src'),TASK_CODE=str(factorial),TASK_PYTHON=sys.executable,TASK_RECEIPTS=str(root),TASK_DEADLINE_UNIX=str(deadline),TASK_PROFILE=str(profile),TASK_PROFILE_SHA256=sha(profile),TASK_DEMOS=str(phase/'admission/capacity-r1/demos.json'),TASK_DEMOS_SHA256='a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4')
 with (root/'factorial-tests-preflight.log').open('x') as log:
  subprocess.run(['bash',str(factorial/'outputs/robotwin-prelift-dose-grip-20261101-r1/CPU-PAYLOAD.sh')],stdout=log,stderr=subprocess.STDOUT,cwd=factorial,env=env,timeout=max(1,deadline-time.time()-5),check=True)
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:rc,error=124,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 if charged>180:rc,error=1,(error or '')+' Inclusive180s CPU envelope exceeded'
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,180-charged),charged_gpu_seconds=0 if zero_gpu else None)
 write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='frozen_pi05_partial_factorial_cpu_qualification_v1',source_commit=source,phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,tests=['partial11@7eef:tests/test_robotwin_partial11_cache.py','factorial@8632:tests/test_robotwin_prelift_factorial.py'],test_log_sha256=sha(root/'partial11-tests.log') if (root/'partial11-tests.log').exists() else None,preflight_sha256=sha(root/'preflight04-factorial.json') if (root/'preflight04-factorial.json').exists() else None,no_models=True,no_simulator=True,robot_actions=0,optimizer_training=False,charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=180,charged_gpu_seconds=0 if zero_gpu else None))
 (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
PARTIAL_FACTORIAL_CPU
