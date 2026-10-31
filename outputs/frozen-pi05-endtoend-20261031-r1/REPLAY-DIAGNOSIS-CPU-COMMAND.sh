bash <<'REPLAY_DIAGNOSIS_CPU'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=4bbf6e5ce47b69273b2e71e0f6201b88dec30bcd
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/robotwin-replay-diagnosis-cpu-20261031-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/robotwin-replay-diagnosis-cpu-20261031-r1
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
# One all-in300s envelope.225s outer+5s forced cleanup+60s scheduler allowance leaves10s margin.
timeout --signal=INT --kill-after=5s 225 "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib,json,math,os,socket,subprocess,sys,time,traceback
from pathlib import Path
root,source,started=Path(sys.argv[1]),sys.argv[2],float(sys.argv[3]);deadline=started+210
phase=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1')
code=Path.cwd()
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def write(name,value):
 p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');t.replace(p)
clock=dict(format='frozen_pi05_endtoend_cpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',source_commit=source,started_unix=started,deadline_unix=deadline,maximum_cpu_worker_seconds=300,prior_phase_cpu_charge=746,maximum_phase_cpu_worker_seconds=1800,interrupted_charge_cpu_worker_seconds=300,status='started',attempts=1,charged_gpu_seconds=None)
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
 write('allocation.json',dict(format='frozen_pi05_replay_cpu_ticket_v1',phase_id='frozen-pi05-endtoend-20261031-r1',hostname=socket.gethostname(),gpu_count=0,source_commit=source,output=str(root),inspection_output=str(root/'inspection'),maximum_cpu_worker_seconds=300,prior_phase_cpu_charge=746,maximum_phase_cpu_worker_seconds=1800,actual_cpu_cores=cores,actual_memory_bytes=memory,runtime=sys.executable,zero_gpu_attestation_sha256=sha(proof),started_unix=started,deadline_unix=deadline,governing_grant_sha256='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4',scope='13 synthetic tests plus no-model/no-simulator inspection of exactly three closed teacher replays'))
 evidence=code/'outputs/robotwin-replay-diagnosis-20261031-r1/SOURCE.json'
 assert sha(evidence)=='b20b5be52f1ac40506f3aa97f13c45b71a5a53faf5837150df7d0e89c509e8c5'
 with (root/'tests.log').open('x') as log:
  subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','tests/test_robotwin_replay_diagnosis.py','--junitxml='+str(root/'replay.junit.xml')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),timeout=max(1,deadline-time.time()-5),check=True)
 args=[sys.executable,'scripts/diagnostics/robotwin_replay_diagnosis.py','--result',str(phase/'capacity-r1/experiment/result.json'),'--result-sha256','e9035c734ff5f7aa688d93e5a476794e452f1b19c2dedc813e5af262f0acd05b','--demos',str(phase/'admission/capacity-r1/demos.json'),'--demos-sha256','a80c5ed4776ced9d02e2c1c2ab44059f542d47eb8087eed561b12ab6fe700ad4','--source-evidence',str(evidence),'--source-evidence-sha256',sha(evidence),'--allocation',str(root/'allocation.json'),'--allocation-sha256',sha(root/'allocation.json'),'--zero-gpu-proof',str(proof),'--zero-gpu-proof-sha256',sha(proof),'--source-commit',source,'--deadline-unix',str(deadline),'--output',str(root/'inspection')]
 write('invocation.json',dict(arguments=args,original_deadline_unix=deadline))
 with (root/'inspection.log').open('x') as log:
  subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),timeout=max(1,deadline-time.time()),check=True)
 rc=0
except KeyboardInterrupt:rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:rc,error=124,traceback.format_exc()
except BaseException:rc,error=1,traceback.format_exc()
finally:
 ended=time.time();charged=math.ceil(ended-started)+60
 clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,300-charged),charged_gpu_seconds=0 if zero_gpu else None)
 write('cpu-worker-clock.json',clock)
 write('summary.json',dict(format='frozen_pi05_replay_diagnosis_cpu_qualification_v1',source_commit=source,phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,tests='tests/test_robotwin_replay_diagnosis.py',test_log_sha256=sha(root/'tests.log') if (root/'tests.log').exists() else None,no_models=True,no_simulator=True,robot_actions=0,optimizer_training=False,charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=300,charged_gpu_seconds=0 if zero_gpu else None,inspection_index_sha256=sha(root/'inspection/index.json') if (root/'inspection/index.json').exists() else None))
 (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
REPLAY_DIAGNOSIS_CPU
