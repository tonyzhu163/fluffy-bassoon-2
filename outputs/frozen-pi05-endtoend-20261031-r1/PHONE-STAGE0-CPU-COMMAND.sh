bash <<'PHONE_STAGE0_CPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=9e1953e6ee2ea4c91fdc494355fce2ab55c99c9d
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/frozen-pi05-capacity-data-cpu-20261031-r1-source9e1953e6
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/capacity-stage0-cpu-r1
TASK_DATA=/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/capacity-data-r1
TASK_PYTHON=/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python
cd "$TASK_CODE"
test "$(git rev-parse HEAD)" = "$TASK_SOURCE"
test -z "$(git status --porcelain)"
test ! -e "$TASK_OUT"
test ! -e "$TASK_DATA"
mkdir "$TASK_OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export PYTHONPATH=$TASK_CODE/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
# One shared <=240s inner clock, including the tests/data work and cleanup.
# Hard interruption leaves another60 scheduler/startup seconds within300.
timeout --signal=INT --kill-after=5s 230 "$TASK_PYTHON" - "$TASK_OUT" "$TASK_DATA" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib, json, math, os, socket, subprocess, sys, time, traceback
from pathlib import Path
root, data, source, started = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])
phase = root.parent
deadline = started + 240
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name, value):
    p = root/name
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(p)
clock = dict(format='frozen_pi05_endtoend_cpu_clock_v1', phase_id='frozen-pi05-endtoend-20261031-r1',
    prior_phase_cpu_charge=331, maximum_phase_cpu_worker_seconds=1800,
    maximum_cpu_worker_seconds=300, source_commit=source, started_unix=started,
    shared_inner_deadline_unix=deadline, status='started', attempts=1,
    interrupted_charge_cpu_worker_seconds=300, charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc, zero_gpu, error, tests_rc, data_rc = 1, False, None, None, None
test_file='tests/test_robotwin_capacity_data_stage0.py'
snapshot=phase/'action-budget/snapshot.json'
try:
    assert sys.platform=='linux' and socket.gethostname().startswith('job-')
    assert sys.version_info[:2]==(3,11)
    import torch
    initialized=torch.cuda.is_initialized()
    visible={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')}
    count=torch.cuda.device_count()
    zero_gpu=count==0 and not initialized and not torch.cuda.is_initialized()
    paths=['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max',
           '/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current']
    raw={p:Path(p).read_text().strip() for p in paths}
    quota,period=raw['/sys/fs/cgroup/cpu.max'].split()
    cores,memory=int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
    attestation=dict(passed=zero_gpu,hostname=socket.gethostname(),device_count_before_mask=count,
        cuda_initialized_before_mask=initialized,environment_before_mask=visible,
        torch_version=torch.__version__,torch_path=torch.__file__,runtime=sys.executable,
        python=sys.version,raw_cgroup=raw,actual_cpu_cores=cores,actual_memory_bytes=memory,
        required_cpu_cores=2,required_memory_bytes=16*1024**3)
    write('zero-gpu-attestation.json',attestation)
    assert zero_gpu,'Require actual0GPU before masking'
    assert cores==2 and memory==16*1024**3,'Require exact2CPU/16GiB'
    assert str(torch.__version__).startswith('2.11.'),'Require qualified Torch2.11'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    before=digest(snapshot)
    ticket=dict(format='robotwin_capacity_stage0_allocation_v1',phase_id='frozen-pi05-endtoend-20261031-r1',
        hostname=socket.gethostname(),gpu_count=0,source_commit=source,output=str(data),
        maximum_cpu_worker_seconds=300,prior_phase_cpu_charge=331,
        parent_grant_sha256='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4',
        runtime=sys.executable,zero_gpu_attestation_sha256=digest(root/'zero-gpu-attestation.json'),
        actual_cpu_cores=cores,actual_memory_bytes=memory,action_snapshot_sha256_before=before,
        scope='Phone-only source data/numerical/alignment/gripper inspection; no model/simulator/action work')
    write('allocation.json',ticket)
    with (root/'tests.log').open('x') as log:
        tests_rc=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            test_file,'--junitxml='+str(root/'stage0.xml')],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ),timeout=max(1,deadline-time.time()-30),check=False).returncode
    if tests_rc:
        rc=tests_rc
    else:
        remaining=int(deadline-time.time()-20)
        if remaining<1:raise TimeoutError('Shared stage0 CPU allowance exhausted after tests')
        command=[sys.executable,'scripts/diagnostics/robotwin_capacity_data_stage0.py',
            '--source-commit',source,'--allocation',str(root/'allocation.json'),
            '--allocation-sha256',digest(root/'allocation.json'),'--max-seconds',str(min(240,remaining)),
            '--phase-root',str(phase),'--output',str(data),
            '--robotwin-root','/mowenjie/fluffy-bassoon-2/sources/robotwin-baseline-smoke-20261028-r1/RoboTwin',
            '--archive-cache','/mowenjie/fluffy-bassoon-2/inputs/robotwin-phone-clean50-3dc3b798-20261031-r1/aloha-agilex_clean_50.zip',
            '--a1-corpus',str(phase/'a1-r1/experiment/corpus'),
            '--a1-corpus',str(phase/'a1-r2/experiment/corpus')]
        write('stage0-invocation.json',dict(command=command,shared_deadline_unix=deadline))
        with (root/'data.log').open('x') as log:
            data_rc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ),
                timeout=max(1,deadline-time.time()-10),check=False).returncode
        rc=data_rc
    after=digest(snapshot)
    write('action-snapshot-no-mutation.json',dict(before_sha256=before,after_sha256=after,
        unchanged=before==after,action_ledger_opened=False))
    if before!=after:raise RuntimeError('Closed canonical action snapshot unexpectedly changed')
except KeyboardInterrupt:
    rc,error=130,traceback.format_exc()
except subprocess.TimeoutExpired:
    rc,error=124,traceback.format_exc()
except BaseException:
    rc,error=1,traceback.format_exc()
finally:
    ended=time.time();charged=math.ceil(ended-started)+60
    clock.update(status='finished',ended_unix=ended,exit_code=rc,charged_cpu_worker_seconds=charged,
        remaining_cpu_worker_seconds=max(0,300-charged),charged_gpu_seconds=0 if zero_gpu else None)
    write('cpu-worker-clock.json',clock)
    write('summary.json',dict(format='robotwin_phone_stage0_cpu_qualification_v1',source_commit=source,
        phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,
        tests=[test_file],tests_exit_code=tests_rc,data_exit_code=data_rc,
        runtime=sys.executable,real_vla_weights=False,real_simulator=False,robot_actions=0,
        optimizer_training=False,charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=300,
        charged_gpu_seconds=0 if zero_gpu else None,capacity_gpu_admitted=False))
    (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
PHONE_STAGE0_CPU_JOB
