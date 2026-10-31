bash <<'FROZEN_COMPOSITE_CPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=8fb991c8d5a3b9f86a81e2a4f2cf55b185cec81f
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/frozen-pi05-capacity-fixture-cpu-20261031-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-capacity-fixture-cpu-20261031-r1
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
timeout --signal=INT --kill-after=5s 270 "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib, json, math, os, socket, subprocess, sys, time, traceback
from pathlib import Path
root, source, started = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
def write(name, value):
    p = root/name; temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(p)
assert 672+345 <=1800
clock = dict(qualification_aggregate_ceiling=600,prior_qualification_cpu_charge=255,prior_phase_cpu_charge=672, maximum_phase_cpu_worker_seconds=1800, format='frozen_pi05_endtoend_cpu_clock_v1', maximum_cpu_worker_seconds=345,
    phase_id='frozen-pi05-endtoend-20261031-r1', source_commit=source, started_unix=started, status='started', attempts=1,
    interrupted_charge_cpu_worker_seconds=345, charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc, zero_gpu, error = 1, False, None
tests = ['tests/test_robotwin_reactive_capacity.py::test_actual_metadata_composition_keeps_absent_flags_and_fixed_whole_split', 'tests/test_robotwin_reactive_capacity.py::test_fail_closed_demo_identity_and_author_binding', 'tests/test_robotwin_reactive_capacity.py::test_hdf5_reader_rejects_extra_or_truncated_timeline', 'tests/test_robotwin_reactive_capacity.py::test_author_jpeg_writer_inverse_retains_native_channels_camera_order_and_state', 'tests/test_robotwin_reactive_capacity.py::test_native_paired_plan_charges_all_teachers_and_arms_without_locked_panel']
try:
    prior_path=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-condition-composite-cpu-20261031-r1/cpu-worker-clock.json')
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest()=='1a1c54b684463c0094ff1ae72494ba70673ef6d16295a13b8b2eb3a5f993a32e', 'Prior CPU clock bytes changed'
    prior_clock=json.loads(prior_path.read_text())
    assert prior_clock['status']=='finished' and prior_clock['charged_cpu_worker_seconds']==134
    assert prior_clock['maximum_cpu_worker_seconds']==600
    prior2_path=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-condition-composite-cpu-20261031-r2/cpu-worker-clock.json')
    assert hashlib.sha256(prior2_path.read_bytes()).hexdigest()=='46f15a9d06364d17d0aa23a6cc169ec114174016cb2dbad291484d5db6541869', 'Second CPU clock bytes changed'
    prior2=json.loads(prior2_path.read_text())
    assert prior2['status']=='finished' and prior2['charged_cpu_worker_seconds']==121
    assert prior2['maximum_cpu_worker_seconds']==466 and 134+121+345==600
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert sys.version_info[:2] == (3,11)
    import torch
    initialized = torch.cuda.is_initialized()
    visible = {k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES','PYTHONPATH')}
    count = torch.cuda.device_count()
    zero_gpu = count == 0 and not initialized and not torch.cuda.is_initialized()
    paths = ['/proc/self/cgroup','/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/memory.max',
             '/sys/fs/cgroup/cpuset.cpus.effective','/sys/fs/cgroup/memory.current']
    raw = {p:Path(p).read_text().strip() for p in paths}
    quota,period = raw['/sys/fs/cgroup/cpu.max'].split()
    cores,memory = int(quota)/int(period),int(raw['/sys/fs/cgroup/memory.max'])
    write('zero-gpu-attestation.json',dict(passed=zero_gpu, hostname=socket.gethostname(),
        device_count_before_mask=count,cuda_initialized_before_mask=initialized,
        environment_before_mask=visible,torch_version=torch.__version__,torch_path=torch.__file__,
        runtime=sys.executable,python=sys.version,raw_cgroup=raw,
        actual_cpu_cores=cores,actual_memory_bytes=memory,
        required_cpu_cores=2,required_memory_bytes=16*1024**3))
    assert zero_gpu, 'Require actual0GPU before applying a CPU-only mask'
    assert cores == 2 and memory == 16*1024**3, 'Require exact2CPU/16GiB allocation'
    assert str(torch.__version__).startswith('2.11.'), 'Require qualifiedTorch2.11 runtime'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    write('allocation.json',dict(format='frozen_pi05_composite_cpu_ticket_v1',phase_id='frozen-pi05-endtoend-20261031-r1',hostname=socket.gethostname(),gpu_count=0,source_commit=source,output=str(root),maximum_cpu_worker_seconds=345,prior_phase_cpu_charge=672,maximum_phase_cpu_worker_seconds=1800,actual_cpu_cores=cores,actual_memory_bytes=memory,runtime=sys.executable,zero_gpu_attestation_sha256=hashlib.sha256((root/'zero-gpu-attestation.json').read_bytes()).hexdigest(),governing_grant_sha256='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4',qualification_aggregate_ceiling=600,prior_qualification_cpu_charge=255,prior_clock_sha256='1a1c54b684463c0094ff1ae72494ba70673ef6d16295a13b8b2eb3a5f993a32e',scope='Only10 previously blocked capacity fixture cases; source10/capacity22 and original87passes retained; no source collection, VLA, simulator or real fit'))
    with (root/'tests.log').open('x') as log:
        rc = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            '--junitxml='+str(root/'capacity-fixture.junit.xml'),*tests],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ),timeout=max(1,started+230-time.time()),check=False).returncode
except KeyboardInterrupt:
    rc,error = 130,traceback.format_exc()
except subprocess.TimeoutExpired:
    rc,error = 124,traceback.format_exc()
except BaseException:
    rc,error = 1,traceback.format_exc()
finally:
    ended=time.time(); charged=math.ceil(ended-started)+60
    clock.update(status='finished',ended_unix=ended,exit_code=rc,
        charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,345-charged),
        charged_gpu_seconds=0 if zero_gpu else None)
    write('cpu-worker-clock.json',clock)
    log=root/'tests.log'
    write('summary.json',dict(format='frozen_pi05_condition_composite_cpu_qualification_v1',source_commit=source,
        phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,tests=tests,
        test_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None,
        synthetic_tests_only=True,real_vla_weights=False,real_simulator=False,
        robot_actions=0,optimizer_training=False,synthetic_backward_checks=True,
        charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=345,
        charged_gpu_seconds=0 if zero_gpu else None))
    (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
FROZEN_COMPOSITE_CPU_JOB
