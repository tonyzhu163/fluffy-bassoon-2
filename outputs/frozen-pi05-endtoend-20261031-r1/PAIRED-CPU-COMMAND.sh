bash <<'FROZEN_PAIRED_CPU_JOB'
set -Eeuo pipefail
TASK_STARTED=$(date +%s)
TASK_SOURCE=969fb44c503fac409373abc88ce5687b71195068
TASK_CODE=/mowenjie/fluffy-bassoon-2/worktrees/frozen-pi05-paired-reset-cpu-20261031-r1
TASK_OUT=/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-paired-reset-cpu-20261031-r1
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
timeout --signal=INT --kill-after=20s 540 "$TASK_PYTHON" - "$TASK_OUT" "$TASK_SOURCE" "$TASK_STARTED" <<'PY'
import hashlib, json, math, os, socket, subprocess, sys, time, traceback
from pathlib import Path
root, source, started = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
def write(name, value):
    p = root/name; temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(p)
assert 244+600 <=1800
clock = dict(prior_phase_cpu_charge=244, maximum_phase_cpu_worker_seconds=1800, format='frozen_pi05_endtoend_cpu_clock_v1', maximum_cpu_worker_seconds=600,
    phase_id='frozen-pi05-endtoend-20261031-r1', source_commit=source, started_unix=started, status='started', attempts=1,
    interrupted_charge_cpu_worker_seconds=600, charged_gpu_seconds=None)
write('cpu-worker-clock.json',clock)
rc, zero_gpu, error = 1, False, None
tests = ['tests/test_pi05_residual.py','tests/test_robotwin_noise.py','tests/test_robotwin_steering.py','tests/test_robotwin_campaign_budget.py','tests/test_robotwin_paired_reset.py']
try:
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
    with (root/'tests.log').open('x') as log:
        rc = subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider',
            '--junitxml='+str(root/'paired-reset.xml'),*tests],stdout=log,stderr=subprocess.STDOUT,
            env=dict(os.environ),timeout=max(1,started+350-time.time()),check=False).returncode

    if rc == 0:
        from fluffy_bassoon.meta_rl.robotwin_steering_screen import read_bound, build_plan, validate_instruction_migration
        proof_path=Path('/mowenjie/fluffy-bassoon-2/outputs/frozen-pi05-endtoend-20261031-r1/admission/a1-r2/A1-R2-MIGRATION.json')
        proof_sha='04c7c7c078b2e344ba9ac0aa574c592dd32192030e246752e49c4f3243c84097'
        proof=read_bound(proof_path,proof_sha)
        ledger=Path(proof['journal']['root']);snapshot=ledger/'snapshot.json';before=snapshot.read_bytes();old=json.loads(before)
        count_before=len(list((ledger/'events').glob('*.json')))
        assert old['journal_events']==count_before==352 and old['journal_sha256']==proof['journal']['sha256']
        assert old['actual_actions']==old['charged_actions']==5355 and old['pending_actions']==old['uncertain_actions']==0
        saved=torch.load(proof['recovery']['path'],map_location='cpu',weights_only=False)
        assert hashlib.sha256(Path(proof['recovery']['path']).read_bytes()).hexdigest()==proof['recovery']['sha256']
        prospective=dict(prospective_migration_only=True,actual_gpu_admission=False,prior_a1_gpu_seconds=632,prior_phase_gpu_seconds=632,action_ledger=dict(root=str(ledger)))
        result=validate_instruction_migration(saved,proof,resume=Path(proof['recovery']['path']),resume_sha=proof['recovery']['sha256'],plan=build_plan(saved['state']['plan']['limits']),source=source,scope_sha='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447',interface_sha='c7a6dbf1bee96f5984f9a708c732fd81d4f21c0bbccbb5b2dca4922d6c728ceb',allocation=prospective)
        after=snapshot.read_bytes();count_after=len(list((ledger/'events').glob('*.json')))
        assert before==after and count_before==count_after==352
        assert not torch.cuda.is_initialized() and torch.cuda.device_count()==0
        write('actual-migration-dryrun.json',dict(format='frozen_pi05_actual_migration_cpu_dryrun_v1',source_commit=source,proof_path=str(proof_path),proof_sha256=proof_sha,passed=True,result=result,prospective_allocation=prospective,canonical_snapshot_sha256_before=hashlib.sha256(before).hexdigest(),canonical_snapshot_sha256_after=hashlib.sha256(after).hexdigest(),event_count_before=count_before,event_count_after=count_after,canonical_journal_unchanged=True,campaign_budget_constructed=False,model_constructed=False,robot_actions=0,gpu_seconds=0))
except KeyboardInterrupt:
    rc,error = 130,traceback.format_exc()
except subprocess.TimeoutExpired:
    rc,error = 124,traceback.format_exc()
except BaseException:
    rc,error = 1,traceback.format_exc()
finally:
    ended=time.time(); charged=math.ceil(ended-started)+60
    clock.update(status='finished',ended_unix=ended,exit_code=rc,
        charged_cpu_worker_seconds=charged,remaining_cpu_worker_seconds=max(0,600-charged),
        charged_gpu_seconds=0 if zero_gpu else None)
    write('cpu-worker-clock.json',clock)
    log=root/'tests.log'
    write('summary.json',dict(format='frozen_pi05_paired_reset_cpu_qualification_v1',source_commit=source,
        phase_id='frozen-pi05-endtoend-20261031-r1',passed=rc==0,exit_code=rc,error=error,runtime=sys.executable,tests=tests,
        test_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None,
        synthetic_tests_and_actual_saved_metadata_validation_only=True,real_vla_weights=False,real_simulator=False,actual_migration_dryrun_passed=(root/'actual-migration-dryrun.json').exists(),
        robot_actions=0,optimizer_training=False,synthetic_backward_checks=True,
        charged_cpu_worker_seconds=charged,maximum_cpu_worker_seconds=600,
        charged_gpu_seconds=0 if zero_gpu else None))
    (root/'exit-code.txt').write_text(str(rc)+'\n')
raise SystemExit(rc)
PY
FROZEN_PAIRED_CPU_JOB
