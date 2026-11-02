"""One finite allocated GPU, fresh process for each 8/16/32-lane measurement."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank, PPO_TASKS

root = Path('/mowenjie/fluffy-bassoon-2')
code = Path(os.environ['TASK_META'])
out = Path(os.environ['TASK_OUT'])
started = int(os.environ['TASK_STARTED'])
deadline = started + 1710
assert socket.gethostname().startswith('job-') and torch.cuda.device_count() == 1
assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) >= 64
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) >= 256 << 30
space = os.statvfs(out)
assert space.f_bavail * space.f_frsize >= 32 << 30
(out / 'write-probe').write_text('capacity 16/32 write probe\n')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
probe = out / 'gpu-probe.json'
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '1',
    '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(probe)], check=True, timeout=90)
device = json.loads(probe.read_text())['devices'][0]
write_new(out / 'allocation.json', dict(hostname=socket.gethostname(), device=device,
    source_commit=os.environ['TASK_CODE_PIN'], started_unix=started, deadline_unix=deadline,
    free_bytes=space.f_bavail * space.f_frsize, quota='UNVERIFIED'))
pipeline = json.loads((root / 'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
cases = []
banks = {task: case_bank(task, 'baseline', count=8) for task in PPO_TASKS}
for slot in range(32):
    case = dict(banks[PPO_TASKS[slot % 5]][slot // 5])
    case.update(bank='capacity_1632', native_seed=6_731_000 + slot, slot=slot)
    cases.append(case)
write_new(out / 'cases.json', cases)
env = dict(os.environ)
env.update(pipeline['policy_env'])
env.update(PYTHONPATH=str(code / 'src') + ':' + str(root / 'sources/robocasa365-public-pi05-nnx-leaves-20261101-r1/RLinf'),
    CUDA_VISIBLE_DEVICES=device['visibility_token'], LYCO_BAIGE_GPU_MAP=str(probe),
    LYCO_BAIGE_ISOLATED_UUID=device['uuid'], MUJOCO_EGL_DEVICE_ID=str(device['allocation_ordinal']),
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
rows = []
for lanes in (8, 16, 32):
    if time.time() + 180 >= deadline:
        rows.append(dict(batch_size=lanes, completed=False, error='Remaining finite clock insufficient'))
        continue
    profile = dict(output=str(out / f'lanes-{lanes}'), rank=0, batch_size=lanes, cases=cases,
        initializer=pipeline['training']['initializer'],
        initial_checkpoint=str(root / 'outputs/robocasa-residual-approved-20261101-r1/S/boundary-000.pt'),
        accounting_root=str(out / f'actions-{lanes}'), deadline_unix=deadline, closure_seconds=90,
        source_commit=os.environ['TASK_CODE_PIN'])
    profile_path = out / f'profile-{lanes}.json'
    write_new(profile_path, profile)
    with (out / f'lanes-{lanes}.log').open('x') as stream:
        child = subprocess.Popen([sys.executable, '-B', str(code / 'scripts/diagnostics/robocasa_rollout_batch_benchmark.py'),
            '--profile', str(profile_path)], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=max(1, deadline - time.time()))
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    path = out / f'lanes-{lanes}/result.json'
    row = json.loads(path.read_text()) if path.exists() else dict(batch_size=lanes, completed=False, error='No final result; inspect retained log')
    row['exit_code'] = child.returncode
    rows.append(row)
    write_new(out / f'completed-{lanes}.json', row)
write_new(out / 'RESULT.json', dict(completed=all(row['completed'] for row in rows),
    configurations=rows, optimizer_updates=0, source_commit=os.environ['TASK_CODE_PIN'],
    elapsed_seconds=time.time()-started, GPU_seconds_upper=time.time()-started,
    interpretation='One GPU; capacity-only; shared native model plus zero-initialized action residual and real three-view simulator episodes. Flora colocation and joint steering history overhead are not measured.'))
