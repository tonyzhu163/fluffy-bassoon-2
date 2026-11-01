"""Four scheduler nodes, two actual native rollout GPUs each; one same learner."""
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

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
node_rank = int(os.environ['RANK'])
assert os.environ['WORLD_SIZE'] == '4' and os.environ['NPROC_PER_NODE'] == '2' and 0 <= node_rank < 4
assert torch.cuda.device_count() == 2
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
start = int(os.environ['TASK_STARTED'])
deadline = start + 14040
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) == 32
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 128 << 30
assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
space = os.statvfs(out)
assert space.f_bavail * space.f_frsize >= 128 << 30
write_new(out / 'resource.json', dict(hostname=socket.gethostname(), node_rank=node_rank, local_gpu_count=2,
    global_gpu_count=8, cpus=32, memory_gib=128, free_bytes=space.f_bavail * space.f_frsize,
    quota='UNVERIFIED', allocated_start_unix=start, deadline_unix=deadline))
probe = out / 'gpu-probe.json'
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '2',
    '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(probe)],
    check=True, timeout=120)
devices = json.loads(probe.read_text())['devices']
run = root / 'outputs/robocasa-eight-gpu-speed-20261102-r1'
job = run / 'S8-distributed-job'
dispatch = job / 'dispatch'
dispatch.mkdir(exist_ok=True)
write_new(out / 'manifest.json', dict(node_rank=node_rank, hostname=socket.gethostname(),
    source_commit=os.environ['TASK_CODE_PIN'], allocated_start_unix=start, probe_path=str(probe), devices=devices))
pipeline = json.loads((root / 'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())
native_env = pipeline['policy_env']
native_env.update(PYTHONPATH=str(code / 'src') + ':' + os.environ['TASK_RLINF'],
                  ROBOCASA_NATIVE_OUTPUT=str(out))
node_profile = dict(node_rank=node_rank, devices=devices, dispatch_root=str(dispatch), native_env=native_env,
    probe_path=str(probe), source_commit=os.environ['TASK_CODE_PIN'], deadline_unix=deadline)
write_new(out / 'node-profile.json', node_profile)
service = subprocess.Popen([sys.executable, '-B', str(code / 'scripts/diagnostics/robocasa_residual_node.py'),
    '--profile', str(out / 'node-profile.json')], start_new_session=True)
try:
    if node_rank == 0:
        expected = [job / f'node-{rank:02d}/manifest.json' for rank in range(4)]
        while not all(path.exists() for path in expected):
            if time.time() > min(start + 300, deadline - 180):
                raise TimeoutError('All four actual node GPU probes must advance before policy loads')
            time.sleep(1)
        os.environ['TASK_NODE_COUNT'] = '4'
        subprocess.run([sys.executable, '-B', str(Path(os.environ['TASK_INPUT']) / 'S8-WORKLOAD.py')],
            check=True, timeout=max(1, deadline - time.time()))
        write_new(dispatch / 'shutdown.json', dict(completed=True, reason='PPO closed with final recovery/evaluation'))
    else:
        while not (dispatch / 'shutdown.json').exists():
            if service.poll() is not None:
                raise RuntimeError(f'Node {node_rank} worker service exited{service.returncode}')
            if time.time() >= deadline:
                raise TimeoutError('Finite node closure before coordinator shutdown')
            time.sleep(1)
except BaseException as error:
    write_new(out / 'error.json', dict(error=f'{type(error).__name__}: {error}'))
    if node_rank == 0 and not (dispatch / 'shutdown.json').exists():
        write_new(dispatch / 'shutdown.json', dict(completed=False, error=f'{type(error).__name__}: {error}'))
    raise
finally:
    if service.poll() is None:
        service.terminate()
    try:
        service.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(service.pid, signal.SIGKILL)
        service.wait()
