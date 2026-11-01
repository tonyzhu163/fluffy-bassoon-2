"""Run only inside the actual eight-GPU scheduler allocation."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import flora_jobs, sha, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
start = int(os.environ['TASK_STARTED'])
deadline = start + 2820
assert socket.gethostname().startswith('job-') and torch.cuda.device_count() == 8
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(8)))
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) == 128
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 1024 << 30
assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
space = os.statvfs(out)
assert space.f_bavail * space.f_frsize >= 128 << 30
write_new(out / 'resource.json', dict(hostname=socket.gethostname(), gpu_count=8, cpus=128,
    memory_gib=1024, free_bytes=space.f_bavail * space.f_frsize, quota='UNVERIFIED', deadline_unix=deadline))
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '8',
    '--purpose', 'evaluation', '--baige-container', '--output', str(out / 'gpu-probe.json')], check=True, timeout=120)
subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', str(code / 'tests/test_residual_parallel.py'),
    str(code / 'tests/test_flora_terminal_worker.py'), '--junitxml=' + str(out / 'affected-core.xml')],
    check=True, timeout=120)
devices = json.loads((out / 'gpu-probe.json').read_text())['devices']
episodes_path = root / 'outputs/robocasa-native-residual-headroom-q4-20261101-r1/OpenDrawer/episodes.json'
assert sha(episodes_path) == 'f912ddb229bd836f53ca37b78815bae3c8fb1b457a6dbd39db318277ee6a2999'
base_jobs = flora_jobs(json.loads(episodes_path.read_text()), protocol('OpenDrawer'))
assert len(base_jobs) == 8
jobs = [dict(job, id=f'replay-{repeat:02d}-{job["id"]}') for repeat in range(8) for job in base_jobs]
python = root / 'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
official = root / 'deps/nvidia-flora-129a2ada-20261101-r1/repo'
toolkit = root / 'outputs/flora-ppo-official-inference-q7-20261101-r1/toolkit-site/nvidia/cu13'
children, streams = [], []
try:
    for rank, (batch_size, device) in enumerate(zip([1, 2, 4, 6, 8, 12, 16, 24], devices)):
        directory = out / f'batch-{batch_size:02d}'
        directory.mkdir()
        temporary = Path(f'/tmp/fb-{rank}')
        temporary.mkdir(exist_ok=True)
        profile = dict(output=str(directory), batch_size=batch_size, jobs=jobs,
            model=str(root / 'inputs/flora-rewarder-031299e4-20261101-r1'),
            source_commit=os.environ['TASK_CODE_PIN'], fixture_sha256=sha(episodes_path))
        write_new(directory / 'profile.json', profile)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device['visibility_token'],
            LYCO_BAIGE_GPU_MAP=str(out / 'gpu-probe.json'), LYCO_BAIGE_ISOLATED_UUID=device['uuid'],
            PYTHONPATH=str(code / 'src') + ':' + str(official),
            CUDA_HOME=str(toolkit), CUDA_PATH=str(toolkit), CUDACXX=str(toolkit / 'bin/nvcc'),
            PATH=str(toolkit / 'bin') + ':' + str(python.parent) + ':' + os.environ['PATH'],
            LD_LIBRARY_PATH=str(toolkit / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
            LIBRARY_PATH=str(toolkit / 'lib') + ':' + os.environ.get('LIBRARY_PATH', ''),
            TMPDIR=str(temporary), XDG_CACHE_HOME=str(directory / 'cache'),
            TORCHINDUCTOR_CACHE_DIR=str(directory / 'cache/torch'),
            FLASHINFER_WORKSPACE_BASE=str(directory / 'cache/flashinfer'),
            VLLM_CACHE_ROOT=str(directory / 'cache/vllm'), TRITON_CACHE_DIR=str(directory / 'cache/triton'),
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', MAX_JOBS='4',
            OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        stream = (directory / 'console.log').open('x')
        streams.append(stream)
        children.append(subprocess.Popen([str(python), '-B', str(code / 'scripts/diagnostics/flora_batch_benchmark.py'),
            '--profile', str(directory / 'profile.json')], env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True))
    while any(child.poll() is None for child in children):
        if time.time() >= deadline:
            raise TimeoutError('Finite benchmark deadline; preserve each completed or failed configuration')
        time.sleep(2)
    write_new(out / 'summary.json', dict(completed=True, return_codes=[child.returncode for child in children],
        new_robot_actions=0, batch_sizes=[1, 2, 4, 6, 8, 12, 16, 24], elapsed_wall_seconds=time.time() - start,
        interpretation='Throughput replay only; failed configurations retained and never silently replaced.'))
finally:
    import signal
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    for stream in streams:
        stream.close()
