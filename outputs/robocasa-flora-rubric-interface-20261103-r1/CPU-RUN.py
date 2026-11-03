"""Finite allocated-Linux processor audit, preserving any failed attempt."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path('/mowenjie/fluffy-bassoon-2')
run = 'robocasa-flora-rubric-interface-20261103-r1'
code = Path(os.environ['TASK_META'])
assert not (root / 'outputs' / run).exists()
inputs = json.loads((code / 'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json').read_text())
py = root / 'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
           PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
           PYTHONPATH=str(code / 'src') + ':' + inputs['official_code'])
started = time.time()
child = subprocess.Popen([str(py), '-B', str(code / 'outputs' / run / 'CPU-QUALIFY.py'),
                          '--code', str(code)], env=env, start_new_session=True)
try:
    rc = child.wait(timeout=1000)
except BaseException:
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    raise
receipt = dict(source_commit=os.environ['TASK_CODE_PIN'], exit_code=rc,
               seconds=time.time()-started, gpu_seconds=0,
               cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
               memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip())
path = Path('/root/workspace') / (run + '-cpu-exit.json')
path.write_text(json.dumps(receipt, indent=2) + '\n')
if (root / 'outputs' / run).exists():
    (root / 'outputs' / run / 'CPU-EXIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('RUBRIC_CPU_EXIT', json.dumps(receipt), flush=True)
assert rc == 0, f'CPU qualification failed: {rc}; no automatic retry'
