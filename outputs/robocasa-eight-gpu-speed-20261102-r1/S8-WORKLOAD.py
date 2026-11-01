"""Actual allocated eight-worker baseline, preserving the complete PPO batch."""
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
run = root / 'outputs/robocasa-eight-gpu-speed-20261102-r1'
old = root / 'outputs/robocasa-residual-approved-20261101-r1'
start = int(os.environ['TASK_STARTED'])
deadline = start + 14040
assert socket.gethostname().startswith('job-') and torch.cuda.device_count() == 8
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(8)))
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) == 128
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 512 << 30
assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
space = os.statvfs(out)
assert space.f_bavail * space.f_frsize >= 128 << 30
write_new(out / 'resource.json', dict(hostname=socket.gethostname(), gpu_count=8, cpus=128,
    memory_gib=512, free_bytes=space.f_bavail * space.f_frsize, quota='UNVERIFIED', deadline_unix=deadline))
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '8',
    '--purpose', 'training', '--baige-container', '--baige-exclusive-training',
    '--output', str(out / 'gpu-probe.json')], check=True, timeout=120)
# No policy loading or physics until the two new numerical and four affected
# scorer core cases pass. The benchmark may carry this exact result later.
subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', str(code / 'tests/test_residual_parallel.py'),
    str(code / 'tests/test_flora_terminal_worker.py'), '--junitxml=' + str(out / 'affected-core.xml')],
    check=True, timeout=120)
suite = ET.parse(out / 'affected-core.xml').getroot().find('testsuite')
assert suite is not None and int(suite.attrib['tests']) == 6
assert all(int(suite.attrib[key]) == 0 for key in ('failures', 'errors', 'skipped'))
write_new(out / 'core-qualification.json', dict(completed=True, tests=6, failures=0, errors=0, skipped=0,
    xml=dict(path=str(out / 'affected-core.xml'), sha256=sha(out / 'affected-core.xml')),
    source_commit=os.environ['TASK_CODE_PIN']))
paused = json.loads((old / 'S-PAUSED-FOR-EIGHT-GPU.json').read_text())
assert paused['job'] == 'job-wjwi6vuhvwmq' and paused['optimizer_updates'] == 0
assert paused['prior_training_actions_conservatively_charged'] == 13350
journal_rows = [json.loads(path.read_text()) for path in (old / 'S-actions').glob('rank-*/*.json')]
prior_charge = sum(sum(row.get('executed_or_conservative', row['requested']) for row in v['calls']) for v in journal_rows)
assert prior_charge == paused['prior_training_actions_conservatively_charged']
checkpoint = old / 'S/boundary-000.pt'
saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert saved['iteration'] == 0 and saved['stage'] == 'boundary'
assert not saved['optimizer']['state'] and saved['environment']['cursor'] == 0
assert saved['contract']['source_commit'] == 'd365878e38162c0ff750150b4fa014b6464b8869'
assert not list((old / 'S').glob('update-*.json'))
probe = out / 'gpu-probe.json'
devices = json.loads(probe.read_text())['devices']
migration = dict(kind='eight_gpu_rollout_only', checkpoint_sha256=sha(checkpoint),
    original_contract=saved['contract'], from_source_commit=saved['contract']['source_commit'],
    to_source_commit=os.environ['TASK_CODE_PIN'], prior_partial_training_actions_charged=prior_charge,
    direct_human_direction='Start eight-GPU PPO now through the same batch pipeline',
    incomplete_legacy_rollout_used_for_SGD=False)
write_new(out / 'migration.json', migration)
pipeline = json.loads((old / 'S-job/pipeline-profile.json').read_text())
profile = copy.deepcopy(pipeline['training'])
initial = old / 'common-before/initial_DEV-000/evaluation.json'
profile.update(source_commit=os.environ['TASK_CODE_PIN'], output=str(run / 'S8'),
    monitoring_root=str(run / 'monitoring/S8'), accounting_root=str(run / 'S8-actions'), attempt='S8-train-r1',
    allocated_start_unix=start, deadline_unix=deadline, scheduler_seconds=14340,
    maximum_all_in_actions=971600-prior_charge, rollout_devices=devices,
    actual_gpu_probe=dict(path=str(probe), sha256=sha(probe)),
    core_qualification=[dict(path=str(out / 'core-qualification.json'), sha256=sha(out / 'core-qualification.json'))],
    baseline_validation=pipeline['calibration'], calibration_exception=pipeline['calibration_exception'],
    independent_stopping=True, initial_dev_evaluation=dict(path=str(initial), sha256=sha(initial)),
    resume=str(checkpoint), resume_sha256=sha(checkpoint),
    rollout_migration=dict(path=str(out / 'migration.json'), sha256=sha(out / 'migration.json')),
    first_iteration_wall_ceiling_seconds=3600, dev_evaluation_wall_ceiling_seconds=600,
    final_evaluation_wall_reserve_seconds=1200)
write_new(out / 'training-profile.json', profile)
environment = dict(os.environ, **pipeline['policy_env'])
environment.update(PYTHONPATH=str(code / 'src') + ':' + os.environ['TASK_RLINF'],
    CUDA_VISIBLE_DEVICES=devices[0]['visibility_token'], LYCO_BAIGE_GPU_MAP=str(probe),
    LYCO_BAIGE_ISOLATED_UUID=devices[0]['uuid'], MUJOCO_EGL_DEVICE_ID='0',
    ROBOCASA_NATIVE_OUTPUT=str(out), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
print(json.dumps(dict(event='eight_worker_simulator_residual_PPO_start', workers=8,
    complete_batch_episodes=64, episodes_per_worker=8, global_chunk_decisions=960,
    initializer='Human30075000', resume_iteration=0, prior_partial_actions_charged=prior_charge)), flush=True)
subprocess.run([sys.executable, '-B', str(code / 'scripts/diagnostics/robocasa_residual_ppo.py'),
    '--profile', str(out / 'training-profile.json')], env=environment, check=True,
    timeout=max(1, deadline-time.time()))
