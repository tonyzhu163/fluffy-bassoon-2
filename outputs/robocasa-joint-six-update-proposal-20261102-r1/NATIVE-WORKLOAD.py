"""Finite common-before or S joint job through the existing eight native workers.

Prepared only. The separately supplied, hashed admission must contain the actual
human action decision and immutable five-task calibration/exception references.
"""
import copy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, write_new
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import PPO_TASKS, joint_case_bank

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
binding_path = Path(os.environ['TASK_BINDINGS'])
binding = read_ref(dict(path=str(binding_path), sha256=os.environ['TASK_BINDINGS_SHA256']))
kind = os.environ['TASK_LAUNCH_KIND']
assert kind in ('common', 'S') and binding['stage'] == kind and binding['status'] == 'ADMITTED'
assert binding['source_commit'] == os.environ['TASK_CODE_PIN']
authority = read_ref(binding['human_action_authority'])
assert authority['allow_joint_stage_actions'] is True and authority['maximum_stage_actions'] == 2_500_000
assert authority['direct_human_instruction'] and authority['old_trial_spend_preserved'] is True
assert binding['tasks'] == list(PPO_TASKS) and binding['maximum_iterations'] == 6
assert binding['human_task_authority'] == binding['policy_profile']['calibration_exceptions'].get('PickPlaceCounterToCabinet')
assert read_ref(binding['human_task_authority'])['direct_human_instruction']
node_rank, start = int(os.environ['RANK']), int(os.environ['TASK_STARTED'])
worker_seconds = 7200 if kind == 'common' else 21600
slice_seconds = worker_seconds - 60
closure = 300
deadline = start + slice_seconds - closure
ram_gib = 256
action_cap = None  # Uncapped, actions still journaled (human, 2026-11-02).
output_gib = 48 if kind == 'common' else 80
assert os.environ['WORLD_SIZE'] == '4' and os.environ['NPROC_PER_NODE'] == '2' and 0 <= node_rank < 4
assert socket.gethostname().startswith('job-') and torch.cuda.device_count() == 2
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) == 64 and int(Path('/sys/fs/cgroup/memory.max').read_text()) == ram_gib << 30
assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == binding['source_commit']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
# Actual namespace/output sizing and the protected old recoveries are checked
# before admission. This child verifies that the admitted remaining space exists.
storage = read_ref(binding['storage_proof'])
assert storage['completed'] is True and storage['write_probe_pass'] is True
assert storage['old_recoveries_preserved'] is True and storage['stage_supported_upper_bytes'] <= 256 << 30
free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
assert free >= storage['remaining_output_reservation_bytes'] >= output_gib << 30
write_probe = out / 'write-probe'
with write_probe.open('x') as stream:
    stream.write('fresh native joint namespace\n')
    stream.flush()
    os.fsync(stream.fileno())
write_probe.unlink()
probe = out / 'gpu-probe.json'
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '2',
    '--purpose', 'training', '--baige-container', '--baige-exclusive-training', '--output', str(probe)],
    check=True, timeout=min(120, max(1, deadline - time.time())))
devices = json.loads(probe.read_text())['devices']
run = root / 'outputs' / os.environ['TASK_RUN']
job, dispatch = run / 'job', run / 'job/dispatch'
dispatch.mkdir(exist_ok=True)
write_new(out / 'manifest.json', dict(node_rank=node_rank, hostname=socket.gethostname(),
    source_commit=binding['source_commit'], allocated_start_unix=start, probe_path=str(probe), devices=devices,
    cpus=64, ram_gib=ram_gib, free_bytes=free, quota=storage.get('quota', 'UNVERIFIED'),
    human_action_authority=binding['human_action_authority'], binding_sha256=os.environ['TASK_BINDINGS_SHA256']))
manifests = [job / f'node-{rank:02d}/manifest.json' for rank in range(4)]
while not all(path.exists() for path in manifests):
    if any((job / f'node-{rank:02d}/error.json').exists() for rank in range(4)):
        raise RuntimeError('An owned native node failed before its manifest')
    if time.time() >= min(start + 300, deadline):
        raise TimeoutError('All four actual allocated nodes must advance before policy construction')
    time.sleep(1)
manifests = [json.loads(path.read_text()) for path in manifests]
assert [m['node_rank'] for m in manifests] == list(range(4))
assert all(m['source_commit'] == binding['source_commit'] and m['binding_sha256'] == os.environ['TASK_BINDINGS_SHA256'] for m in manifests)
global_devices = [dict(d, node_rank=rank, probe_path=m['probe_path']) for rank, m in enumerate(manifests) for d in m['devices']]
assert len(global_devices) == len({d['uuid'] for d in global_devices}) == 8
deadline = min(m['allocated_start_unix'] for m in manifests) + slice_seconds - closure
profile = copy.deepcopy(binding['policy_profile'])
execution = json.loads((code/'configs/native_ppo/robocasa_residual_execution.json').read_text())
profile['rollout_lanes_per_gpu'] = execution['rollout_lanes_per_gpu']
assert profile['task'] == 'joint_five' and profile['tasks'] == list(PPO_TASKS)
assert profile['maximum_iterations'] == 6 and set(profile['baseline_validations']) == set(PPO_TASKS)
assert not any(profile.get(key) for key in ('resume', 'rollout_migration', 'checkpoint', 'prior_dev_evaluations'))
native_env = dict(PYTHONPATH=str(code / 'src') + ':' + os.environ['TASK_RLINF'], RLINF_PATH=os.environ['TASK_RLINF'],
    ROBOCASA_ASSETS_PATH=str(root / 'setups/robocasa365-native-20261101-r1/assets'),
    OPENPI_DATA_HOME=str(root / 'setups/robocasa365-native-20261101-r1/cache/.cache/openpi'),
    MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', ROBOCASA_NATIVE_OUTPUT=str(out),
    ROBOCASA_PI05_TORCH_MODEL=profile['initializer']['model_directory'],
    ROBOCASA_PI05_NATIVE_NORM=profile['initializer']['norm_path'])
write_new(out / 'node-profile.json', dict(node_rank=node_rank, devices=devices, dispatch_root=str(dispatch),
    native_env=native_env, probe_path=str(probe), source_commit=binding['source_commit'], deadline_unix=deadline))
service = subprocess.Popen([sys.executable, '-B', str(code / 'scripts/diagnostics/robocasa_residual_node.py'),
    '--profile', str(out / 'node-profile.json')], start_new_session=True)
def halt(signum, frame):
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, halt)
signal.signal(signal.SIGINT, halt)
try:
    if node_rank == 0:
        profile.update(mode='common_before' if kind == 'common' else 'train', arm='S',
            source_commit=binding['source_commit'], output=str(run / 'policy'), monitoring_root=str(run / 'monitoring'),
            accounting_root=str(run / 'actions'), attempt=os.environ['TASK_RUN'], episode_id_namespace='joint-',
            allocated_start_unix=start, deadline_unix=deadline, scheduler_seconds=slice_seconds, closure_seconds=closure,
            maximum_all_in_actions=action_cap, rollout_lanes_per_gpu=8, rollout_devices=global_devices, parallel_controller_cpu_offload=True,
            actual_gpu_probe=dict(path=str(probe), sha256=sha(probe)), distributed_rollout_root=str(dispatch),
            gpu_allocation_starts_unix=[m['allocated_start_unix'] for m in manifests for _ in range(2)])
        if kind == 'S':
            common = read_ref(binding['common_final'])
            assert common['completed'] is True and common['policy_updates'] == 0 and common['ID_episodes'] == common['OOD_episodes'] == 250
            profile.update(stopping='fixed', initial_policy_fingerprint=binding['initial_policy_fingerprint'],
                initial_dev_evaluation=binding['initial_dev_evaluation'],
                training_cases=[case for iteration in range(1, 7) for case in joint_case_bank('train', iteration=iteration)])
        write_new(out / 'policy-profile.json', profile)
        environment = dict(os.environ, **native_env)
        environment.update(CUDA_VISIBLE_DEVICES=global_devices[0]['visibility_token'], LYCO_BAIGE_GPU_MAP=str(probe),
            LYCO_BAIGE_ISOLATED_UUID=global_devices[0]['uuid'], MUJOCO_EGL_DEVICE_ID='0')
        subprocess.run([sys.executable, '-B', str(code / 'scripts/diagnostics/robocasa_residual_ppo.py'),
            '--profile', str(out / 'policy-profile.json')], env=environment, check=True,
            timeout=max(1, deadline - time.time()))
        marker = run / 'policy' / ('COMMON-FINAL.json' if kind == 'common' else 'EVALUATION-COMPLETE.json')
        assert json.loads(marker.read_text())['completed'] is True
        actual_bytes = sum(path.stat().st_size for path in run.rglob('*') if path.is_file())
        assert actual_bytes <= output_gib << 30
        write_new(run / 'COMPLETED.json', dict(completed=True, stage=kind, actual_output_bytes=actual_bytes,
            result=dict(path=str(marker), sha256=sha(marker)), new_scope_authority=binding['human_action_authority']))
        write_new(dispatch / 'shutdown.json', dict(completed=True))
    else:
        while not (dispatch / 'shutdown.json').exists():
            if service.poll() is not None:
                raise RuntimeError('Owned native node service failed')
            if time.time() >= deadline:
                raise TimeoutError('Finite native node closure')
            time.sleep(1)
        if json.loads((dispatch / 'shutdown.json').read_text())['completed'] is not True:
            raise RuntimeError('Coordinator failed; preserve partial evidence and action charge')
except BaseException as error:
    write_new(out / 'error.json', dict(error=f'{type(error).__name__}: {error}'))
    if node_rank == 0 and not (dispatch / 'shutdown.json').exists():
        write_new(dispatch / 'shutdown.json', dict(completed=False, error=str(error)))
    raise
finally:
    if service.poll() is None:
        service.terminate()
    try:
        service.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(service.pid, signal.SIGKILL)
        service.wait()
