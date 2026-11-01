"""Finite8GPU throughput replay; no model computation on a development host."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import episode_video_refs, flora_jobs, sha, write_new
from fluffy_bassoon.meta_rl.robocasa_video import REWARD_CAMERAS
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import protocol

root, code, out = map(Path, (os.environ['TASK_ROOT'], os.environ['TASK_META'], os.environ['TASK_OUT']))
inputs = json.loads((Path(os.environ['TASK_INPUT']) / 'INPUTS.json').read_text())
start, deadline = int(os.environ['TASK_STARTED']), int(os.environ['TASK_STARTED']) + 2700
node_rank, node_count = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
assert node_count == 4 and 0 <= node_rank < 4 and int(os.environ['NPROC_PER_NODE']) == 2
assert socket.gethostname().startswith('job-') and torch.cuda.device_count() == 2
if not os.environ.get('CUDA_VISIBLE_DEVICES'):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
assert int(cpu[0]) / int(cpu[1]) == 32
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 256 << 30
head = lambda path: subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
assert head(code) == os.environ['TASK_CODE_PIN']
assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain'], text=True).strip()
space = os.statvfs(out)
assert space.f_bavail * space.f_frsize >= 128 << 30
with (out / 'write-probe').open('x') as stream:
    stream.write('fresh allocated output write probe\n')
    stream.flush()
    os.fsync(stream.fileno())
(out / 'write-probe').unlink()
write_new(out / 'resource.json', dict(hostname=socket.gethostname(), local_gpu_count=2, local_cpus=32,
    local_memory_gib=256, node_rank=node_rank, node_count=4, aggregate_requested_gpus=8,
    free_bytes=space.f_bavail * space.f_frsize, write_probe_pass=True,
    quota='UNVERIFIED', deadline_unix=deadline))
subprocess.run([sys.executable, '-B', str(code / 'scripts/[train]gpu_probe.py'), '--expected-count', '2',
    '--purpose', 'evaluation', '--baige-container', '--output', str(out / 'gpu-probe.json')],
    check=True, timeout=min(120, max(1, deadline-time.time())))
devices = json.loads((out / 'gpu-probe.json').read_text())['devices']
assert len(devices) == 2 and len({row['uuid'] for row in devices}) == 2
assert all(row['inventory']['memory_total_mib'] >= 90000 for row in devices)

# Carry the successful scorer cases by their exact executable bytes. The PPO
# source may have later metadata/distribution changes without changing these.
qualification_path = Path(os.environ['TASK_CORE_QUALIFICATION']).resolve()
assert qualification_path.is_relative_to(root / 'outputs')
assert sha(qualification_path) == inputs['qualification_receipt_sha256']
qualification = json.loads(qualification_path.read_text())
assert qualification['completed'] and qualification['failures'] == qualification['errors'] == qualification['skipped'] == 0
assert sha(qualification['xml']['path']) == qualification['xml']['sha256']
assert qualification['xml']['sha256'] == inputs['qualified_xml_sha256']
qualified_source = qualification['source_commit']
for path, expected in inputs['qualified_file_sha256'].items():
    assert qualification['files'][path] == expected == sha(code / path)
test_cases = ET.parse(qualification['xml']['path']).getroot().findall('.//testcase')
for name in inputs['required_scorer_tests']:
    matches = [case for case in test_cases if case.attrib['name'] == name
               and case.attrib.get('classname', '').endswith(inputs['required_scorer_test_classnames'][name])]
    assert len(matches) == 1 and not any(matches[0].find(tag) is not None for tag in ('failure', 'error', 'skipped'))
write_new(out / 'carried-core.json', dict(completed=True, tests_carried=inputs['required_scorer_tests'],
    qualification=dict(path=str(qualification_path), sha256=sha(qualification_path)),
    qualified_source_commit=qualified_source, executable_sha256=inputs['qualified_file_sha256'], tests_rerun=0))

official, model = Path(inputs['official_code']), Path(inputs['model'])
assert head(official) == inputs['official_code_commit']
receipt = inputs['model_receipt']
assert sha(receipt['path']) == receipt['sha256']
model_receipt = json.loads(Path(receipt['path']).read_text())
assert model_receipt['completed'] and model_receipt['model_revision'] == inputs['model_revision']
assert Path(model_receipt['model_directory']).resolve() == model.resolve()
episodes_path = Path(inputs['fixture']['path'])
assert sha(episodes_path) == inputs['fixture']['sha256']
episodes = json.loads(episodes_path.read_text())
assert 1 <= len(episodes) == inputs['fixture']['episode_count'] <= 8
assert all(row['camera_order'] == list(REWARD_CAMERAS) for row in episodes)
base_jobs = flora_jobs(episodes, protocol(inputs['fixture']['task']))
assert all(sha(ref['path']) == ref['sha256'] for row in episodes for ref in episode_video_refs(row))
jobs = [dict(base_jobs[slot % len(base_jobs)], id=f'replay-{slot:02d}-{base_jobs[slot % len(base_jobs)]["id"]}')
        for slot in range(64)]
assert len(jobs) == len({row['id'] for row in jobs}) == inputs['requests_per_configuration'] == 64
python = root / 'envs/flora-rewarder-129a2ada-py312-20261101-r1/bin/python'
toolkit = Path(inputs['toolkit'])
write_new(out / 'node-ready.json', dict(node_rank=node_rank, node_count=4,
    hostname=socket.gethostname(), source_commit=os.environ['TASK_CODE_PIN'],
    gpu_probe=dict(path=str(out / 'gpu-probe.json'), sha256=sha(out / 'gpu-probe.json')),
    gpu_uuids=[row['uuid'] for row in devices], model_receipt=receipt,
    qualification_sha256=sha(qualification_path), fixture_sha256=inputs['fixture']['sha256']))
ready_paths = [out.parent / f'node-{rank:02d}/node-ready.json' for rank in range(4)]
while not all(path.exists() for path in ready_paths):
    if time.time() >= deadline:
        raise TimeoutError('Finite deadline before complete eight-GPU resource binding; no partial benchmark')
    time.sleep(2)
ready_rows = [json.loads(path.read_text()) for path in ready_paths]
assert [row['node_rank'] for row in ready_rows] == list(range(4))
assert len({uuid for row in ready_rows for uuid in row['gpu_uuids']}) == 8
for row in ready_rows:
    assert row['node_count'] == 4 and len(row['gpu_uuids']) == 2
    assert row['source_commit'] == os.environ['TASK_CODE_PIN']
    assert row['model_receipt'] == receipt and row['qualification_sha256'] == sha(qualification_path)
    assert row['fixture_sha256'] == inputs['fixture']['sha256']
    assert sha(row['gpu_probe']['path']) == row['gpu_probe']['sha256']
write_new(out / 'cluster-binding.json', dict(completed=True, aggregate_requested_gpus=8,
    node_ready=[dict(path=str(path), sha256=sha(path)) for path in ready_paths]))

# All resource identities are bound before this CPU child masks its own CUDA.
# Measure the real processor expansion once, before any of the eight engines.
geometry_path = out.parent / 'request-geometry.json'
if node_rank == 0:
    geometry_profile = out.parent / 'request-geometry-profile.json'
    write_new(geometry_profile, dict(output=str(geometry_path), jobs=base_jobs,
        model=str(model), source_commit=os.environ['TASK_CODE_PIN'], max_model_len=16384))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
        PYTHONPATH=str(code / 'src') + ':' + str(official),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    try:
        with (out.parent / 'request-geometry.log').open('x') as log:
            check = subprocess.run([str(python), '-B',
                str(code / 'scripts/diagnostics/flora_request_geometry.py'),
                '--profile', str(geometry_profile)], env=environment,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=min(180, max(1, deadline-time.time())))
        if not geometry_path.exists():
            write_new(geometry_path, dict(completed=False, model_constructed=False,
                error=f'Processor preflight exited {check.returncode} without a receipt'))
    except BaseException as error:
        if not geometry_path.exists():
            write_new(geometry_path, dict(completed=False, model_constructed=False,
                error=f'{type(error).__name__}: {error}'))
while not geometry_path.exists():
    if time.time() >= deadline:
        raise TimeoutError('No exact processor/context evidence before finite benchmark deadline')
    time.sleep(1)
geometry = json.loads(geometry_path.read_text())
if geometry.get('completed') is not True:
    raise RuntimeError('Three-view processor preflight failed; no engines or truncated requests')
write_new(out / 'request-geometry-binding.json', dict(completed=True,
    path=str(geometry_path), sha256=sha(geometry_path), context_limit=16384))
children, streams = [], []
local_batch_sizes = inputs['batch_sizes'][node_rank * 2:(node_rank + 1) * 2]
result = dict(completed=False, new_robot_actions=0, batch_sizes=local_batch_sizes,
              node_rank=node_rank, node_count=4, requested_gpus_per_node=2,
              interpretation='Three synchronized native views; throughput replay only, no score/task selection.')
try:
    for rank, (batch_size, device) in enumerate(zip(local_batch_sizes, devices)):
        directory = out / f'batch-{batch_size:02d}'
        directory.mkdir()
        temporary = Path(f'/tmp/fb-3v1-{node_rank}-{rank}')
        temporary.mkdir(exist_ok=True)
        profile = dict(output=str(directory), batch_size=batch_size, jobs=jobs, model=str(model),
            source_commit=os.environ['TASK_CODE_PIN'], fixture_sha256=inputs['fixture']['sha256'], camera_order=list(REWARD_CAMERAS),
            model_receipt=receipt, official_code_commit=inputs['official_code_commit'],
            model_revision=inputs['model_revision'])
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
            raise TimeoutError('Finite benchmark deadline; preserve each completed/failed configuration')
        time.sleep(2)
    result['completed'] = True
except BaseException as error:
    result['error'] = f'{type(error).__name__}: {error}'
    raise
finally:
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
    result.update(return_codes=[child.returncode for child in children],
        all_configurations_succeeded=len(children) == 2 and all(child.returncode == 0 for child in children),
        elapsed_wall_seconds=time.time()-start,
        result_paths=[str(out / f'batch-{size:02d}/result.json') for size in local_batch_sizes])
    write_new(out / 'summary.json', result)

if node_rank == 0:
    summaries = [out.parent / f'node-{rank:02d}/summary.json' for rank in range(4)]
    while not all(path.exists() for path in summaries):
        if time.time() >= deadline:
            raise TimeoutError('Same finite clock ended before all four node summaries; do not claim complete8')
        time.sleep(2)
    records = [json.loads(path.read_text()) for path in summaries]
    assert [row['node_rank'] for row in records] == list(range(4))
    assert [size for row in records for size in row['batch_sizes']] == inputs['batch_sizes']
    write_new(out.parent / 'aggregate-summary.json', dict(completed=all(row['completed'] for row in records),
        all_configurations_succeeded=all(row['all_configurations_succeeded'] for row in records),
        new_robot_actions=0, aggregate_requested_gpus=8,
        node_summaries=[dict(path=str(path), sha256=sha(path)) for path in summaries],
        interpretation='Eight TP1 configurations on four2GPU nodes; throughput replay only, not calibration.'))
