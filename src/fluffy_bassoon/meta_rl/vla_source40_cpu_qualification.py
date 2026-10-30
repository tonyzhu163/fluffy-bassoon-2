"""Zero-GPU allocated metadata tests and optional immutable corpus capture.

No scheduler, learner, policy construction, inference, training or simulator.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import sysconfig
import time

FORMAT = 'source40_zero_gpu_cpu_qualification_v1'
CPU_ENV = Path('/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde')
DEFAULT_TESTS = ('tests/test_vla_source40_topup_worker.py',
    'tests/test_vla_source40_topup_assembly.py', 'tests/test_vla_source40_topup.py',
    'tests/test_vla_source40_cpu_qualification.py')


def write(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def cgroup_resources():
    paths = ('/proc/self/cgroup', '/sys/fs/cgroup/cpu.max', '/sys/fs/cgroup/cpu.stat',
        '/sys/fs/cgroup/cpuset.cpus.effective', '/sys/fs/cgroup/memory.max',
        '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/cpu/cpu.cfs_quota_us',
        '/sys/fs/cgroup/cpu/cpu.cfs_period_us', '/sys/fs/cgroup/memory/memory.limit_in_bytes')
    raw = {}
    for path in paths:
        try:
            raw[path] = Path(path).read_text().strip()
        except (OSError, PermissionError):
            raw[path] = None
    cores, memory = None, None
    try:
        quota, period = raw['/sys/fs/cgroup/cpu.max'].split()
        cores = None if quota == 'max' else int(quota)/int(period)
    except (TypeError, ValueError, AttributeError):
        try:
            quota = int(raw['/sys/fs/cgroup/cpu/cpu.cfs_quota_us'])
            cores = quota/int(raw['/sys/fs/cgroup/cpu/cpu.cfs_period_us']) if quota > 0 else None
        except (TypeError, ValueError):
            pass
    for key in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            if raw[key] not in (None, 'max'):
                memory = int(raw[key])
                break
        except ValueError:
            pass
    return dict(raw=raw, effective_cpu_quota_cores=cores, memory_limit_bytes=memory,
        process_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        requested=dict(cpu_cores=4, memory_bytes=16*1024**3, gpu_count=0),
        cgroup_matches_request=dict(cpu=None if cores is None else cores == 4,
            memory=None if memory is None else memory == 16*1024**3))


def require_zero_gpu(torch, *, environment=None):
    """Observe allocation visibility BEFORE applying our CPU-only mask."""
    environment = os.environ if environment is None else environment
    if not getattr(torch.version, 'cuda', None):
        raise ValueError('Use the qualified CUDA-capable cp310 Torch build to attest zero allocated devices')
    if torch.cuda.is_initialized():
        raise ValueError('CUDA was initialized before the zero-GPU allocation check')
    before = {k: environment.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'PYTHONPATH')}
    count = torch.cuda.device_count()
    if count != 0 or torch.cuda.is_initialized():
        raise ValueError('CPU qualification requires zero CUDA-visible devices before any mask or initialization')
    environment['CUDA_VISIBLE_DEVICES'] = ''
    return dict(passed=True, device_count_before_mask=count, cuda_initialized_before_mask=False,
        environment_before_mask=before, cuda_visible_devices_after_mask='', torch_version=str(torch.__version__),
        torch_cuda_build=torch.version.cuda, torch_module_path=str(torch.__file__),
        device_nodes=sorted(str(p) for p in Path('/dev').glob('nvidia*')),
        rule='Pre-mask allocation check; no GPU-count evidence manufactured by our mask')


def validate_tests(code, supplied):
    tests = list(dict.fromkeys([*DEFAULT_TESTS, *supplied]))
    for name in tests:
        path = Path(name)
        if (path.is_absolute() or '..' in path.parts or len(path.parts) != 2 or path.parts[0] != 'tests'
                or not path.name.startswith('test_') or path.suffix != '.py' or not (Path(code)/path).is_file()):
            raise ValueError('Additional CPU tests must be existing reviewed tests/test_*.py files')
    return tests


def finish(root, exit_code, *, now=None):
    path = Path(root)/'cpu-worker-clock.json'
    data = json.loads(path.read_text())
    ended = time.time() if now is None else float(now)
    # Final shell teardown is included. Re-closing preserves the original end
    # evidence and adds the final observed charge rather than erasing it.
    data.setdefault('closure_history', []).append(dict(ended_unix=ended, exit_code=int(exit_code)))
    data.update(status='finished', ended_unix=ended, exit_code=int(exit_code),
        charged_cpu_worker_seconds=math.ceil(max(0., ended-data['started_unix']))+60,
        charged_gpu_seconds=0 if data.get('zero_gpu_attested') is True else None)
    data['cpu_time_overrun'] = data['charged_cpu_worker_seconds'] > 1800
    write(path, data)
    return data


def run(a):
    code = Path(__file__).resolve().parents[3]
    if (sys.platform != 'linux' or not socket.gethostname().startswith('job-')
            or socket.gethostname() != a.expected_hostname or sys.version_info[:2] != (3, 10)
            or Path(sys.prefix).resolve() != CPU_ENV.resolve()):
        raise ValueError('Require the exact qualified cp310 runtime in the identified Baige CPU job')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() != a.source
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=code, text=True).strip()):
        raise ValueError('Require the exact clean reviewed CPU qualification source')
    if (a.output.name != a.run or not a.run or any(not(c.isalnum() or c == '-') for c in a.run)
            or sha(a.plan) != a.plan_sha256 or not a.ledger.is_file()
            or any(bad in os.environ.get('PYTHONPATH', '') for bad in ('cu128-overlay', 'python3.11', 'cp311'))
            or not math.isfinite(a.started_unix) or not 0 <= time.time()-a.started_unix < 1740):
        raise ValueError('CPU run/plan/runtime/deadline binding differs')
    tests = validate_tests(code, a.test_file)
    a.output.mkdir(parents=True, exist_ok=False)  # No overwrite/retry/reselection.
    clock = dict(format=FORMAT, source_commit=a.source, run_id=a.run, hostname=a.expected_hostname,
        started_unix=a.started_unix, status='started', maximum_cpu_worker_seconds=1800,
        execution_seconds=1740, startup_allowance_seconds=60, charged_gpu_seconds=None,
        zero_gpu_attested=False,
        interrupted_charge_cpu_seconds=1800, additional_simulator_actions=0)
    write(a.output/'cpu-worker-clock.json', clock)
    contract = dict(format=FORMAT, source_commit=a.source, run_id=a.run, hostname=a.expected_hostname,
        plan_path=str(a.plan.resolve()), plan_file_sha256=a.plan_sha256, canonical_ledger=str(a.ledger.resolve()),
        tests=tests, capture_assemble=a.capture_assemble, model_construction=False, inference=False,
        optimizer_updates=0, simulator_actions=0, requested_gpu_count=0,
        runtime=sys.executable, python_abi=sysconfig.get_config_var('SOABI'))
    write(a.output/'contract.json', contract)
    code_rc = 1
    phases = []

    def command(name, argv):
        remaining = a.started_unix + 1700 - time.time()
        if remaining < 1:
            raise TimeoutError('CPU qualification deadline before the next metadata phase')
        receipt = dict(phase=name, command=argv, started_unix=time.time(), timeout_seconds=remaining)
        write(a.output/(name+'-intent.json'), receipt)
        try:
            with (a.output/(name+'.log')).open('x') as log:
                result = subprocess.run(argv, cwd=code, env=dict(os.environ), stdout=log,
                    stderr=subprocess.STDOUT, timeout=remaining, check=False)
            receipt.update(exit_code=result.returncode, completed=True, ended_unix=time.time())
        except BaseException as error:
            receipt.update(completed=False, ended_unix=time.time(), error=repr(error))
            write(a.output/(name+'-receipt.json'), receipt)
            raise
        write(a.output/(name+'-receipt.json'), receipt)
        phases.append(receipt)
        print(json.dumps(dict(phase=name, exit_code=result.returncode,
            elapsed_seconds=receipt['ended_unix']-receipt['started_unix'])), flush=True)
        if result.returncode:
            raise RuntimeError(f'CPU metadata phase {name} failed; preserve its original log/output')

    try:
        import torch
        attestation = require_zero_gpu(torch)
        attestation.update(hostname=a.expected_hostname, python=sys.executable, sys_prefix=sys.prefix,
            python_abi=sysconfig.get_config_var('SOABI'), resources=cgroup_resources())
        write(a.output/'zero-gpu-attestation.json', attestation)
        clock.update(zero_gpu_attested=True, charged_gpu_seconds=0)
        write(a.output/'cpu-worker-clock.json', clock)
        print(json.dumps(dict(zero_gpu_pre_mask=True, hostname=a.expected_hostname,
            measured_cpu_quota=attestation['resources']['effective_cpu_quota_cores'],
            measured_memory_limit=attestation['resources']['memory_limit_bytes'])), flush=True)
        command('metadata-tests', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            '--junitxml='+str(a.output/'junit.xml'), *tests])
        outputs = {}
        if a.capture_assemble:
            base = [sys.executable, '-m', 'fluffy_bassoon.meta_rl.vla_source40_topup_assembly']
            command('capture', [*base, 'capture', '--plan', str(a.plan), '--plan-sha256', a.plan_sha256,
                '--ledger', str(a.ledger), '--output', str(a.output/'snapshot')])
            admission = a.output/'snapshot/admission.json'
            command('assemble', [*base, 'assemble', '--admission', str(admission),
                '--admission-sha256', sha(admission), '--output', str(a.output/'assembly')])
            outputs = {name:dict(path=str(path.resolve()), sha256=sha(path)) for name,path in (
                ('admission', admission), ('assembly', a.output/'assembly/assembly.json'),
                ('fitting_inputs', a.output/'assembly/topup-inputs.json'))}
        if torch.cuda.is_initialized() or torch.cuda.device_count() != 0:
            raise RuntimeError('CUDA state changed during CPU-only qualification')
        write(a.output/'index.json', dict(contract, completed=True, phases=phases, outputs=outputs,
            zero_gpu_attestation_sha256=sha(a.output/'zero-gpu-attestation.json'),
            junit_sha256=sha(a.output/'junit.xml'), no_gpu_budget_consumed=True,
            cuda_uninitialized_at_completion=True, allocated_gpu_count=0))
        code_rc = 0
        return 0
    except BaseException as error:
        write(a.output/'failure.json', dict(error=repr(error), completed=False, phases=phases,
            partial_outputs_preserved=True, gpu_seconds=0 if clock['zero_gpu_attested'] else None,
            zero_gpu_attested=clock['zero_gpu_attested'], simulator_actions=0))
        raise
    finally:
        finish(a.output, code_rc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    r = sub.add_parser('run')
    for n in ('output', 'plan', 'ledger'):
        r.add_argument('--'+n, type=Path, required=True)
    for n in ('source', 'run', 'plan-sha256', 'expected-hostname'):
        r.add_argument('--'+n, required=True)
    r.add_argument('--started-unix', type=float, required=True)
    r.add_argument('--capture-assemble', action='store_true')
    r.add_argument('--test-file', action='append', default=[])
    f = sub.add_parser('finish')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--exit-code', type=int, required=True)
    a = p.parse_args(argv)
    if a.operation == 'finish':
        print(json.dumps(finish(a.output, a.exit_code)))
        return 0
    return run(a)


if __name__ == '__main__':
    raise SystemExit(main())
