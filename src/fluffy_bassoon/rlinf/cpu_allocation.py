"""Observe cgroup limits and accelerator visibility inside the allocated CPU step."""
import os
from pathlib import Path


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
        raise ValueError('Use the qualified CUDA-capable base Torch build to attest zero allocated devices')
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
