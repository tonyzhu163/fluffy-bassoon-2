"""CPU launcher metadata invariants; run only in an actual Linux allocation."""
import json
from types import SimpleNamespace

import pytest

from fluffy_bassoon.meta_rl import vla_source40_cpu_qualification as module


def fake_torch(count=0, initialized=False):
    return SimpleNamespace(version=SimpleNamespace(cuda='12.8'), __version__='qualified', __file__='env/torch.py',
        cuda=SimpleNamespace(device_count=lambda:count, is_initialized=lambda:initialized))


def test_zero_device_attestation_happens_before_mask_and_context_initialization():
    env = {'CUDA_VISIBLE_DEVICES':'scheduler-token', 'NVIDIA_VISIBLE_DEVICES':'void'}
    attestation = module.require_zero_gpu(fake_torch(), environment=env)
    assert attestation['device_count_before_mask'] == 0
    assert attestation['environment_before_mask']['CUDA_VISIBLE_DEVICES'] == 'scheduler-token'
    assert env['CUDA_VISIBLE_DEVICES'] == ''
    for torch in (fake_torch(count=1), fake_torch(initialized=True)):
        before = {'CUDA_VISIBLE_DEVICES':'0'}
        with pytest.raises(ValueError):
            module.require_zero_gpu(torch, environment=before)
        assert before['CUDA_VISIBLE_DEVICES'] == '0'


def test_cpu_only_torch_build_cannot_manufacture_zero_gpu_proof():
    torch = fake_torch()
    torch.version.cuda = None
    with pytest.raises(ValueError, match='CUDA-capable'):
        module.require_zero_gpu(torch, environment={})


def test_test_paths_are_reviewed_local_python_files_and_required_metadata_suite_remains(tmp_path):
    for name in (*module.DEFAULT_TESTS, 'tests/test_future_competence.py'):
        path = tmp_path/name
        path.parent.mkdir(exist_ok=True)
        path.write_text('# metadata-only fixture\n')
    selected = module.validate_tests(tmp_path, ['tests/test_future_competence.py'])
    assert set(module.DEFAULT_TESTS).issubset(selected)
    assert selected[-1] == 'tests/test_future_competence.py'
    with pytest.raises(ValueError, match='reviewed'):
        module.validate_tests(tmp_path, ['../test_arbitrary.py'])


def test_cpu_clock_preserves_actual_overrun_and_never_charges_gpu(tmp_path):
    module.write(tmp_path/'cpu-worker-clock.json', dict(started_unix=1000., status='started', zero_gpu_attested=True))
    first = module.finish(tmp_path, 1, now=1100.)
    assert first['charged_cpu_worker_seconds'] == 160 and first['charged_gpu_seconds'] == 0
    last = module.finish(tmp_path, 124, now=2800.)
    assert last['charged_cpu_worker_seconds'] == 1860 and last['cpu_time_overrun']
    assert len(json.loads((tmp_path/'cpu-worker-clock.json').read_text())['closure_history']) == 2


def test_failed_allocation_admission_does_not_claim_zero_gpu_time(tmp_path):
    module.write(tmp_path/'cpu-worker-clock.json', dict(started_unix=1000., status='started', zero_gpu_attested=False))
    result = module.finish(tmp_path, 1, now=1100.)
    assert result['charged_gpu_seconds'] is None
