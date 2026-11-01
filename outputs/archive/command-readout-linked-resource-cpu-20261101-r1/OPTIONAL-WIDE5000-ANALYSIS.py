"""Read-only fixed5000 diagnosis on allocated Linux; CPU head, no VLA/physics/SGD.

Compare raw and actuator-clipped gripper errors on every original cached row.
The interval oracle is a per-slot lower bound, not a realizable policy.
Run in a separate process; retain scheduler GPU visibility unchanged.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('endpoint', 'cache', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--endpoint-sha256', required=True)
    p.add_argument('--cache-sha256', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--visible-gpu-count', type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    assert 0 < a.deadline - time.time() <= 120 and not a.output.exists()

    def guard():
        if time.time() + 2 >= a.deadline:
            raise TimeoutError('Owner analysis deadline; no result published')

    import torch
    from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual
    assert not torch.cuda.is_initialized()
    assert torch.cuda.device_count() == a.visible_gpu_count
    torch.set_num_threads(2)
    guard()
    assert sha(a.endpoint) == a.endpoint_sha256
    assert sha(a.cache) == a.cache_sha256
    sealed = json.loads(a.cache.read_text())
    canonical = hashlib.sha256(json.dumps(sealed, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert canonical == '5f8ada37185fb7d5add69f9eb17cb032fc63bbc90f562fea825d161ca2a9d7dd'
    saved = torch.load(a.endpoint, map_location='cpu', weights_only=False)
    assert saved['kind'] == 'action' and saved['iteration'] == 5000
    assert saved['contract']['complete_cache_sha256'] == canonical
    assert saved['contract']['source_commit'] == '400eed69ead8b8672686c62f33ed5531f6f60558'
    assert sealed['complete'] and sealed['source_eligible'] is False
    assert sealed['bank']['outcome_evidence']['phone-04']['success'] is False
    with torch.random.fork_rng(devices=[]):
        head = WideActionResidual.from_config(saved['config']).eval().requires_grad_(False)
    head.load_state_dict(saved['policy'])
    assert head.contract_sha256() == '0fa57d4fa9e380353e304cc66d4f842c8ff0c6b08e0df4820e2c998b64ddfdde'
    arm, grip = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], [6, 13]
    groups, seen = {}, set()

    def add(key, split, base, target, prediction, ideal, native_actions):
        g = groups.setdefault(key, dict(split=split, submitted_slots=0,
            historical_native_actions=0, rows=0, models={}))
        assert g['split'] == split
        g['submitted_slots'] += len(target)
        g['historical_native_actions'] += native_actions
        g['rows'] += 1
        for name, values in [('native', base), ('wide5000', prediction),
                             ('interval_lower_bound', ideal)]:
            m = g['models'].setdefault(name, dict(raw14_SSE=0., arm12_SSE=0.,
                gripper2_SSE=0., clipped_gripper2_SSE=0., clipped_gripper_values=0))
            error = (values - target).square().double()
            m['raw14_SSE'] += float(error.sum())
            m['arm12_SSE'] += float(error[:, arm].sum())
            m['gripper2_SSE'] += float(error[:, grip].sum())
            m['clipped_gripper2_SSE'] += float((values[:, grip].clamp(0, 1)
                - target[:, grip].clamp(0, 1)).square().double().sum())
            m['clipped_gripper_values'] += int(((values[:, grip] < 0)
                                               | (values[:, grip] > 1)).sum())

    rows = sealed['records']
    assert len(rows) == 90
    with torch.inference_mode():
        for row in rows:
            guard()
            identity, split = row['context']['teacher_id'], row['context']['split']
            key = (identity, row['anchor'])
            assert key not in seen
            seen.add(key)
            assert split == ('train' if int(identity[-2:]) < 8 else 'validation')
            assert sha(row['path']) == row['sha256']
            packet = torch.load(row['path'], map_location='cpu', weights_only=False)
            data = packet['data']
            base, target, valid = data['base14'], data['teacher14'], data['valid'].bool()
            assert base.shape == target.shape == (1, 32, 14) and valid.shape == (1, 32)
            assert int(valid.sum()) == row['valid_actions']
            assert 0 <= row['native_actual_actions'] <= row['valid_actions']
            result = head(base, data['features']).actions
            assert torch.isfinite(result).all() and torch.isfinite(target).all()
            lower, upper = base - head.bounds, base + head.bounds
            lower[..., arm] = torch.maximum(lower[..., arm], head.arm_lower)
            upper[..., arm] = torch.minimum(upper[..., arm], head.arm_upper)
            assert (lower <= upper).all()
            ideal = torch.maximum(lower, torch.minimum(upper, target))
            tensors = [v[valid] for v in (base, target, result, ideal)]
            add(identity, split, *tensors, row['native_actual_actions'])
            add('ALL_' + split, split, *tensors, row['native_actual_actions'])
    assert {k for k in groups if k.startswith('phone-')} == {f'phone-{i:02d}' for i in range(12)}
    for g in groups.values():
        for m in g['models'].values():
            for stem, width in [('raw14', 14), ('arm12', 12), ('gripper2', 2),
                                ('clipped_gripper2', 2)]:
                m[stem + '_MSE'] = m.pop(stem + '_SSE') / (g['submitted_slots'] * width)
    deltas = {}
    for split in ('train', 'validation'):
        observed = groups['ALL_' + split]['models']['wide5000']['raw14_MSE']
        published = saved['diagnostics'][split]['decoded_physical14_MSE']
        deltas[split] = dict(published=published, CPU_reconstructed=observed,
            absolute_difference=abs(observed - published),
            within_roundoff_tolerance=abs(observed - published) <= max(1e-6, 5e-4 * published))
    guard()
    assert not torch.cuda.is_initialized()
    report = dict(format='wide5000_cached_clipping_diagnosis_v1',
        endpoint=dict(path=str(a.endpoint), sha256=a.endpoint_sha256),
        cache=dict(path=str(a.cache), file_sha256=a.cache_sha256, content_sha256=canonical),
        script_sha256=sha(__file__), head_module_sha256=sha(inspect.getsourcefile(WideActionResidual)),
        hostname=socket.gethostname(), checked_unix=time.time(), visible_gpu_count=a.visible_gpu_count,
        CPU_only=True, CUDA_initialized=False, new_actions=0, new_SGD=0,
        cases=groups, aggregate_reconstruction=deltas,
        failed04_success=False, source_eligible=False,
        interpretation_limits='Full submitted-plan slots, including known unexecuted tails; historical '
            'native actions are separate. Clip only gripper coordinates to qualified actuator [0,1]. '
            'Independent per-slot interval floor is not network capacity or physical success. '
            'CPU/GPU arithmetic may differ; inspect aggregate reconstruction before conclusions.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(a.output.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if __name__ == '__main__':
    main()
