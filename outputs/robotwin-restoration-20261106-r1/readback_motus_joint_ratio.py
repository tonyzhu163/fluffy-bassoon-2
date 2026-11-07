"""Independent CPU reduction of the closed eight-rank likelihood fixtures."""
import io
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    assert x.size and np.isfinite(x).all()
    return dict(n=x.size, mean=float(x.mean()), std=float(x.std()),
                minimum=float(x.min()), maximum=float(x.max()),
                quantiles=dict(zip(('p01','p05','p50','p95','p99'),
                                   map(float, np.quantile(x, (.01,.05,.5,.95,.99))))))


def close_distribution(actual, expected, tolerance):
    assert actual['n'] == expected['n']
    for key in ('mean','minimum','maximum'):
        assert abs(actual[key]-expected[key]) <= tolerance, (key, actual, expected, tolerance)
    assert abs(actual['std']-expected['std']) <= 2*tolerance
    for key in actual['quantiles']:
        assert abs(actual['quantiles'][key]-expected['quantiles'][key]) <= tolerance


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    started = time.monotonic()
    source = BASE/'motusjointaudit01-signals'
    result = json.loads((source/'RESULT.json').read_bytes())
    assert result['status'] == 'completed_read_only_joint_likelihood_measurement'
    assert result['optimizer_updates'] == result['simulator_actions'] == 0
    assert json.loads((source/'EXIT.json').read_bytes())['exit_code'] == 0
    assert json.loads((source/'resources.json').read_bytes())['sampled_reserve_passed']
    rows = sorted(result['actors'], key=lambda x:x['rank'])
    assert [r['rank'] for r in rows] == list(range(8))
    combined = {micro:[] for micro in (32,256)}
    rank_summaries = []
    for row in rows:
        path = source/f"joint-ratios-rank-{row['rank']:02d}.pt"
        assert sha(path) == row['fixture_sha256']
        fixture = torch.load(io.BytesIO(path.read_bytes()), map_location='cpu', weights_only=True)
        old = fixture['original_rollout_logprobs'].double().numpy()
        mask = fixture['loss_mask'].numpy().astype(bool)
        advantages = fixture['advantages'].double().numpy()
        assert old.shape == (768,32,14) and mask.shape == advantages.shape == (768,)
        assert int(mask.sum()) == row['valid_decisions']
        assert row['parameters_unchanged'] and row['optimizer_updates'] == row['simulator_actions'] == 0
        detail = dict(rank=row['rank'], valid_decisions=int(mask.sum()), comparisons={})
        for micro in (32,256):
            replay = fixture[f'actor_batch_{micro}'].double().numpy()
            assert replay.shape == old.shape and np.isfinite(replay).all()
            # Float64 coordinate differences avoid reproducing the GPU's float32
            # summation order. Its measured rounding discrepancy bounds every
            # comparison against the original native float32 summary below.
            joint = (replay-old).sum(axis=(1,2))[mask]
            ratio = np.exp(joint)
            report = row['comparisons'][f'actor_batch_{micro}']
            rounding = report['separate_float32_sum_vs_float64_difference_max_abs'] + 1e-8
            close_distribution(distribution(joint), report['joint_log_ratio'], rounding)
            ratio_tolerance = float(ratio.max()*np.expm1(rounding)) + 1e-8
            close_distribution(distribution(ratio), report['ratio'], ratio_tolerance)
            coordinate_mae = float(np.abs(replay-old)[mask].mean())
            assert abs(coordinate_mae-report['coordinate_mae']) < 1e-6
            low_ratio, high_ratio = np.exp(joint-rounding), np.exp(joint+rounding)
            outside_low = float(((high_ratio<.8)|(low_ratio>1.2)).mean())
            outside_high = float(((low_ratio<.8)|(high_ratio>1.2)).mean())
            assert outside_low-1e-7 <= report['ratio_outside_08_12_fraction'] <= outside_high+1e-7
            a = advantages[mask]
            clipped_low = float((((a>0)&(low_ratio>1.2))|((a<0)&(high_ratio<.8))).mean())
            clipped_high = float((((a>0)&(high_ratio>1.2))|((a<0)&(low_ratio<.8))).mean())
            assert clipped_low-1e-7 <= report['clipped_surrogate_fraction_without_sgd'] <= clipped_high+1e-7
            detail['comparisons'][str(micro)] = dict(
                coordinate_mae=coordinate_mae, joint_log_ratio_float64=distribution(joint),
                ratio_float64=distribution(ratio), native_float32_rounding_bound=rounding,
                ratio_outside_08_12_fraction_float64=float(((ratio<.8)|(ratio>1.2)).mean()),
                clipped_surrogate_fraction_float64=float((((a>0)&(ratio>1.2))|((a<0)&(ratio<.8))).mean()))
            combined[micro].append((joint,a))
        rank_summaries.append(detail)
    summaries = {}
    for micro, pieces in combined.items():
        joint = np.concatenate([x[0] for x in pieces]); a = np.concatenate([x[1] for x in pieces])
        ratio = np.exp(joint)
        summaries[str(micro)] = dict(valid_decisions=len(joint),
            joint_log_ratio_float64=distribution(joint), ratio_float64=distribution(ratio),
            ratio_outside_08_12_fraction_float64=float(((ratio<.8)|(ratio>1.2)).mean()),
            clipped_surrogate_fraction_without_sgd_float64=float((((a>0)&(ratio>1.2))|((a<0)&(ratio<.8))).mean()))
    output = BASE/'motusjointaudit01-readback01'; output.mkdir()
    write(output/'RESULT.json', dict(status='passed_independent_joint_likelihood_readback',
        source_commit=os.environ['TASK_SOURCE'], gpu_result_sha256=sha(source/'RESULT.json'),
        all8_fixture_hashes_verified=True, native_summary_comparisons_passed=True,
        floating_point_reduction='independent CPU float64; native GPU float32 difference bounded explicitly',
        aggregate=summaries, ranks=rank_summaries, seconds=time.monotonic()-started,
        gpu_seconds=0, learning_gain_claim=False, root_cause_established=False))


if __name__ == '__main__': main()
