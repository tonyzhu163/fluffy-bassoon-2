"""Numerically audit the retained drift tensors on Linux CPU after GPU exit."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            close(a, e)
    else:
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (actual, expected)


def main():
    import torch
    torch.set_num_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    clock = json.loads((root / 'gpu/clock.json').read_text())
    assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 1
    assert clock['observed_gpu_seconds'] <= 10800
    stage = json.loads((root / 'staging/PROGRESS.json').read_text())
    assert stage['completed'] and len(stage['receipts']) == 80
    cohorts = json.loads((root / 'COHORT-READOUT.json').read_text())
    assert cohorts['completed'] and cohorts['native_sampler_matches_formula']
    cohort_by_task = {r['task']: r for r in cohorts['tasks']}
    summaries, tensors = [], []
    steps = [0, 1, 2] + list(range(10, 131, 10))
    for task, cohort in cohort_by_task.items():
        path = root / 'gpu' / task / 'RESULT.json'
        result = json.loads(path.read_text())
        reference = json.loads((args.reference / task / 'RESULT.json').read_text())
        assert result['completed'] and result['new_controls'] == result['SGD'] == 0
        assert result['model_forward_calls'] == 16
        assert result['initial_parity']['max_abs_chunk_log_ratio'] == 0
        assert [row['step'] for row in result['rows']] == steps
        assert [row['checkpoint']['step'] for row in reference['rows']] == steps
        maximum_sample_delta = 0.0
        for row, old in zip(result['rows'], reference['rows']):
            assert row['model_sha256'] == old['checkpoint']['sha256']
            delta_path = root / 'gpu' / task / 'deltas' / f"step-{row['step']:03d}.pt"
            x = torch.load(delta_path, map_location='cpu', weights_only=True)
            assert x.shape == (16, 50, 12) and x.dtype == torch.float64
            assert torch.isfinite(x).all()
            if row['step'] == 0:
                assert torch.count_nonzero(x) == 0
            per_sample = x.sum((1, 2)).tolist()
            per_dimension = x.sum(1).mean(0).tolist()
            close(row['per_sample_chunk_log_ratio_to_initial'], per_sample)
            close(row['per_dimension_chunk_log_ratio'], per_dimension)
            close(row['per_timestep_log_ratio'], x.sum(2).mean(0).tolist())
            close(row['per_sample_per_dimension'], x.sum(1).tolist())
            close(row['mean_chunk_log_ratio_to_initial'], sum(per_sample) / 16)
            total = sum(per_dimension)
            absolute = sum(abs(v) for v in per_dimension)
            negative = sum(max(-v, 0) for v in per_dimension)
            binary = per_dimension[6] + per_dimension[11]
            close(row['binarized_dimensions_share'], binary / total if total else 0)
            close(row['binarized_absolute_dimension_share'],
                  (abs(per_dimension[6]) + abs(per_dimension[11])) / absolute if absolute else 0)
            close(row['binarized_negative_dimension_share'],
                  (max(-per_dimension[6], 0) + max(-per_dimension[11], 0)) / negative if negative else 0)
            mean = sum(per_sample) / 16
            assert abs(mean - old['mean_chunk_log_ratio_to_initial']) <= .01 * max(1, abs(mean))
            maximum_sample_delta = max(maximum_sample_delta,
                max(abs(a - b) for a, b in zip(per_sample, old['per_sample_chunk_log_ratio_to_initial'])))
            ready = root / 'staging' / task / str(row['step']) / 'READY.json'
            consumed = root / 'gpu' / task / f"CONSUMED-{row['step']}.json"
            receipt = json.loads(consumed.read_text())
            assert receipt['ready_sha256'] == sha(ready)
            assert receipt['model_sha256'] == row['model_sha256']
            tensors.append(dict(task=task, step=row['step'], sha256=sha(delta_path),
                                bytes=delta_path.stat().st_size))
        final = result['rows'][-1]
        values = final['per_dimension_chunk_log_ratio']
        total = final['mean_chunk_log_ratio_to_initial']
        summaries.append(dict(task=task, denoise_index=cohort['denoise_index'],
            transition_variance=cohort['transition_variance'],
            mean_chunk_log_ratio=total,
            variance_times_log_ratio=cohort['transition_variance'] * total,
            per_dimension_chunk_log_ratio=values,
            binary_signed_share=final['binarized_dimensions_share'],
            binary_absolute_share=final['binarized_absolute_dimension_share'],
            arm_signed_share=sum(values[:6]) / total,
            arm_absolute_share=final['arm_absolute_dimension_share'],
            base_motion_signed_share=sum(values[7:10]) / total,
            torso_signed_share=values[10] / total,
            maximum_per_sample_r3_difference=maximum_sample_delta,
            result_sha256=sha(path)))
    assert len(tensors) == 80
    output = root / 'FINAL-AUDIT.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(completed=True, analysis_sha256=sha(__file__),
        clock=clock, tensors=tensors, tasks=summaries,
        decomposition_tolerance=1e-10, checkpoint_forwards=80, new_controls=0, SGD=0,
        scope='One fixed saved-chain cohort per task at different denoising indices; '
              'conditional transition likelihood, not marginal action KL or behavior.'), indent=2) + '\n')
    with (root / 'dimension-summary.csv').open('x', newline='') as f:
        fields = ['task', 'denoise_index', 'mean_chunk_log_ratio', 'binary_signed_share',
                  'binary_absolute_share', 'arm_signed_share', 'base_motion_signed_share',
                  'torso_signed_share', 'transition_variance', 'variance_times_log_ratio']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(dict(completed=True, clock=clock, tasks=summaries)), flush=True)


if __name__ == '__main__':
    main()
