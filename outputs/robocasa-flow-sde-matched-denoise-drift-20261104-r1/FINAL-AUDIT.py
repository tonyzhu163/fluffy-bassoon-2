"""Independent Linux CPU audit of matched-index fixtures and float64 reductions."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import numpy as np
    import torch
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-script', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    root = args.root
    prepared_path = root / 'prepared/PREPARED.json'
    prepared = read(prepared_path)
    assert prepared['completed'] and prepared['common_indices'] == list(range(10))
    assert prepared['per_index'] == 16 and len(prepared['tasks']) == 5
    source_sha = sha(args.source_script)
    times = np.linspace(1., 0., 11, dtype=np.float64)
    denominator = np.where(times == 1, times[1], times)
    std = .3 * np.sqrt(np.diff(-times) * times[:-1] / (1 - denominator[:-1]))
    stage = read(root / 'staging/PROGRESS.json')
    stage_keys = [(r['task'], r['step']) for r in stage['receipts']]
    assert len(stage_keys) == len(set(stage_keys))
    steps = [0, 1, 2] + list(range(10, 131, 10))
    maximum_error = 0.

    def close(actual, expected):
        nonlocal maximum_error
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        maximum_error = max(maximum_error, float(np.max(np.abs(a - b))))
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)

    tasks, tensors = [], []
    for entry in prepared['tasks']:
        task = entry['task']
        result_path = root / 'gpu' / task / 'RESULT.json'
        if not result_path.exists() or not read(result_path)['completed']:
            assert args.allow_incomplete, f'{task} is not complete'
            continue
        result = read(result_path)
        assert result['task'] == task and result['new_controls'] == result['SGD'] == 0
        assert result['runtime'] == prepared['runtime']
        assert result['common_indices'] == list(range(10))
        assert result['model_forward_calls'] == 160
        assert [r['step'] for r in result['rows']] == steps
        close(result['transition_std'], std)
        assert set(result['initial_parity']) == set(map(str, range(10)))
        assert all(v['passed'] and v['max_abs_chunk_log_ratio'] == 0
                   for v in result['initial_parity'].values())
        assert sha(entry['config']) == entry['config_sha256']

        # Reconstruct every selected fixture from its original Q1 actor records.
        flat_parts, old_parts = [], []
        for record in entry['actor_batches']:
            assert sha(record['path']) == record['sha256']
            batch = torch.load(record['path'], map_location='cpu', weights_only=False)['rollout_batch']
            time_size, batch_size = batch['prev_logprobs'].shape[:2]
            assert [time_size, batch_size] == record['shape']
            flat_parts.append({k: v.reshape(time_size * batch_size, *v.shape[2:])
                               for k, v in batch['forward_inputs'].items()})
            old_parts.append(batch['prev_logprobs'].reshape(time_size * batch_size, 50, 12))
            del batch
        original = {k: torch.cat([part[k] for part in flat_parts]) for k in flat_parts[0]}
        original_old = torch.cat(old_parts)
        fixture_records = []
        for fixture in entry['batches']:
            index = fixture['denoise_index']
            assert index in range(10) and sha(fixture['fixture']) == fixture['fixture_sha256']
            data = torch.load(fixture['fixture'], map_location='cpu', weights_only=False)
            positions = fixture['positions']
            assert len(positions) == len(set(positions)) == 16
            assert torch.equal(data['old_logprobs'], original_old[positions])
            for name, value in data['forward_inputs'].items():
                assert torch.equal(value, original[name][positions]), (task, index, name)
                digest = hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                assert digest == fixture['input_tensor_sha256'][name]
            assert bool((data['forward_inputs']['denoise_inds'] == index).all())
            fixture_records.append(dict(index=index, sha256=fixture['fixture_sha256'],
                                        selected_original_samples_exact=True))
        assert len(fixture_records) == 10
        del original, original_old, flat_parts, old_parts, data

        for row in result['rows']:
            step = row['step']
            ready_path = root / 'staging' / task / str(step) / 'READY.json'
            consumed_path = root / 'gpu' / task / f'CONSUMED-{step}.json'
            ready, consumed = read(ready_path), read(consumed_path)
            checkpoint = ready['checkpoint']
            assert ready['completed'] and ready['task'] == task
            assert ready['script_sha256'] == source_sha
            assert ready['result_root'] == str(root / 'gpu')
            assert checkpoint['step'] == step and checkpoint['sha256'] == row['model_sha256']
            assert sha(checkpoint['marker']) == checkpoint['marker_sha256']
            marker = read(checkpoint['marker'])
            assert marker['completed'] and marker['global_step'] == step
            assert any(r['sha256'] == row['model_sha256'] and r['path'] == checkpoint['path']
                       for r in marker['actor_checkpoint_files'])
            assert consumed['task'] == task and consumed['step'] == step
            assert consumed['strict_load_passed'] and consumed['model_sha256'] == row['model_sha256']
            assert consumed['ready_sha256'] == sha(ready_path)
            assert (task, step) in stage_keys
            delta_path = root / 'gpu' / task / 'deltas' / f'step-{step:03d}.pt'
            deltas = torch.load(delta_path, map_location='cpu', weights_only=True)
            assert set(deltas) == set(range(10))
            normalized = []
            for index, tensor in sorted(deltas.items()):
                assert tensor.dtype == torch.float64 and tuple(tensor.shape) == (16, 50, 12)
                array = tensor.numpy()
                assert np.isfinite(array).all()
                if step == 0:
                    assert np.count_nonzero(array) == 0
                per_sample = np.sum(array, axis=(1, 2), dtype=np.float64)
                per_dimension = np.mean(np.sum(array, axis=1, dtype=np.float64), axis=0)
                report = row['per_index'][str(index)]
                close(report['per_sample_chunk_log_ratio'], per_sample)
                close(report['mean_chunk_log_ratio'], np.mean(per_sample))
                close(report['per_dimension_chunk_log_ratio'], per_dimension)
                scale = (std[index] / std[0]) ** 2
                value = np.mean(per_sample) * scale
                close(report['variance_normalized_mean_chunk_log_ratio'], value)
                close(np.sum(per_dimension), np.mean(per_sample))
                normalized.append(value)
            close(row['mean_variance_normalized_over_indices'], np.mean(normalized))
            tensors.append(dict(task=task, step=step, delta_sha256=sha(delta_path),
                                READY_sha256=sha(ready_path), CONSUMED_sha256=sha(consumed_path),
                                model_sha256=row['model_sha256'], index_tensors=10))
        final = result['rows'][-1]
        tasks.append(dict(task=task, checkpoint_loads=16, index_tensors=160,
                          fixture_sources_exact=True, fixtures=fixture_records,
                          result_sha256=sha(result_path),
                          final_normalized_mean=final['mean_variance_normalized_over_indices'],
                          final_per_index=final['per_index']))

    whole = len(tasks) == 5
    clock = read(root / 'gpu/clock.json') if (root / 'gpu/clock.json').exists() else None
    if not args.allow_incomplete:
        assert whole and len(tensors) == 80 and stage['completed'] and len(stage_keys) == 80
        assert clock and clock['exit_code'] == 0 and clock['allocated_gpus'] == 1
        assert clock['observed_gpu_seconds'] <= 14400
        assert set(stage_keys) == {(r['task'], s) for r in prepared['tasks'] for s in steps}
        for record in stage['receipts']:
            ready = read(root / 'staging' / record['task'] / str(record['step']) / 'READY.json')
            assert record['private_copy_reclaimed'] == ready['temporary']
            if ready['temporary']:
                assert not Path(ready['path']).exists()
    report = dict(completed_for_recorded_tasks=True, whole_run_completed=whole and not args.allow_incomplete,
                  checked_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  audit_source_sha256=sha(__file__), executable_source_sha256=source_sha,
                  prepared_sha256=sha(prepared_path), completed_tasks=len(tasks), checkpoint_loads=len(tensors),
                  index_tensors=10 * len(tensors), maximum_independent_reduction_error=maximum_error,
                  tasks=tasks, tensors=tensors, stage_completed=stage['completed'],
                  staging_receipts=len(stage_keys), clock=clock, CPU_only=True,
                  new_GPU_seconds=0, new_controls=0, SGD=0,
                  scope='Matched-index conditional saved-transition likelihood; not action marginal KL or robot performance.')
    with args.destination.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['tasks', 'tensors']}), flush=True)


if __name__ == '__main__':
    main()
