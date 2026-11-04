"""Independent Linux CPU receipt, tensor and native-loss audit after replay."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def close(actual, expected, tolerance=1e-6):
    assert math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=tolerance), (actual, expected)


def main():
    import torch
    import rlinf.algorithms.losses
    from rlinf.algorithms.registry import policy_loss
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    result_path = root / 'gpu-r2/replay/RESULT.json'
    result = json.loads(result_path.read_text())
    prepared_path = root / 'fixtures-r1/PREPARED.json'
    prepared = json.loads(prepared_path.read_text())
    clock = json.loads((root / 'gpu-r2/clock.json').read_text())
    stage = json.loads((root / 'staging-r2/PROGRESS.json').read_text())
    assert result['completed'] and result['cohorts'] == 16 and result['decisions'] == 256
    assert result['new_controls'] == result['SGD'] == 0
    assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 1
    assert stage['completed'] and len(stage['receipts']) == 4
    assert result['prepared_sha256'] == sha(prepared_path)
    assert result['script_sha256'] == sha(args.source / 'REPLAY.py')
    fixtures = {(r['policy_step'], r['actor_rank'], r['time_index']): r for r in prepared['cohorts']}
    summaries, receipts = [], []
    for step in [0, 1, 10, 120]:
        ready_path = root / f'staging-r2/{step:03d}/READY.json'
        consumed_path = root / f'gpu-r2/replay/CONSUMED-{step:03d}.json'
        ready, consumed = [json.loads(p.read_text()) for p in [ready_path, consumed_path]]
        assert consumed['strict_load_passed'] and consumed['step'] == ready['step'] == step
        assert consumed['ready_sha256'] == sha(ready_path)
        assert ready['script_sha256'] == result['script_sha256']
        assert ready['prepared_sha256'] == sha(prepared_path)
        assert ready['checkpoint']['sha256'] == consumed['model_sha256']
        assert sha(ready['checkpoint']['marker']) == ready['checkpoint']['marker_sha256']
        receipt = next(r for r in stage['receipts'] if r['step'] == step)
        assert receipt['consumed_sha256'] == sha(consumed_path)
        assert receipt['private_copy_reclaimed'] and not Path(ready['path']).exists()
        receipts.append(dict(step=step, ready_sha256=sha(ready_path), consumed_sha256=sha(consumed_path)))
        gaps, suppressed, tensor_receipts, mean_roundoff = [], 0, [], 0.0
        rows = [r for r in result['rows'] if r['policy_step'] == step]
        assert len(rows) == 4
        for row in rows:
            key = (step, row['actor_rank'], row['time_index'])
            fixture_ref = fixtures[key]
            assert row['fixture_sha256'] == fixture_ref['fixture_sha256'] == sha(fixture_ref['fixture'])
            assert row['model_sha256'] == ready['checkpoint']['sha256']
            assert fixture_ref['native_record_joins_exact']
            assert sha(fixture_ref['config']) == fixture_ref['config_sha256']
            assert sha(fixture_ref['join']) == fixture_ref['join_sha256']
            path = Path(row['tensors'])
            assert path.parent == root / 'gpu-r2/replay/tensors'
            assert sha(path) == row['tensors_sha256']
            tensors = torch.load(path, map_location='cpu', weights_only=True)
            fixture = torch.load(fixture_ref['fixture'], map_location='cpu', weights_only=False)
            all_values = tensors['all_suffix_values']
            assert all_values.shape == (16, 10) and torch.isfinite(all_values).all()
            old = tensors['saved_old_values']
            indices = fixture['forward_inputs']['denoise_inds'][:, 0].long()
            selected = all_values[torch.arange(16), indices]
            assert torch.equal(old, fixture['old_values'].flatten())
            # CUDA and CPU use different float32 reduction orders. Audit the
            # ten retained float32 inputs in float64 with a declared bound.
            mean_error = float((all_values.double().mean(1) - old.double()).abs().max())
            assert mean_error <= 2e-7, mean_error
            mean_roundoff = max(mean_roundoff, mean_error)
            assert torch.equal(selected, tensors['selected_values'])
            for field in ['returns', 'loss_mask']:
                assert torch.equal(tensors[field], fixture[field])
            gap = selected - old
            close(gap.abs().max(), row['value_gap_max_abs'])
            close(gap.abs().mean(), row['value_gap_mean_abs'])
            close(gap.square().mean().sqrt(), row['value_gap_rms'])
            close((gap.abs() > .2).float().mean(), row['initial_value_clip_fraction'])
            assert row['saved_mean_max_abs_error'] == row['native_selected_max_abs_error'] == 0
            assert row['logprob_parity']['passed'] and row['sampler_rng_unchanged']
            assert row['logprob_parity']['max_abs_chunk_log_ratio'] == 0
            assert row['logprob_parity']['max_abs_coordinate_log_ratio'] == 0
            gradients = []
            for label, reference, loss_key in [
                ('native', old, 'native_critic_loss'),
                ('consistent', selected, 'consistent_reference_critic_loss'),
            ]:
                values = selected.clone().requires_grad_(True)
                loss, metrics = policy_loss(task_type='embodied', loss_type='actor_critic',
                    reward_type='chunk_level', logprob_type='chunk_level', single_action_dim=12,
                    logprobs=fixture['old_logprobs'], old_logprobs=fixture['old_logprobs'],
                    advantages=torch.zeros(16, 1), values=values, prev_values=reference[:, None],
                    returns=fixture['returns'], loss_mask=fixture['loss_mask'],
                    loss_mask_sum=fixture['loss_mask_sum'], max_episode_steps=600,
                    clip_ratio_low=.2, clip_ratio_high=.2, value_clip=.2, huber_delta=10.)
                gradient, = torch.autograd.grad(loss, values)
                close(metrics['critic/value_loss'], row[loss_key])
                assert torch.allclose(gradient, tensors[f'{label}_prediction_gradient'], atol=1e-7, rtol=1e-6)
                gradients.append(gradient)
            suppressed_here = int(((gradients[0] == 0) & (gradients[1] != 0)).sum())
            close(suppressed_here / 16, row['suppressed_prediction_gradient_fraction'])
            suppressed += suppressed_here
            assert all(math.isfinite(row[k]) for k in [
                'native_head_gradient_norm', 'consistent_head_gradient_norm', 'head_gradient_delta_norm'])
            gaps.append(gap.double())
            tensor_receipts.append(dict(actor_rank=row['actor_rank'], time_index=row['time_index'],
                sha256=sha(path), bytes=path.stat().st_size))
        gap = torch.cat(gaps)
        summaries.append(dict(policy_step=step, decisions=len(gap), value_gap_max_abs=float(gap.abs().max()),
            value_gap_mean_abs=float(gap.abs().mean()), value_gap_rms=float(gap.square().mean().sqrt()),
            pre_sgd_clip_count=int((gap.abs() > .2).sum()), suppressed_prediction_gradients=suppressed,
            cpu_float64_mean_to_saved_max_abs_error=mean_roundoff,
            maximum_reported_head_gradient_relative_delta=max(r['head_gradient_relative_delta'] for r in rows),
            tensors=tensor_receipts))
    qualified = json.loads((root / 'QUALIFIED-FIX.json').read_text())
    assert qualified['completed'] and qualified['cases'] == 20
    output = dict(completed=True, audit_script_sha256=sha(__file__), result_sha256=sha(result_path),
        replay_script_sha256=result['script_sha256'], prepared_sha256=sha(prepared_path),
        qualified_fix_sha256=sha(root / 'QUALIFIED-FIX.json'), clock=clock, receipts=receipts,
        cohorts=16, decisions=256, by_policy_step=summaries, new_controls=0, SGD=0,
        cpu_float64_mean_absolute_tolerance=2e-7,
        checks='Independent tensor reductions, archive/fixture/model receipt hashes, selected-step gather, '
               'saved rollout mean, and CPU native-loss/prediction gradients.',
        limits='Head-gradient vectors were not retained, so their norms/deltas are GPU replay reports; '
               'historical returns fixed, no corrected GAE or learning outcome measured.')
    (root / 'FINAL-AUDIT.json').write_text(json.dumps(output, indent=2) + '\n')
    with (root / 'critic-summary.csv').open('w', newline='') as stream:
        fields = [key for key in summaries[0] if key != 'tensors']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(dict(completed=True, cohorts=16, decisions=256, by_policy_step=summaries)))


if __name__ == '__main__':
    main()
