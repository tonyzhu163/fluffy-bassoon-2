"""Read retained fixed-cohort dimensions and noise scales on allocated Linux CPU."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(x):
    x = x.double().reshape(-1, x.shape[-1])
    return {k: v.tolist() for k, v in dict(
        mean=x.mean(0), std=x.std(0), min=x.min(0).values, max=x.max(0).values).items()}


def main():
    import torch
    torch.set_num_threads(2)
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--sampler', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('native_sampler', args.sampler)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    prepared = args.root / 'offline-initial-ratio-inputs-r2/PREPARED.json'
    admission = json.loads(prepared.read_text())
    x = torch.zeros(10, 1, 1, dtype=torch.float32)
    _, sigma = sampler.sample_mean_var(x, x, torch.arange(10),
        noise_method='flow_sde', noise_level=0.3, num_steps=10)
    times = sampler.get_timesteps(10, x.device)
    denominator = torch.where(times == 1, times[1], times)
    reference = 0.3 * torch.sqrt((times[:-1] - times[1:]) * times[:-1] / (1 - denominator[:-1]))
    assert torch.allclose(sigma[:, 0, 0], reference, atol=1e-7, rtol=1e-6)
    rows = []
    for fixture in admission['tasks']:
        assert sha(fixture['fixture']) == fixture['fixture_sha256']
        data = torch.load(fixture['fixture'], map_location='cpu', weights_only=False)
        inputs = data['forward_inputs']
        action = inputs['action'].reshape(16, 50, 12)
        model_action = inputs['model_action'].reshape(16, 50, 32)
        indices = torch.unique(inputs['denoise_inds']).tolist()
        assert len(indices) == 1
        index = indices[0]
        std = sigma[index, 0, 0].item()
        rows.append(dict(task=fixture['task'], fixture_sha256=fixture['fixture_sha256'],
            denoise_index=index, time=times[index].item(), transition_std=std,
            transition_variance=std**2,
            inverse_variance_relative_to_step0=(sigma[0, 0, 0].item() / std)**2,
            decoded_action_shape=list(action.shape), model_action_shape=list(model_action.shape),
            decoded_action=describe(action), model_action=describe(model_action),
            binary_threshold=0.5,
            original_cohort_binary_plus_fraction={str(i): (action[:, :, i] >= .5).double().mean().item()
                for i in (6, 11)}))
    value = dict(completed=True, prepared_sha256=sha(prepared), sampler=str(args.sampler),
        sampler_sha256=sha(args.sampler), analysis_sha256=sha(__file__),
        native_sampler_matches_formula=True, tasks=rows,
        metric_scope='Original fixed cohorts only; no later-policy action probabilities or behavior. '
        'Different denoising indices and states confound raw cross-task likelihood comparisons.',
        new_controls=0, SGD=0, model_calls=0)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    for row in rows:
        print(row['task'], row['denoise_index'], row['transition_std'],
              row['inverse_variance_relative_to_step0'], flush=True)


if __name__ == '__main__':
    main()
