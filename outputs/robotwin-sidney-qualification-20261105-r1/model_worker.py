"""Native LeRobot inference in its own Python runtime, inside the allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    import numpy as np
    import torch
    from safetensors.torch import load_file, load_model
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    cfg = PI05Config.from_pretrained(str(args.model))
    assert cfg.chunk_size == cfg.n_action_steps == 50
    assert cfg.num_inference_steps == 10 and not cfg.use_relative_actions
    assert cfg.output_features['action'].shape == (14,)
    cfg.device = 'cuda'
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    policy = PI05Policy(cfg)
    # Unlike PI05Policy.from_pretrained, this fails on any loading error.
    # load_model understands safetensors' omission of shared/tied aliases.
    missing, unexpected = load_model(policy, str(args.model / 'model.safetensors'), strict=True)
    assert not missing and not unexpected
    policy.eval().requires_grad_(False)
    versions = {name: p._version for name, p in policy.named_parameters()}
    tokenizer = Path('/mowenjie/.cache/huggingface/hub/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c')
    assert tokenizer.is_dir()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(args.model),
        preprocessor_overrides={'device_processor': {'device': 'cuda'},
                                'tokenizer_processor': {'tokenizer_name': str(tokenizer)}})
    stats = load_file(str(args.model / 'policy_preprocessor_step_3_normalizer_processor.safetensors'))
    output_stats = load_file(str(args.model / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    for k in ('action.mean', 'action.std'):
        assert torch.equal(stats[k], output_stats[k]), k
    assert all(torch.isfinite(v).all() for v in stats.values())
    assert (stats['observation.state.std'] >= 0).all()
    ready = dict(status='ready', strict_weight_load=True, missing_keys=list(missing),
                 unexpected_keys=list(unexpected), parameter_count=sum(p.numel() for p in policy.parameters()),
                 chunk_size=50, flow_steps=10, absolute_joint_actions=True,
                 normalization='saved MEAN_STD', compile_model=False,
                 torch=torch.__version__, python=sys.version, cuda_device=torch.cuda.get_device_name(0),
                 normalization_keys=sorted(stats), optimizer_updates=0,
                 tokenizer_revision=tokenizer.name,
                 tokenizer_files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(tokenizer.iterdir()) if p.is_file()})
    save(args.output / 'READY.json', ready)
    for line in sys.stdin:
        request = json.loads(line)
        source = Path(request['observation'])
        target = Path(request['actions'])
        with np.load(source, allow_pickle=False) as obs:
            batch = {'observation.state': torch.from_numpy(obs['state'].copy()),
                     'task': str(obs['instruction'].item())}
            for i, key in enumerate(('cam_high', 'cam_left_wrist', 'cam_right_wrist')):
                view = obs['images'][i]
                assert view.dtype == np.uint8 and view.ndim == 3 and view.shape[-1] == 3
                # LeRobot dataset contract: CHW float RGB in [0, 1].
                batch[f'observation.images.{key}'] = torch.from_numpy(view.copy()).permute(2, 0, 1).float() / 255
        processed = pre(batch)
        state = batch['observation.state']
        expected = (state - stats['observation.state.mean']) / (stats['observation.state.std'] + 1e-8)
        torch.testing.assert_close(processed['observation.state'].cpu().reshape(14), expected.reshape(14), atol=1e-5, rtol=1e-5)
        torch.manual_seed(int(request['noise_seed']))
        torch.cuda.manual_seed_all(int(request['noise_seed']))
        rng = torch.cuda.get_rng_state()
        cpu_rng = torch.get_rng_state()
        started = time.monotonic()
        with torch.inference_mode():
            normalized = policy.predict_action_chunk(processed)
            actions = post(normalized).float().cpu()
            expected_actions = normalized.float().cpu() * (output_stats['action.std'] + 1e-8) + output_stats['action.mean']
            torch.testing.assert_close(actions, expected_actions, atol=1e-6, rtol=1e-6)
            parity = None
            if request.get('check_queue_parity'):
                torch.cuda.set_rng_state(rng)
                torch.set_rng_state(cpu_rng)
                policy.reset()
                queued = torch.stack([post(policy.select_action(processed)).float().cpu() for _ in range(50)], dim=1)
                parity = float((queued - actions).abs().max())
                torch.testing.assert_close(queued, actions, atol=1e-5, rtol=1e-5)
                policy.reset()
        torch.cuda.synchronize()
        assert actions.shape == (1, 50, 14) and torch.isfinite(actions).all()
        assert not policy.training and not any(p.requires_grad for p in policy.parameters())
        assert versions == {name: p._version for name, p in policy.named_parameters()}
        array = actions[0].numpy()
        with target.with_suffix('.part').open('wb') as f:
            np.savez(f, actions=array, normalized=normalized.float().cpu().numpy())
        target.with_suffix('.part').replace(target)
        save(target.with_suffix('.json'), dict(seconds=time.monotonic()-started,
             queue_parity_max_abs=parity, frozen_unchanged=True,
             actions_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
             action_min=array.min(axis=0).tolist(), action_max=array.max(axis=0).tolist()))


if __name__ == '__main__':
    main()
