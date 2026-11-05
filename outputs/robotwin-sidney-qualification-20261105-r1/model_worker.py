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
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--precision', choices=('bfloat16','float32'),default='bfloat16')
    args = parser.parse_args()
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    cfg = PreTrainedConfig.from_pretrained(str(args.model))
    assert isinstance(cfg, PI05Config)
    assert cfg.chunk_size == cfg.n_action_steps == 50
    assert cfg.num_inference_steps == 10 and not cfg.use_relative_actions
    assert tuple(cfg.output_features['action'].shape) == (14,)
    cfg.device = 'cuda'
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.dtype = args.precision
    policy = PI05Policy(cfg)
    # Unlike PI05Policy.from_pretrained, this fails on any loading error.
    # load_model understands safetensors' omission of shared/tied aliases.
    missing, unexpected = load_model(policy, str(args.model / 'model.safetensors'), strict=True)
    assert not missing and not unexpected
    policy.eval().requires_grad_(False)
    # Keep the attention implementation constant from the first call, including
    # SigLIP. This is an explicit deterministic numerical-reference runtime.
    for module in policy.modules():
        if hasattr(getattr(module, 'config', None), '_attn_implementation'):
            module.config._attn_implementation = 'eager'
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
                 parameter_precision=args.precision,
                 deterministic_algorithms=True, attention='all eager', tf32=False,
                 cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
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
        def tensor_hash(value):
            return hashlib.sha256(value.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        def input_hashes():
            return {k:tensor_hash(v) for k,v in processed.items() if isinstance(v,torch.Tensor)}
        inputs_before = input_hashes()
        buffers_before = {k:tensor_hash(v) for k,v in policy.named_buffers()}
        prepared_images,_ = policy._preprocess_images(processed)
        with target.with_suffix('.inputs.npz').open('wb') as handle:
            np.savez(handle, images=np.stack([v.permute(0,2,3,1).cpu().numpy() for v in prepared_images],axis=1),
                     state=processed['observation.state'].cpu().numpy(),
                     tokens=processed['observation.language.tokens'].cpu().numpy(),
                     token_mask=processed['observation.language.attention_mask'].cpu().numpy())
        state = batch['observation.state']
        expected = (state - stats['observation.state.mean']) / (stats['observation.state.std'] + 1e-8)
        torch.testing.assert_close(processed['observation.state'].cpu().reshape(14), expected.reshape(14), atol=1e-5, rtol=1e-5)
        torch.manual_seed(int(request['noise_seed']))
        torch.cuda.manual_seed_all(int(request['noise_seed']))
        def attention_modes():
            return {name:getattr(module.config,'_attn_implementation')
                    for name,module in policy.named_modules()
                    if hasattr(getattr(module,'config',None),'_attn_implementation')}
        modes_before = attention_modes()
        started = time.monotonic()
        with torch.inference_mode():
            noise = policy.model.sample_noise((1,50,32),torch.device('cuda'))
            normalized = policy.predict_action_chunk(processed,noise=noise.clone())
            modes_after_chunk = attention_modes()
            actions = post(normalized).float().cpu()
            expected_actions = normalized.float().cpu() * (output_stats['action.std'] + 1e-8) + output_stats['action.mean']
            torch.testing.assert_close(actions, expected_actions, atol=1e-6, rtol=1e-6)
            parity = None
            if request.get('check_queue_parity'):
                policy.reset()
                original_sampler = policy.model.sample_noise
                calls = []
                def fixed_noise(shape,device):
                    assert tuple(shape)==tuple(noise.shape) and torch.device(device)==noise.device
                    calls.append(tuple(shape))
                    return noise.clone()
                policy.model.sample_noise = fixed_noise
                try:
                    queued_normalized = torch.stack([policy.select_action(processed).clone() for _ in range(50)],dim=1)
                finally:
                    policy.model.sample_noise = original_sampler
                assert len(calls)==1
                queued = torch.stack([post(queued_normalized[:,i]).float().cpu() for i in range(50)],dim=1)
                parity = float((queued - actions).abs().max())
                diagnostic = dict(gaussian_noise='one explicit shared [1,50,32] tensor',
                    noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
                    normalized_max_abs=float((queued_normalized-normalized).abs().max()),
                    decoded_max_abs=parity, tolerance=dict(atol=1e-5,rtol=1e-5),
                    attention_modes_before=modes_before,attention_modes_after_chunk=modes_after_chunk,
                    attention_modes_after_queue=attention_modes(),
                    cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    inputs_before=inputs_before,inputs_after=input_hashes(),
                    buffers_before=buffers_before,buffers_after={k:tensor_hash(v) for k,v in policy.named_buffers()},
                    parameters_unchanged=versions=={name:p._version for name,p in policy.named_parameters()})
                if parity>1e-5:
                    repeated = policy.predict_action_chunk(processed,noise=noise.clone())
                    diagnostic.update(repeated_vs_first_normalized_max_abs=float((repeated-normalized).abs().max()),
                                      repeated_vs_queue_normalized_max_abs=float((repeated-queued_normalized).abs().max()))
                save(target.with_suffix('.parity.json'),diagnostic)
                torch.testing.assert_close(queued, actions, atol=1e-5, rtol=1e-5)
                policy.reset()
        torch.cuda.synchronize()
        assert actions.shape == (1, 50, 14) and torch.isfinite(actions).all()
        assert not policy.training and not any(p.requires_grad for p in policy.parameters())
        assert versions == {name: p._version for name, p in policy.named_parameters()}
        array = actions[0].numpy()
        with target.with_suffix('.part').open('wb') as f:
            np.savez(f, actions=array, normalized=normalized.float().cpu().numpy(),noise=noise.float().cpu().numpy())
        target.with_suffix('.part').replace(target)
        save(target.with_suffix('.json'), dict(seconds=time.monotonic()-started,
             queue_parity_max_abs=parity, frozen_unchanged=True,
             actions_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
             action_min=array.min(axis=0).tolist(), action_max=array.max(axis=0).tolist()))


if __name__ == '__main__':
    main()
