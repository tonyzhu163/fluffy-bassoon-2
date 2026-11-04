"""Native all-layer KV parity and differentiability, with zero optimizer updates."""
import argparse
import gzip
import json
import os
from pathlib import Path
import pickle
import socket
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise ValueError('Use the allocated Linux accelerator')
    sys.path.insert(0, str(args.code / 'src'))
    import numpy as np
    import torch
    from openpi import transforms
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    from fluffy_bassoon.meta_rl.flow_history import (DirectActionCodec, FlowBridgeCodec,
        FlowBridgeConfig, FlowBridgeMotor, HistoryFlowSteeringPolicy)
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions, _json, native_video_writer
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, tensor_hash, checked_checkout
    from fluffy_bassoon.meta_rl.steering_history_eval import weights_digest
    from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend
    from fluffy_bassoon.meta_rl.vla_history import file_sha256
    checked_checkout(args.code, args.commit)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    torch.manual_seed(531105)
    root = Path('/mowenjie/fluffy-bassoon-2')
    run = root / 'outputs/robocasa-eight-hour-babysit-20261104-r1'
    args.output.mkdir(parents=True, exist_ok=False)
    original = run / 'gpu/native-six-qualification-01/full/OpenDrawer/bank-000/RESULT.json'
    wave = json.loads(original.read_text())
    saved = torch.load(wave['references'][0]['path'], map_location='cpu', weights_only=False)
    ref = saved['chunks'][0]
    assert file_sha256(ref['path']) == ref['sha256']
    with gzip.open(ref['path'], 'rb') as stream:
        sample = pickle.load(stream)
    observation = sample['observation']
    norm = root / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    model_path = root / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output / 'native'),
        ROBOCASA_PI05_TORCH_MODEL=str(model_path), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model_path), norm_path=str(norm),
        source_commit=args.commit, runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'), args.code)
    assert signature['model_sha256'] == '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
    assert signature['norm_sha256'] == '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
    frozen = tensor_hash(_versions(native.model))
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    codec = FlowBridgeCodec.from_stats(stats)
    with torch.no_grad():
        from fluffy_bassoon.meta_rl.flow_history import FlowDirectMotor
        features, prepared = FlowDirectMotor(native, DirectActionCodec.from_stats(stats)).prepare_batch(observation)
    hidden, mask, cache = prepared['prefix']
    assert len(features) == 1 and len(cache) > 1
    config = FlowBridgeConfig(interface='kv_bridge', width=256, layers=6, heads=8,
        context_length=384, kv_heads=cache[0][0].shape[2], head_dim=cache[0][0].shape[3],
        kv_rank=4, kv_layers=tuple(range(len(cache))), scale=.1)
    policy = HistoryFlowSteeringPolicy(config).to('cuda:0').eval()
    token = torch.cat((torch.as_tensor(features, device='cuda:0'), torch.zeros((1,603), device='cuda:0')), -1)
    draw = torch.as_tensor(sample['native_noise'][None], device='cuda:0')
    builder = native.model.build_prefix_cache
    try:
        native.model.build_prefix_cache = lambda *a, **k: prepared['prefix']
        with torch.no_grad():
            baseline, _ = native.model.predict_action_batch(prepared['observation'], mode='eval', noise=draw)
    finally:
        native.model.build_prefix_cache = builder
    motor = FlowBridgeMotor(native, codec, policy)
    with torch.no_grad():
        control, _, _ = policy.step(token, deterministic=True)
        actual = motor.execute_batch(prepared, noise=draw.cpu().numpy(), steering=control.cpu().numpy(), spec=codec)
    baseline = baseline.detach().float().cpu().numpy()
    np.testing.assert_array_equal(actual, baseline)
    zero_policy_digest = weights_digest(policy)

    # Canonical representatives of mapped physical commands, not recovered
    # pre-binarization continuous values. Use the actual official transforms.
    physical = np.asarray(sample['executed_commands'], np.float32)
    padded = transforms.PadStatesAndActions(32)(dict(state=np.zeros(25,np.float32), actions=physical.copy()))
    leaf = transforms.NormStats(mean=np.asarray(stats['actions']['mean']), std=np.asarray(stats['actions']['std']))
    normalizer = transforms.Normalize(dict(actions=leaf), use_quantiles=False)
    target_array = normalizer(padded)['actions'].astype(np.float32)
    recovered = transforms.Unnormalize(dict(actions=leaf), use_quantiles=False)(dict(actions=target_array))['actions'][:,:12]
    recovered[:,[6,11]] = np.where(recovered[:,[6,11]] > .5, 1., -1.)
    np.testing.assert_allclose(recovered, physical, rtol=0, atol=2e-7)
    policy.requires_grad_(True)
    with torch.no_grad():
        _, prepared = motor.prepare_batch(observation)
    _, mask, cache = prepared['prefix']
    target = torch.as_tensor(target_array[None], device='cuda:0')
    tau = torch.tensor([.5], device='cuda:0')
    gradients = []
    for perturb in (False, True):
        if perturb:
            with torch.no_grad():
                for basis in policy.kv_adapter.output_bases:
                    basis.weight.normal_(std=.01)
        policy.zero_grad(set_to_none=True)
        context, _ = policy.encode_step(token)
        loss = policy.flow_matching_loss(native.model, prepared['observation'], target, draw, tau, cache, mask, context)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None for p in native.model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
        basis_norm = sum(float(p.grad.abs().sum()) for p in policy.kv_adapter.output_bases.parameters() if p.grad is not None)
        core_norm = sum(float(p.grad.abs().sum()) for p in policy.core.parameters() if p.grad is not None)
        assert basis_norm > 0 and (not perturb or core_norm > 0)
        edited = policy.kv_adapter(cache, mask, context)
        assert all(torch.equal(a[~mask], b[~mask]) for old,new in zip(cache,edited,strict=True) for a,b in zip(old,new,strict=True))
        gradients.append(dict(perturbed_probe_only=perturb, loss=float(loss.detach()),
            output_basis_gradient_L1=basis_norm, history_core_gradient_L1=core_norm))
    with torch.no_grad():
        for basis in policy.kv_adapter.output_bases:
            basis.weight.zero_()
    policy.zero_grad(set_to_none=True); policy.eval().requires_grad_(False)
    assert weights_digest(policy) == zero_policy_digest and tensor_hash(_versions(native.model)) == frozen
    # Exercise the complete cache length required by the prospective longest
    # task, using synthetic boundaries; this is a numerical capacity check.
    with torch.no_grad():
        state = None
        for index in range(378):
            query = token.clone(); query[:,-1] = float(index % 63 == 0)
            _, _, state = policy.step(query, state, deterministic=True)
        assert int(state['length'][0]) == 378
    native._prepared = None
    _json(args.output / 'NUMERICAL.json', dict(zero_actions_exact=True, layers=len(cache),
        adapter_rank=4, context_length=384, exercised_tokens=378, gradients=gradients,
        official_physical_normalization_roundtrip_max_abs=float(np.abs(recovered-physical).max()),
        physical_binary_target='canonical +/-1 representative; not the original continuous prediction',
        native_parameters_frozen=True, native_gradient_tensors=0, optimizer_updates=0,
        perturbed_probe_basis_restored=True, source_chunk=ref, signature=signature))
    case = saved['case']
    journal = ActionJournal(args.output / 'accounting', rank=0, world=1, attempt='kv-zero', maximum_actions=4368)
    backend = NativeAttemptBackend(output=args.output/'attempts',
        canonical_root=run/'gpu/canonical-qualification-ID', motor=motor, spec=codec,
        native_factory=factory.paired, journal=journal, writer_factory=native_video_writer,
        binding=dict(source_commit=args.commit, model=signature, render_contract='bounded_two_intensity_levels',
            qualification_only=True, zero_initialized_all_layer_KV=True), maximum_rgb_delta=2,
        actual_attempts=6, fixed_case_order=True)
    rows = backend.run_bank(policy,[dict(case=case,horizon=600)],lanes=1,attempts=6,deadline_unix=time.time()+1100)
    assert len(rows)==6 and backend.last_batch_sizes==[1]*72
    assert tensor_hash(_versions(native.model)) == frozen
    _json(args.output/'RESULT.json',dict(completed=True, episodes=rows, references=backend.references,
        controls_charged=journal.charged(), optimizer_updates=0, native_parameters_frozen=True,
        zero_actions_exact=True, exercised_tokens=378, hostname=socket.gethostname()))


if __name__ == '__main__':
    main()
