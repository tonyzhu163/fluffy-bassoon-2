"""Joint native frozen-expert steering: actual RGB queries, action-derived FM loss."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time


ROOT = Path('/mowenjie/fluffy-bassoon-2')
OLD = ROOT / 'outputs/robocasa-three-task-meta-baseline-20261104-r2'
RUN = 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1'
ASSEMBLY_SHA = 'fc3f388ef3feddde4c44ebd45ae9d7e73dab4a0fc58252c506315697be2a4758'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--resume-contract-commit', help='Explicit implementation migration preserving an existing learner contract')
    p.add_argument('--member-cache', type=Path, help='Complete hash-verified PFS source members')
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interface', choices=('kv_bridge', 'velocity_bridge'), required=True)
    p.add_argument('--objective', choices=('ad', 'dpt', 'dit', 'ic_iql'), required=True)
    p.add_argument('--execution-batch', type=int, default=8)
    p.add_argument('--max-seconds', type=int, default=6300)
    p.add_argument('--qualify', action='store_true')
    p.add_argument('--reuse-zero', type=Path)
    p.add_argument('--qualify-updates', type=int, default=50)
    args = p.parse_args()
    if sys.platform != 'linux' or not socket.gethostname().startswith('job-'):
        raise ValueError('Run native computation inside the actual scheduler allocation')
    sys.path[:0] = [str(args.code / 'src'), os.environ['RLINF_PATH']]
    import numpy as np
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import checked_checkout, tensor_hash
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _load_native, _versions, _json
    from fluffy_bassoon.meta_rl.flow_history import (DirectActionCodec, FlowBridgeConfig,
        FlowBridgeCodec, FlowDirectMotor, FlowBridgeMotor, HistoryFlowSteeringPolicy,
        sample_bridge_query_batch, export_bridge_policy)
    from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
    from fluffy_bassoon.meta_rl.flow_bridge_train import (bridge_update, query_weights,
        validate_batches, save_resume, restore_resume)
    from fluffy_bassoon.meta_rl.vla_history import load_dataset, file_sha256, HistoryConfig
    from fluffy_bassoon.meta_rl.history_value import HistoryValueLearner, ValueConfig
    from fluffy_bassoon.meta_rl.vla_history_train import advance_plateau
    checked_checkout(args.code, args.commit)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(args.probe)
    verify_selected_cuda(torch)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(selected_device()['allocation_ordinal'])
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    assembly_path = OLD / 'joint/preparation/ASSEMBLY.json'
    if file_sha256(assembly_path) != ASSEMBLY_SHA:
        raise ValueError('The fixed pooled three-task assembly changed')
    assembly = json.loads(assembly_path.read_text())
    train, validation, manifests = [], [], []
    for split in ('train', 'validation'):
        item = assembly['datasets'][args.objective + '_' + split]
        trials, manifest = load_dataset(item['path'], expected_sha256=item['sha256'])
        manifests.append(manifest)
        (train if split == 'train' else validation).extend(trials)
    tm, vm = manifests
    if (len(train), len(validation)) != (336, 48) or set(tm['metadata']['trial_ids']) & set(vm['metadata']['trial_ids']):
        raise ValueError('Preserve every original train/validation stream')
    codec_record = tm['metadata']['noise_contract']
    action_codec = DirectActionCodec(tuple(codec_record['mean']), tuple(codec_record['std']))
    sources = {}
    for task, reference in assembly['source_assemblies'].items():
        path = Path(reference['path'])
        if file_sha256(path) != reference['sha256']:
            raise ValueError('Task source assembly changed')
        record = json.loads(path.read_text())
        index = path.parent / 'TAR-INDEX.json'
        sources[task] = dict(index_path=str(index), index_sha256=file_sha256(index),
                             archive=record['provenance']['bos_archive'])
    reader = FlowSourceQueries(sources, action_codec, member_cache=args.member_cache)
    norm = ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    model_path = ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(args.output / 'native'),
        ROBOCASA_PI05_TORCH_MODEL=str(model_path), ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, factory, signature = _load_native(dict(model_directory=str(model_path), norm_path=str(norm),
        source_commit=args.commit, runtime_commit='8f9bfa7661928ef8c17494b04f4d7bf3125634e1'), args.code)
    if (signature['model_sha256'] != '661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5'
            or signature['norm_sha256'] != '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'):
        raise ValueError('Require the unchanged original VLM, action expert and normalization')
    motor = FlowDirectMotor(native, action_codec)
    first_observation = reader.observations([(0, 0)], train, tm['metadata']['trial_ids'])
    with torch.no_grad():
        _, prepared = motor.prepare_batch(first_observation)
    _, first_mask, first_cache = prepared['prefix']
    config = FlowBridgeConfig(interface=args.interface, width=256, layers=6, heads=8,
        context_length=192, kv_heads=first_cache[0][0].shape[2], head_dim=first_cache[0][0].shape[3],
        kv_rank=4, kv_layers=tuple(range(len(first_cache))), scale=.1)
    native._prepared = None
    del prepared, first_cache, first_mask
    policy = HistoryFlowSteeringPolicy(config).to('cuda:0')
    policy.fit_observation_normalization(train)
    values = None
    value_reference = None
    if args.objective in ('dit', 'ic_iql'):
        closure = json.loads((OLD / 'DIRECT-ACTION-CLOSURE-PREFLIGHT.json').read_text())['methods'][args.objective]
        value_reference = dict(path=closure['path'], sha256=closure['sha256'])
        if file_sha256(closure['path']) != closure['sha256']:
            raise ValueError('The preserved independently converged value state changed')
        saved = torch.load(closure['path'], map_location='cpu', weights_only=False)
        if (not saved['plateaus']['value']['converged'] or saved['best_value_state'] is None
                or saved['contract']['train_sha256'] != assembly['datasets'][args.objective + '_train']['sha256']):
            raise ValueError('Require the converged value estimator on the exact same source split')
        values = HistoryValueLearner(HistoryConfig(**saved['contract']['config']),
                                   ValueConfig(**saved['contract']['value_config'])).to('cuda:0')
        values.load_state_dict(saved['best_value_state'], strict=True)
        values.eval().requires_grad_(False)
        del saved
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    sampler = np.random.default_rng(0)
    generator = torch.Generator(device='cuda:0').manual_seed(771105)
    frozen = tensor_hash(_versions(native.model))
    contract = dict(source_commit=args.resume_contract_commit or args.commit, interface=args.interface, objective=args.objective,
        config=asdict(config), assembly_sha256=ASSEMBLY_SHA,
        train_sha256=assembly['datasets'][args.objective + '_train']['sha256'],
        validation_sha256=assembly['datasets'][args.objective + '_validation']['sha256'],
        native_model_sha256=signature['model_sha256'], native_norm_sha256=signature['norm_sha256'],
        sources=sources, logical_batch=8, max_episodes=12, learning_rate=.0003,
        seed=0, value_reference=value_reference,
        value_reuse='only independent converged Q/V used; direct-action actors never initialize or control a steering policy',
        target_space='canonical mapped physical actions encoded with official32-coordinate native normalization',
        loss='action-derived flow matching on active12 coordinates; one causal query per sampled history window',
        qualification_only=args.qualify,
        schedule=dict(minimum_updates=1000, eval_every=100, validation_batches=16,
                      patience=10, relative_delta=.001, checkpoint_selection='validation_flow_loss'))
    contract = json.loads(json.dumps(contract))
    progress = dict(update=0, best_loss=None, plateau={}, converged=False, elapsed_seconds=0.)
    if args.resume_contract_commit and not (args.output / 'latest.pt').exists():
        raise ValueError('Implementation migration requires an existing full learner checkpoint')
    if (args.output / 'latest.pt').exists():
        progress = restore_resume(args.output / 'latest.pt', contract=contract, policy=policy,
            optimizer=optimizer, sampler=sampler, generator=generator)
        if progress['converged']:
            raise ValueError('Do not resume an already converged actor')
        if args.resume_contract_commit:
            _json(args.output / 'implementations' / args.commit / f'{socket.gethostname()}-{os.getpid()}.json',
                dict(implementation_commit=args.commit, preserved_contract_commit=args.resume_contract_commit,
                     resumed_update=progress['update'], checkpoint_sha256=file_sha256(args.output / 'latest.pt'),
                     all_contract_fields_and_policy_optimizer_sampler_RNG_restored=True,
                     member_cache=str(args.member_cache) if args.member_cache else None,
                     member_cache_manifest_sha256=file_sha256(args.member_cache / 'CACHE.json') if args.member_cache else None,
                     migration='Verified storage-read retries and nonfatal read-only close; learner unchanged'))
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        _json(args.output / 'contract.json', contract)

    def prepare_queries(collection, metadata):
        def prepare(indices):
            observation = reader.observations(indices, collection, metadata['trial_ids'])
            try:
                with torch.no_grad():
                    _, batch_prepared = motor.prepare_batch(observation)
                _, mask, cache = batch_prepared['prefix']
                return batch_prepared['observation'], cache, mask
            finally:
                native._prepared = None
        return prepare

    train_prepare = prepare_queries(train, tm['metadata'])
    val_prepare = prepare_queries(validation, vm['metadata'])

    def batch(collection, rng, flow_rng, *, reactive=False):
        item = sample_bridge_query_batch(collection, policy, rng, action_codec=action_codec,
            native_action_stats=stats['actions'], batch_size=8, max_episodes=12,
            objective=args.objective, discount=.99)
        if reactive:
            lanes = torch.arange(8, device='cuda:0')
            token = item['tokens'][item['query_rel'], lanes][None].clone()
            token[..., config.observation_dim:] = 0.; token[..., -1] = 1.
            item.update(tokens=token, query_rel=torch.zeros(8, dtype=torch.long, device='cuda:0'),
                valid=torch.ones(1, 8, dtype=torch.bool, device='cuda:0'),
                starts=torch.ones(1, 8, dtype=torch.bool, device='cuda:0'))
        item['noise'] = torch.randn(item['target_actions'].shape, generator=flow_rng, device='cuda:0')
        item['timesteps'] = torch.rand(8, generator=flow_rng, device='cuda:0')
        item['weights'] = query_weights(item, args.objective, values)
        return item

    def validation_batches(reactive=False):
        rng = np.random.default_rng(734001)
        flow_rng = torch.Generator(device='cuda:0').manual_seed(734001)
        return [batch(validation, rng, flow_rng, reactive=reactive) for _ in range(16)]

    validation_plan = validation_batches(reactive=args.qualify)
    if args.qualify:
        if args.objective != 'dpt' or progress['update']:
            raise ValueError('Native qualification uses a fresh disposable DPT reactive adapter')
        # Same observation, native draw and batch: zero steering preserves the expert.
        with torch.no_grad():
            _, prepared = motor.prepare_batch(first_observation)
            draw = torch.zeros((1, 50, 32), device='cuda:0')
            original_builder = native.model.build_prefix_cache
            try:
                native.model.build_prefix_cache = lambda *a, **kw: prepared['prefix']
                expected, _ = native.model.predict_action_batch(prepared['observation'], mode='eval', noise=draw)
            finally:
                native.model.build_prefix_cache = original_builder
            deployment = FlowBridgeMotor(native, FlowBridgeCodec(action_codec.mean, action_codec.std,
                interface=args.interface), policy)
            token = torch.cat((torch.as_tensor(train[0]['observations'][0:1], device='cuda:0'),
                               torch.zeros(1, 603, device='cuda:0')), -1)
            control, _, _ = policy.step(token, deterministic=True)
            actual = deployment.execute_batch(prepared, noise=draw.cpu().numpy(),
                steering=control.cpu().numpy(), spec=deployment.codec)
            np.testing.assert_array_equal(actual, expected.detach().float().cpu().numpy())
        policy.requires_grad_(True)
        del deployment
        if args.reuse_zero:
            prior_bank = args.reuse_zero / args.interface / 'zero-native'
            audit = json.loads((prior_bank / 'AUDIT.json').read_text())
            if (not audit['passed'] or audit['episodes'] != 6
                    or audit['result_sha256'] != file_sha256(prior_bank / 'RESULT.json')):
                raise ValueError('Require independently audited complete zero-native episodes')
            prior = json.loads((prior_bank / 'RESULT.json').read_text())
            ref = prior['references'][0]
            if file_sha256(ref['path']) != ref['sha256']:
                raise ValueError('Reused native episode reference changed')
            saved_zero = torch.load(ref['path'], map_location='cpu', weights_only=False)
            bound_model = saved_zero['binding']['model']
            if (bound_model['model_sha256'] != signature['model_sha256']
                    or bound_model['norm_sha256'] != signature['norm_sha256']
                    or bound_model['runtime_commit'] != signature['runtime_commit']):
                raise ValueError('Reused native qualification has a different frozen motor')
            _json(args.output / 'REUSED-ZERO-ROLLOUT.json', dict(path=str(prior_bank / 'RESULT.json'),
                sha256=audit['result_sha256'], audit_sha256=file_sha256(prior_bank / 'AUDIT.json'),
                reason='Only validation grad context changed; sampler, motor and zero adapter unchanged'))
        else:
            # Same real six-episode protocol as the previous native qualification.
            from fluffy_bassoon.meta_rl.steering_history_native import NativeAttemptBackend
            from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal
            from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import native_video_writer
            prior = ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1/gpu/native-six-qualification-01/full/OpenDrawer/bank-000/RESULT.json'
            prior_result = json.loads(prior.read_text())
            prior_saved = torch.load(prior_result['references'][0]['path'], map_location='cpu', weights_only=False)
            zero_out = args.output / 'zero-native'
            journal = ActionJournal(zero_out / 'accounting', rank=0, world=1,
                                    attempt='zero-bridge', maximum_actions=4368)
            deployment = FlowBridgeMotor(native, FlowBridgeCodec(action_codec.mean, action_codec.std,
                interface=args.interface), policy)
            backend = NativeAttemptBackend(output=zero_out / 'attempts',
                canonical_root=ROOT / 'outputs/robocasa-eight-hour-babysit-20261104-r1/gpu/canonical-qualification-ID',
                motor=deployment, spec=deployment.codec, native_factory=factory.paired,
                journal=journal, writer_factory=native_video_writer,
                binding=dict(source_commit=args.commit, model=signature,
                             render_contract='bounded_two_intensity_levels', zero_initialized_bridge=True),
                maximum_rgb_delta=2, actual_attempts=6, fixed_case_order=True)
            rows = backend.run_bank(policy, [dict(case=prior_saved['case'], horizon=600)],
                                    lanes=1, attempts=6, deadline_unix=time.time() + 650)
            if len(rows) != 6 or backend.last_batch_sizes != [1] * 72:
                raise ValueError('Native six-episode qualification coverage changed')
            _json(zero_out / 'RESULT.json', dict(episodes=rows, references=backend.references,
                controls_charged=journal.charged(), zero_initialized_bridge=True,
                actual_batch_sizes=backend.last_batch_sizes, optimizer_updates=0,
                native_parameters_frozen=True))
            del deployment, backend
            policy.requires_grad_(True)
        zero_loss = validate_batches(policy, native.model, validation_plan, val_prepare)
        timings = []
        for step in range(args.qualify_updates):
            sample = batch(train, sampler, generator, reactive=True)
            torch.cuda.synchronize(); began = time.monotonic()
            metrics = bridge_update(policy, native.model, optimizer, sample, train_prepare,
                                    execution_batch_size=args.execution_batch)
            torch.cuda.synchronize(); timings.append(time.monotonic() - began)
            print(json.dumps(dict(phase='reactive_qualification', update=step + 1, **metrics)), flush=True)
        fitted_loss = validate_batches(policy, native.model, validation_plan, val_prepare)
        core_grad = sum(float(p.grad.abs().sum()) for p in policy.core.parameters() if p.grad is not None)
        if core_grad <= 0 or not all(math.isfinite(x) for x in timings):
            raise ValueError('The actual frozen-expert fit did not reach the history model')
        # Time the actual history-conditioned update at each execution batch size.
        sample = batch(train, sampler, generator)
        state, opt = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
        benchmarks = []
        for size in (1, 2, 4, 8):
            policy.load_state_dict(state); optimizer.load_state_dict(opt)
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); began = time.monotonic()
            metrics = bridge_update(policy, native.model, optimizer, sample, train_prepare, execution_batch_size=size)
            torch.cuda.synchronize()
            benchmarks.append(dict(execution_batch=size, seconds=time.monotonic() - began,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), **metrics))
        progress.update(update=args.qualify_updates + 1, best_loss=fitted_loss)
        save_resume(args.output / 'latest.pt', contract=contract, policy=policy, optimizer=optimizer,
            sampler=sampler, generator=generator, progress=progress)
        bridge_update(policy, native.model, optimizer, sample, train_prepare, execution_batch_size=8)
        expected = deepcopy(policy.state_dict())
        restore_resume(args.output / 'latest.pt', contract=contract, policy=policy, optimizer=optimizer,
            sampler=sampler, generator=generator)
        bridge_update(policy, native.model, optimizer, sample, train_prepare, execution_batch_size=8)
        exact_resume = all(torch.equal(v, policy.state_dict()[k]) for k, v in expected.items())
        if not exact_resume or tensor_hash(_versions(native.model)) != frozen:
            raise ValueError('Native frozen identity or exact optimizer recovery failed')
        _json(args.output / 'QUALIFICATION.json', dict(passed=True, zero_native_actions_exact=True,
            exact_next_update_resume=True, native_unchanged=True, interface=args.interface,
            zero_validation_flow_loss=zero_loss, fitted_reactive_validation_flow_loss=fitted_loss,
            reactive_updates=args.qualify_updates, mean_reactive_update_seconds=float(np.mean(timings)),
            total_optimizer_calls=args.qualify_updates + 6, zero_initialized_native_episodes=6, fresh_native_episodes=0 if args.reuse_zero else 6,
            history_core_gradient_L1=core_grad,
            full_history_update_benchmarks=benchmarks, signature=signature,
            meaning='Native parity, actual RGB/action query mapping, finite optimization and exact recovery; imitation diagnostic, not a trained meta-steerer or adaptation result'))
        reader.close()
        return

    stop = False
    def request_stop(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    started = time.monotonic()
    elapsed_before = progress['elapsed_seconds']
    while not progress['converged']:
        began = time.monotonic()
        sample = batch(train, sampler, generator)
        metrics = bridge_update(policy, native.model, optimizer, sample, train_prepare,
                                execution_batch_size=args.execution_batch)
        progress['update'] += 1
        timed_out = time.monotonic() - started >= args.max_seconds
        scheduled = progress['update'] == 1 or progress['update'] % 100 == 0
        if scheduled or timed_out or stop:
            heldout = validate_batches(policy, native.model, validation_plan, val_prepare)
            improved = progress['best_loss'] is None or heldout < progress['best_loss']
            if improved:
                progress['best_loss'] = heldout
                export_bridge_policy(args.output / 'candidate.pt', policy,
                    manifest=dict(contract=contract, update=progress['update'], validation_flow_loss=heldout))
            if scheduled:
                progress['plateau'] = advance_plateau(progress['plateau'], heldout, progress['update'],
                    minimum_updates=1000, patience=10, relative_delta=.001)
                progress['converged'] = progress['plateau']['converged']
            progress['elapsed_seconds'] = elapsed_before + time.monotonic() - started
            if tensor_hash(_versions(native.model)) != frozen:
                raise ValueError('Frozen native weights changed')
            row = dict(update=progress['update'], validation_flow_loss=heldout,
                elapsed_seconds=progress['elapsed_seconds'], update_seconds=time.monotonic() - began,
                plateau=progress['plateau'], **metrics)
            with (args.output / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row), flush=True)
            save_resume(args.output / 'latest.pt', contract=contract, policy=policy, optimizer=optimizer,
                sampler=sampler, generator=generator, progress=progress)
            _json(args.output / 'training-status.json', dict(progress,
                stop_reason='validation_plateau' if progress['converged'] else 'operational_checkpoint',
                native_parameters_frozen=True, interface=args.interface, objective=args.objective))
        if timed_out or stop:
            break
    reader.close()


if __name__ == '__main__':
    main()
