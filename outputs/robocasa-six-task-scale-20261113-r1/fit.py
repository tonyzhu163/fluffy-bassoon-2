"""Matched KV fits: fresh corpus-specific Q/V, then exactly 10,000 actor updates."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
import time

from corpus import balanced_batch, load_corpus
from labels import sha, write

ROOT = Path('/mowenjie/fluffy-bassoon-2')
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--corpora', type=Path, required=True)
    p.add_argument('--corpora-sha256', required=True)
    p.add_argument('--corpus-size', type=int, choices=(3, 6), required=True)
    p.add_argument('--objective', choices=('ad', 'dpt', 'dit', 'ic_iql'), required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cold', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--execution-batch', type=int, default=8)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--raw-receipt-sha256', required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--resume-sha256')
    a = p.parse_args()
    assert sys.platform == 'linux' and socket.gethostname().startswith('job-')
    sys.path[:0] = [str(CODE/'src'), os.environ['RLINF_PATH']]
    import numpy as np
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.rlinf.native_ppo import checked_checkout, tensor_hash
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native, _versions
    from fluffy_bassoon.meta_rl.flow_history import (DirectActionCodec, FlowBridgeConfig,
        HistoryFlowSteeringPolicy, FlowDirectMotor, sample_bridge_query_batch, export_bridge_policy)
    from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
    from fluffy_bassoon.meta_rl.flow_bridge_train import bridge_update, query_weights, validate_batches, steering_effect
    from fluffy_bassoon.meta_rl.vla_history import HistoryConfig
    from fluffy_bassoon.meta_rl.history_value import HistoryValueLearner, ValueConfig, sample_value_batch, learning_step
    from fluffy_bassoon.meta_rl.vla_history_train import _rng_state, _restore_rng
    from fluffy_bassoon.meta_rl.warmstart import atomic_save
    checked_checkout(CODE, a.pin)
    os.environ['LYCO_BAIGE_GPU_MAP'] = str(a.probe)
    verify_selected_cuda(torch)
    torch.set_num_threads(4)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    corpus = load_corpus(a.corpora, a.corpora_sha256, a.corpus_size, a.objective)
    train, validation = corpus['data']['train'], corpus['data']['validation']
    identity = corpus['identity']
    codec = DirectActionCodec(tuple(identity['codec']['mean']), tuple(identity['codec']['std']))
    hc = HistoryConfig(2073, 600, width=256, layers=6, heads=8, context_length=192)
    vc = ValueConfig(a.objective) if a.objective in ('dit', 'ic_iql') else None
    contract = dict(source_commit=a.pin, corpus_size=a.corpus_size, objective=a.objective,
        corpus_spec_sha256=a.corpora_sha256, tasks=list(corpus['tasks']), datasets=corpus['bindings'],
        native_identity=identity, interface='kv_bridge', seed=0, logical_batch=8, max_episodes=12,
        task_sampling='uniform task, then uniform training stream', learning_rate=.0003,
        architecture=dict(width=256, layers=6, heads=8, context_length=192, kv_rank=4, scale=.1),
        history_config=asdict(hc), value_config=asdict(vc) if vc else None,
        value_updates=1000 if vc else 0, actor_updates=10000, checkpoint_selection='fixed final',
        validation='16 fixed batches every1000 updates; diagnostic only',
        value_initialization='fresh independent Q/V for this objective and corpus; fixed update1000',
        normalization='this corpus training streams only', noise_contract=codec.record())
    # Contract excludes operational output paths, deadlines and execution batches.
    contract = json.loads(json.dumps(contract))
    a.output.mkdir(parents=True, exist_ok=False); a.cold.mkdir(parents=True, exist_ok=False)
    write(a.cold/'CONTRACT.json', contract)
    sampler = np.random.default_rng(0)
    generator = torch.Generator(device='cuda:0').manual_seed(771105)
    progress = dict(phase='value' if vc else 'actor', value_update=0, actor_update=0, elapsed_seconds=0.)
    saved = None
    if a.resume:
        assert a.resume_sha256 and sha(a.resume) == a.resume_sha256
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        assert saved['format'] == 'matched_kv_fit_v1' and saved['contract'] == contract
        progress = saved['progress']
        assert progress['actor_update'] < 10000
    stop = False
    def stopping(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stopping)
    began = time.monotonic(); elapsed_before = progress['elapsed_seconds']
    values = value_optimizer = policy = optimizer = native = None
    banked = []

    def checkpoint(metrics=None):
        progress['elapsed_seconds'] = elapsed_before + time.monotonic() - began
        state = dict(format='matched_kv_fit_v1', contract=contract, progress=dict(progress),
            values=values.state_dict() if values is not None else None,
            value_optimizer=value_optimizer.state_dict() if value_optimizer is not None else None,
            policy=policy.state_dict() if policy is not None else None,
            optimizer=optimizer.state_dict() if optimizer is not None else None,
            rng=_rng_state(sampler), flow_generator=generator.get_state())
        local = a.output/'latest.pt'; atomic_save(local, state)
        name = f"{progress['phase']}-{progress['value_update']:05d}-{progress['actor_update']:05d}.pt"
        dest = a.cold/name
        assert not dest.exists()
        with local.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst, 16*2**20)
        digest = sha(local); assert sha(dest) == digest
        record = dict(progress, checkpoint=dict(path=str(dest), sha256=digest),
            complete=progress['actor_update'] == 10000, checked_unix=time.time(),
            execution_batch=a.execution_batch, metrics=metrics)
        write(a.cold/'PROGRESS.json', record)
        banked.append(dest)
        # Two complete hash-verified resumable snapshots for this attempt.
        if len(banked) > 2:
            banked.pop(0).unlink()
        print(json.dumps(record, allow_nan=False), flush=True)

    def sample(factory, collection, model, rng, **kwargs):
        return balanced_batch(factory, collection, model, rng, task_count=a.corpus_size,
            streams_per_task=112 if collection is train else 16, batch_size=8, max_episodes=12,
            discount=.99, **kwargs)

    if vc:
        torch.manual_seed(0)
        values = HistoryValueLearner(hc, vc).to('cuda:0')
        values.fit_normalization(train)
        value_optimizer = torch.optim.AdamW([*values.q.parameters(), *values.v.parameters()], lr=3e-4, weight_decay=0.)
        if saved:
            values.load_state_dict(saved['values'], strict=True)
            value_optimizer.load_state_dict(saved['value_optimizer'])
            _restore_rng(saved['rng'], sampler); generator.set_state(saved['flow_generator'])
        while progress['value_update'] < 1000:
            values.train()
            started = time.monotonic()
            batch = sample(sample_value_batch, train, values.actor, sampler)
            metrics = learning_step(values, value_optimizer, batch, phase='value', execution_batch_size=a.execution_batch)
            progress['value_update'] += 1
            metrics['update_seconds'] = time.monotonic() - started
            final = progress['value_update'] == 1000
            if final:
                rng = np.random.default_rng(734001)
                with torch.no_grad():
                    values.eval()
                    losses = [float(values.value_loss(sample(sample_value_batch, validation, values.actor, rng))[0]) for _ in range(16)]
                assert np.isfinite(losses).all()
                metrics['validation_value_loss'] = float(np.mean(losses))
                progress['phase'] = 'actor'
            if progress['value_update'] == 1 or progress['value_update'] % 100 == 0 or stop or time.time() >= a.deadline:
                checkpoint(metrics)
            if stop or time.time() >= a.deadline:
                return
        values.eval().requires_grad_(False)
    if stop or time.time() >= a.deadline:
        return

    model_path = ROOT/'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
    norm = ROOT/'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'
    os.environ.update(ROBOCASA_NATIVE_OUTPUT=str(a.output/'native'), ROBOCASA_PI05_TORCH_MODEL=str(model_path),
                      ROBOCASA_PI05_NATIVE_NORM=str(norm))
    native, _, signature = _load_native(dict(model_directory=str(model_path), norm_path=str(norm),
        source_commit=a.pin, runtime_commit=identity['runtime_commit']), CODE)
    assert signature['model_sha256'] == identity['native_model_sha256']
    assert signature['norm_sha256'] == identity['native_norm_sha256']
    raw_receipt = a.raw_root/'CACHE.json'
    assert sha(raw_receipt) == a.raw_receipt_sha256
    raw = json.loads(raw_receipt.read_text())
    assert raw['completed'] and set(corpus['tasks']) <= raw['tasks'].keys()
    sources = {}
    for task, source in corpus['sources'].items():
        item = raw['tasks'][task]
        assert item['sha256'] == source['archive_sha256'] and item['path'] == str(a.raw_root/task/'raw.tar')
        assert Path(item['path']).stat().st_size == item['bytes']
        sources[task] = dict(source, archive=item['path'])
    reader = FlowSourceQueries(sources, codec, cached_members=8)
    motor = FlowDirectMotor(native, codec)
    first = reader.observations([(0,0)], train, corpus['ids']['train'])
    with torch.no_grad():
        _, prepared = motor.prepare_batch(first)
    cache = prepared['prefix'][2]
    config = FlowBridgeConfig(interface='kv_bridge', width=256, layers=6, heads=8, context_length=192,
        kv_heads=cache[0][0].shape[2], head_dim=cache[0][0].shape[3], kv_rank=4,
        kv_layers=tuple(range(len(cache))), scale=.1)
    del prepared, cache; native._prepared = None
    # Native loading/value fitting must not change policy initialization or its sampling sequence.
    torch.manual_seed(0)
    policy = HistoryFlowSteeringPolicy(config).to('cuda:0')
    policy.fit_observation_normalization(train)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
    sampler = np.random.default_rng(0); generator.manual_seed(771105)
    if saved and saved['policy'] is not None:
        policy.load_state_dict(saved['policy'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        _restore_rng(saved['rng'], sampler); generator.set_state(saved['flow_generator'])
    frozen = tensor_hash(_versions(native.model))
    stats = json.loads(norm.read_text()); stats = stats.get('norm_stats', stats)['actions']

    def prepare(collection, names):
        def queries(indices):
            observation = reader.observations(indices, collection, names)
            try:
                with torch.no_grad():
                    _, prepared = motor.prepare_batch(observation)
                _, mask, cache = prepared['prefix']
                return prepared['observation'], cache, mask
            finally:
                native._prepared = None
        return queries
    train_prepare = prepare(train, corpus['ids']['train']); val_prepare = prepare(validation, corpus['ids']['validation'])

    def batch(collection, rng, flow_rng):
        item = sample(sample_bridge_query_batch, collection, policy, rng, action_codec=codec,
            native_action_stats=stats, objective=a.objective)
        item['noise'] = torch.randn(item['target_actions'].shape, generator=flow_rng, device='cuda:0')
        item['timesteps'] = torch.rand(8, generator=flow_rng, device='cuda:0')
        item['weights'] = query_weights(item, a.objective, values)
        return item
    val_rng = np.random.default_rng(734001)
    val_flow = torch.Generator(device='cuda:0').manual_seed(734001)
    validation_plan = [batch(validation, val_rng, val_flow) for _ in range(16)]
    timings = []
    try:
        while progress['actor_update'] < 10000:
            started = time.monotonic()
            item = batch(train, sampler, generator)
            metrics = bridge_update(policy, native.model, optimizer, item, train_prepare,
                execution_batch_size=a.execution_batch)
            torch.cuda.synchronize(); timings.append(time.monotonic()-started)
            progress['actor_update'] += 1
            final = progress['actor_update'] == 10000
            scheduled = progress['actor_update'] == 1 or progress['actor_update'] % 100 == 0
            if progress['actor_update'] % 1000 == 0:
                metrics['validation_flow_loss'] = validate_batches(policy, native.model, validation_plan, val_prepare)
            if scheduled or final or stop or time.time() >= a.deadline:
                assert tensor_hash(_versions(native.model)) == frozen
                metrics.update(mean_update_seconds=float(np.mean(timings)), peak_cuda_bytes=torch.cuda.max_memory_allocated())
                checkpoint(metrics); timings.clear()
            if final:
                effect = steering_effect(policy, native.model, validation_plan, val_prepare)
                assert tensor_hash(_versions(native.model)) == frozen
                candidate = a.output/'candidate.pt'
                export_bridge_policy(candidate, policy, manifest=dict(contract=dict(contract, config=asdict(config)),
                    update=10000, steering_effect=effect))
                dest = a.cold/'candidate.pt'; shutil.copyfile(candidate, dest); assert sha(dest) == sha(candidate)
                write(a.cold/'COMPLETE.json', dict(completed=True, candidate=dict(path=str(dest),sha256=sha(dest)),
                    progress=progress, contract=contract, config=asdict(config), steering_effect=effect,
                    native_parameters_unchanged=True, checked_unix=time.time()))
                return
            if stop or time.time() >= a.deadline:
                return
    finally:
        reader.close()


if __name__ == '__main__':
    main()
