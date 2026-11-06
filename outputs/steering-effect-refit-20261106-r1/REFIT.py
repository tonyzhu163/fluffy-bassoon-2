"""Continue a copied KV-AD recovery with only the corrected plateau contract."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import time

from RUNTIME import Runtime, PARENT
from REPLAY import replay
import ADMISSION


def identical(first, second):
    import numpy as np
    import torch
    if torch.is_tensor(first):
        return torch.is_tensor(second) and torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(identical(v, second[k]) for k, v in first.items())
    if isinstance(first, (list, tuple)):
        return isinstance(second, type(first)) and len(first) == len(second) and all(identical(a, b) for a, b in zip(first, second))
    return first == second


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('code', 'probe', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--max-seconds', type=int, required=True)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    if args.max_seconds < 60:
        raise ValueError('Reserve at least one minute of actual fitting time')
    run_contract_path = Path(__file__).with_name('CONTRACT.json')
    run_contract = json.loads(run_contract_path.read_text())
    runtime = Runtime(args.code, args.commit, args.probe, run_contract)
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.flow_history import FlowBridgeConfig, HistoryFlowSteeringPolicy, export_bridge_policy, load_bridge_policy
    from fluffy_bassoon.meta_rl.flow_bridge_train import bridge_update, save_resume, restore_resume, validate_batches, steering_effect
    from fluffy_bassoon.meta_rl.vla_history_train import advance_plateau
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _json
    index_path = PARENT / 'finalized-fits-20261106-r1/INDEX.json'
    reference = json.loads(index_path.read_text())['models']['kv_bridge/ad']
    recovery = reference['artifacts']['latest.pt']
    candidate = reference['artifacts']['candidate.pt']
    if runtime.sha(recovery['path']) != recovery['sha256']:
        raise ValueError('Preserved complete parent recovery changed')
    saved = torch.load(recovery['path'], map_location='cpu', weights_only=False)
    if runtime.sha(candidate['path']) != candidate['sha256']:
        raise ValueError('Preserved selected parent candidate changed')
    selected_parent = torch.load(candidate['path'], map_location='cpu', weights_only=False)
    if (set(selected_parent['policy']) != set(saved['policy'])
            or any(not torch.equal(v, selected_parent['policy'][k]) for k, v in saved['policy'].items())):
        raise ValueError('Primary selected parent must exactly match its complete terminal recovery')
    del selected_parent
    parent_contract = saved['contract']
    if (parent_contract['interface'], parent_contract['objective']) != ('kv_bridge', 'ad'):
        raise ValueError('Primary continuation must be the same KV-AD learner')
    config = dict(parent_contract['config']); config['kv_layers'] = tuple(config['kv_layers'])
    policy = HistoryFlowSteeringPolicy(FlowBridgeConfig(**config)).to('cuda:0')
    optimizer = torch.optim.AdamW(policy.parameters(), lr=parent_contract['learning_rate'], weight_decay=0.)
    sampler = np.random.default_rng(0)
    flow_rng = torch.Generator(device='cuda:0').manual_seed(771105)
    contract = deepcopy(parent_contract)
    contract.update(source_commit=args.commit, stopping_rule='relative_abs_reference_v2',
        parent_recovery_sha256=recovery['sha256'], parent_candidate_sha256=candidate['sha256'],
        run_contract_sha256=runtime.sha(run_contract_path))
    contract['schedule']['relative_scale'] = 'abs(reference) with numerical epsilon 1e-12'
    contract = json.loads(json.dumps(contract))
    train, train_manifest = runtime.data('ad', 'train')
    validation, val_manifest = runtime.data('ad', 'validation')
    if (contract['train_sha256'] != runtime.assembly['datasets']['ad_train']['sha256']
            or contract['validation_sha256'] != runtime.assembly['datasets']['ad_validation']['sha256']
            or set(train_manifest['metadata']['trial_ids']) & set(val_manifest['metadata']['trial_ids'])):
        raise ValueError('Preserve the original disjoint source split')
    train_prepare, val_prepare = runtime.prepare(train, train_manifest), runtime.prepare(validation, val_manifest)
    validation_plan = runtime.validation_plan(policy, validation, 'ad')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'latest.pt').exists():
        if json.loads((args.output / 'contract.json').read_text()) != contract:
            raise ValueError('An operational resume must preserve the new fit contract')
        progress = restore_resume(args.output / 'latest.pt', contract=contract, policy=policy,
            optimizer=optimizer, sampler=sampler, generator=flow_rng)
    else:
        if any(args.output.iterdir()):
            raise ValueError('Use a fresh output for the explicit stopping-rule migration')
        progress = restore_resume(recovery['path'], contract=parent_contract, policy=policy,
            optimizer=optimizer, sampler=sampler, generator=flow_rng)
        if progress['update'] != reference['final_update'] or reference['selected_candidate_update'] != progress['update']:
            raise ValueError('Require the selected terminal primary KV-AD recovery')
        original_rows = [json.loads(s) for s in (PARENT / 'fits/kv_bridge/ad/metrics.jsonl').read_text().splitlines()]
        corrected = replay(original_rows, progress['update'], run_contract['refit'])
        if corrected['plateau']['converged']:
            raise ValueError('Corrected historical stop is already met; no continuation justified')
        before = deepcopy(progress)
        progress.update(plateau=corrected['plateau'], converged=False, selected_update=reference['selected_candidate_update'])
        _json(args.output / 'contract.json', contract)
        export_bridge_policy(args.output / 'candidate.pt', policy, manifest=dict(contract=contract,
            update=progress['selected_update'], validation_flow_loss=progress['best_loss']))
        save_resume(args.output / 'latest.pt', contract=contract, policy=policy, optimizer=optimizer,
            sampler=sampler, generator=flow_rng, progress=progress)
        migrated = torch.load(args.output / 'latest.pt', map_location='cpu', weights_only=False)
        for field in ('policy', 'optimizer', 'rng', 'flow_generator'):
            if not identical(saved[field], migrated[field]):
                raise ValueError(f'Stopping-rule migration changed preserved {field}')
        del migrated
        _json(args.output / 'MIGRATION.json', dict(parent_recovery=recovery, parent_candidate=candidate,
            parent_progress=before, corrected_replay=corrected, new_progress=progress,
            unchanged_policy_Adam_sampler_global_and_flow_RNG=True,
            source_commit=args.commit, change='Only corrected stopping-rule state and explicit new contract/output identity'))
    del saved
    if progress['converged']:
        raise ValueError('This corrected fit already converged; do not perform more updates')
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic(); elapsed_before = progress['elapsed_seconds']
    while not progress['converged']:
        began = time.monotonic()
        sample = runtime.batch(policy, train, 'ad', sampler, flow_rng)
        metrics = bridge_update(policy, runtime.native.model, optimizer, sample, train_prepare, execution_batch_size=8)
        progress['update'] += 1
        scheduled = progress['update'] % 100 == 0
        timed_out = time.monotonic() - started >= args.max_seconds or time.time() >= args.deadline
        if scheduled or timed_out or stopping:
            heldout = validate_batches(policy, runtime.native.model, validation_plan, val_prepare)
            if heldout < progress['best_loss']:
                progress.update(best_loss=heldout, selected_update=progress['update'])
                export_bridge_policy(args.output / 'candidate.pt', policy, manifest=dict(contract=contract,
                    update=progress['update'], validation_flow_loss=heldout))
            if scheduled:
                progress['plateau'] = advance_plateau(progress['plateau'], heldout, progress['update'],
                    minimum_updates=1000, patience=10, relative_delta=.001)
                progress['converged'] = progress['plateau']['converged']
            progress['elapsed_seconds'] = elapsed_before + time.monotonic() - started
            runtime.check_frozen()
            row = dict(update=progress['update'], validation_flow_loss=heldout,
                elapsed_seconds=progress['elapsed_seconds'], update_seconds=time.monotonic() - began,
                plateau=progress['plateau'], **metrics)
            with (args.output / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(row, allow_nan=False), flush=True)
            save_resume(args.output / 'latest.pt', contract=contract, policy=policy, optimizer=optimizer,
                sampler=sampler, generator=flow_rng, progress=progress)
            _json(args.output / 'training-status.json', dict(progress,
                stopping_rule='relative_abs_reference_v2', native_parameters_frozen=True,
                interface='kv_bridge', objective='ad',
                stop_reason='validation_plateau' if progress['converged'] else 'operational_checkpoint'))
        if stopping or timed_out:
            break
    # Measurement never changes or selects the fitted weights and never dispatches robots.
    selected, _ = load_bridge_policy(args.output / 'candidate.pt',
        expected_sha256=runtime.sha(args.output / 'candidate.pt'), device='cuda:0')
    effect = steering_effect(selected, runtime.native.model, validation_plan, val_prepare)
    admission = ADMISSION.publish(args.output, run_contract_path, effect, args.commit)
    runtime.check_frozen()
    print(json.dumps(dict(stage='pre_readout_gate', admission=admission), allow_nan=False), flush=True)
    runtime.reader.close()


if __name__ == '__main__':
    main()
