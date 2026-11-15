"""CPU-only diagnostics of frozen Q/V and archived native history interventions."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'src'))
TASKS = ('OpenDrawer', 'CloseToasterOvenDoor', 'CoffeeSetupMug',
         'PickPlaceCounterToStove', 'TurnOffStove', 'TurnOnSinkFaucet')
UPDATES = (0, 9, 19, 29)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def vector_values(critic, batch, *, mode='full'):
    """Existing causal sequence primitive; qualified against deployed stepping below."""
    import torch
    tokens, valid = batch['tokens'], batch['valid']
    tokens = torch.where(valid[..., None], tokens, 0.)
    d = critic.config.observation_dim
    tokens = torch.cat(((tokens[..., :d] - critic.observation_mean) / critic.observation_std,
                        tokens[..., d:]), -1)
    reset = batch['starts'] if mode == 'episode' else torch.zeros_like(valid)
    features = critic.core.sequence(tokens, reset=reset, valid=valid)
    if critic.action_value:
        features = torch.cat((features, torch.where(valid[..., None], batch['actions'], 0.)), -1)
    return critic.head(features).squeeze(-1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == a.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, load_dataset, trial_tokens
    from fluffy_bassoon.meta_rl.history_value import HistoryValueLearner, ValueConfig, discounted_returns, importance_weights
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    state = dict(pin=a.pin, script_sha256=sha(__file__), hostname=socket.gethostname(),
                 pid=os.getpid(), started_unix=began, GPU_hours=0, device='cpu',
                 torch_version=torch.__version__, completed=False, finished=[])
    write(a.output / 'STATE.json', state)

    # Read archived intervention outputs; no native model forward is required.
    effects = {}
    for size in (3, 6):
        for method in ('ad', 'dpt', 'dit', 'ic_iql'):
            rel = (f'fits{size}r1/{method}' if method in ('ad', 'dpt') else
                   f'resume{size}{"dit" if method == "dit" else "iciql"}01/{method}')
            path = a.source / rel / 'COMPLETE.json'
            complete = json.loads(path.read_text())
            assert complete['completed'] and complete['progress']['actor_update'] == 10000
            e = complete['steering_effect']
            effects[f'{size}_{method}'] = dict(e, complete_path=str(path), complete_sha256=sha(path),
                history_relative_to_native_rms=e['relative_steering'] * e['history_fraction'],
                history_loss_improvement_fraction=(e['reset_loss'] - e['steered_loss']) / e['reset_loss'])
    write(a.output / 'ARCHIVED-FLOW-EFFECTS.json', effects)

    # All source streams, reward-only arrays, including every intermediate update.
    corpus = {}
    for task in TASKS:
        assembly = json.loads((a.source / 'datasets01' / task / 'ASSEMBLY.json').read_text())
        corpus[task] = {}
        for split in ('train', 'validation'):
            ref = assembly['datasets']['dit_' + split]
            assert sha(ref['path']) == ref['sha256']
            with np.load(ref['path'], allow_pickle=False) as f:
                rewards = f['rewards'].reshape(-1, 30, 12)
                steps = f['teacher_steps'].reshape(-1, 30, 12)
                assert np.array_equal(steps[0, :, 0], np.arange(30))
                assert set(np.unique(rewards)) <= {0., 1.}
                success = rewards.max(axis=2)
                corpus[task][split] = dict(streams=len(success), episodes=int(success.size),
                    successful_episodes=int(success.sum()), success_by_update=success.mean(axis=0).tolist(),
                    early10_success=float(success[:, :10].mean()), late10_success=float(success[:, -10:].mean()),
                    dataset_sha256=ref['sha256'])
    write(a.output / 'SOURCE-REWARDS.json', corpus)

    for size in (3, 6):
        for method in ('dit', 'ic_iql'):
            name = f'{size}_{method}'
            audit_path = a.source / 'final-fit-audit02' / f'{name}.json'
            audit = json.loads(audit_path.read_text())
            ref = audit['checkpoint']
            assert audit['passed'] and sha(ref['path']) == ref['sha256']
            saved = torch.load(ref['path'], map_location='cpu', weights_only=False)
            c = saved['contract']
            assert c['objective'] == method and c['corpus_size'] == size
            assert saved['progress']['value_update'] == 1000 and saved['progress']['actor_update'] == 10000
            values = HistoryValueLearner(HistoryConfig(**c['history_config']), ValueConfig(**c['value_config']))
            values.load_state_dict(saved['values'], strict=True)
            values.eval().requires_grad_(False)
            del saved
            gc.collect()
            provenance = dict(checkpoint=ref, audit_sha256=sha(audit_path), datasets=c['datasets'],
                              panel_updates=UPDATES, streams='all16 validation streams per task')
            write(a.output / (name + '-INPUTS.json'), provenance)
            with gzip.open(a.output / (name + '-queries.jsonl.gz'), 'wt') as out:
                for task in TASKS[:size]:
                    d = c['datasets'][task]['validation']
                    assert sha(Path(d['path']).with_suffix('.json')) == d['manifest_sha256']
                    trials, manifest = load_dataset(d['path'], expected_sha256=d['sha256'])
                    assert len(trials) == 16
                    for update in UPDATES:
                        start = max(0, update - 11) * 12
                        end = (update + 1) * 12
                        batch = {}
                        for key, field in [('actions','executed_noise'), ('rewards','rewards'),
                                           ('dones','dones'), ('starts','episode_starts')]:
                            batch[key] = torch.from_numpy(np.stack([t[field][start:end] for t in trials], axis=1))
                        batch['valid'] = torch.ones_like(batch['dones'])
                        reset = dict(batch)
                        for mode, b in [('full', batch), ('episode', reset)]:
                            tokens = np.stack([trial_tokens(t, mode=mode)[start:end] for t in trials], axis=1)
                            tokens[0, :, c['history_config']['observation_dim']:-1] = 0
                            b['tokens'] = torch.from_numpy(tokens)
                        with torch.inference_mode():
                            q = vector_values(values.q, batch)[-12:]
                            v = vector_values(values.v, batch)[-12:]
                            qr = vector_values(values.q, reset, mode='episode')[-12:]
                            vr = vector_values(values.v, reset, mode='episode')[-12:]
                            if task == TASKS[0] and update == 19:
                                errors = {}
                                for label, critic in [('q', values.q), ('v', values.v)]:
                                    for mode, source in [('full', batch), ('episode', reset)]:
                                        small = {k: x[:, :2] for k,x in source.items()}
                                        reference = critic.sequence(small, mode=mode)
                                        fast = vector_values(critic, small, mode=mode)
                                        # Float32 accumulation order differs between batched and incremental attention.
                                        torch.testing.assert_close(fast, reference, rtol=1e-5, atol=1e-6)
                                        errors[label + '_' + mode] = float((fast-reference).abs().max())
                                write(a.output / (name + '-NUMERICS.json'), dict(
                                    passed=True, sequence_length=len(batch['tokens']), lanes=2,
                                    reference='HistoryCritic.sequence', maximum_absolute_errors=errors,
                                    rtol=1e-5, atol=1e-6))
                            w = importance_weights(q-v, values.config)
                            wr = importance_weights(qr-vr, values.config)
                        arrays = {k: x.numpy() for k,x in dict(q=q,v=v,weight=w,reset_q=qr,reset_v=vr,reset_weight=wr).items()}
                        assert all(np.isfinite(x).all() for x in arrays.values())
                        if update == 0:
                            assert torch.equal(q, qr) and torch.equal(v, vr), 'No previous episode: full/reset must match'
                        expected = np.exp(np.clip(arrays['q']-arrays['v'], -20, np.log(values.config.maximum_weight)))
                        assert np.allclose(arrays['weight'], expected, rtol=1e-6)
                        for slot,t in enumerate(trials):
                            returns = discounted_returns(t['rewards'], t['dones'], .99)
                            success = int(t['rewards'][update*12:end].max())
                            for query in range(12):
                                row = dict(model=name, task=task, stream=slot+112, update=update,
                                           query=query, success=success, return_=float(returns[update*12+query]))
                                row.update({k:float(x[query,slot]) for k,x in arrays.items()})
                                out.write(json.dumps(row, allow_nan=False) + '\n')
                        state['last'] = dict(model=name, task=task, update=update, elapsed_seconds=time.time()-began)
                        write(a.output / 'STATE.json', state)
                        print(json.dumps(state['last']), flush=True)
                    del trials
                    gc.collect()
            del values
            gc.collect()
            state['finished'].append(name)
            write(a.output / 'STATE.json', state)
    state.update(completed=True, ended_unix=time.time(), elapsed_seconds=time.time()-began)
    state['files'] = {p.name:sha(p) for p in a.output.iterdir() if p.name != 'STATE.json'}
    write(a.output / 'STATE.json', state)


if __name__ == '__main__':
    main()
