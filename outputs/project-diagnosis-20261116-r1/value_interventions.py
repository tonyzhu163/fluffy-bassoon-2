"""CPU frozen-value input interventions; prediction checks, not causal Q labels."""
import argparse
import gc
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'src'))
spec = importlib.util.spec_from_file_location('prior_diagnostic', CODE / 'outputs/robocasa-dit-diagnostic-20261115-r1/diagnose.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin', required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sys.platform == 'linux'
    assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == a.pin
    assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain', '--untracked-files=no'])
    import numpy as np
    import torch
    from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, load_dataset, trial_tokens
    from fluffy_bassoon.meta_rl.history_value import HistoryValueLearner, ValueConfig, discounted_returns
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    state = dict(pin=a.pin, script_sha256=prior.sha(__file__), GPU_hours=0, completed=False,
                 interpretation='Input relevance for observed returns; donor actions have no measured counterfactual returns',
                 donor='Next validation stream, cyclic +1, same task/update/query; no donor outcome selection',
                 history='Replace prior complete episodes plus their final shifted feedback; retain current-episode observations/actions/rewards',
                 reward_blind='Zero only prior-episode reward tokens, retaining other observations and actions',
                 models={})
    old_state = json.loads((a.diagnostic / 'inference01/STATE.json').read_text())
    assert old_state['completed']
    for size in (3, 6):
        for method in ('dit', 'ic_iql'):
            name = f'{size}_{method}'
            audit = json.loads((a.source / 'final-fit-audit02' / f'{name}.json').read_text())
            ref = audit['checkpoint']
            assert audit['passed'] and prior.sha(ref['path']) == ref['sha256']
            saved = torch.load(ref['path'], map_location='cpu', weights_only=False)
            c = saved['contract']
            model = HistoryValueLearner(HistoryConfig(**c['history_config']), ValueConfig(**c['value_config']))
            model.load_state_dict(saved['values'], strict=True)
            model.eval().requires_grad_(False)
            del saved
            old_path = a.diagnostic / 'inference01' / f'{name}-queries.jsonl.gz'
            assert prior.sha(old_path) == old_state['files'][old_path.name]
            with gzip.open(old_path, 'rt') as f:
                originals = {(r['task'], r['stream'], r['update'], r['query']): r for r in map(json.loads, f)}
            rows = []
            error = 0.
            for task in prior.TASKS[:size]:
                ref_data = c['datasets'][task]['validation']
                assert prior.sha(Path(ref_data['path']).with_suffix('.json')) == ref_data['manifest_sha256']
                trials, _ = load_dataset(ref_data['path'], expected_sha256=ref_data['sha256'])
                for update in prior.UPDATES:
                    start, end = max(0, update - 11) * 12, (update + 1) * 12
                    nprev = end - start - 12
                    d, actdim = model.q.config.observation_dim, model.q.config.noise_dim
                    b = {}
                    for key, field in [('actions', 'executed_noise'), ('dones', 'dones'), ('starts', 'episode_starts')]:
                        b[key] = torch.from_numpy(np.stack([t[field][start:end] for t in trials], axis=1))
                    b['valid'] = torch.ones_like(b['dones'])
                    full = np.stack([trial_tokens(t)[start:end] for t in trials], axis=1)
                    full[0, :, d:-1] = 0
                    b['tokens'] = torch.from_numpy(full)
                    donor = dict(b)
                    donor['tokens'] = b['tokens'].clone()
                    if nprev:
                        donor['tokens'][:nprev] = b['tokens'][:nprev].roll(-1, dims=1)
                        donor['tokens'][nprev, :, d:-1] = b['tokens'][nprev, :, d:-1].roll(-1, dims=0)
                    blind = dict(b)
                    blind['tokens'] = b['tokens'].clone()
                    blind['tokens'][:nprev + 1, :, d + actdim] = 0
                    alt_action = dict(b)
                    alt_action['actions'] = b['actions'].roll(-1, dims=1)
                    with torch.inference_mode():
                        values = {'q': prior.vector_values(model.q, b)[-12:],
                                  'v': prior.vector_values(model.v, b)[-12:],
                                  'q_donor_action': prior.vector_values(model.q, alt_action)[-12:]}
                        for label, view in [('donor_history', donor), ('blind_prior_reward', blind)]:
                            for which in ('q', 'v'):
                                values[which + '_' + label] = prior.vector_values(getattr(model, which), view)[-12:]
                        if update == 0:
                            for which in ('q', 'v'):
                                assert torch.equal(values[which], values[which + '_donor_history'])
                                assert torch.equal(values[which], values[which + '_blind_prior_reward'])
                    arrays = {k: v.numpy() for k, v in values.items()}
                    assert all(np.isfinite(v).all() for v in arrays.values())
                    for slot, trial in enumerate(trials):
                        rewards = trial['rewards'][update * 12:end]
                        post = np.cumsum(rewards) - rewards > 0
                        returns = discounted_returns(trial['rewards'], trial['dones'], .99)
                        for query in range(12):
                            old = originals[(task, slot + 112, update, query)]
                            row = dict(task=task, stream=slot + 112, update=update, query=query,
                                       before_success=bool(not post[query]), return_=float(returns[update*12 + query]),
                                       reset_q=old['reset_q'], reset_v=old['reset_v'])
                            row.update({k: float(v[query, slot]) for k, v in arrays.items()})
                            for which in ('q', 'v'):
                                error = max(error, abs(row[which] - old[which]))
                            assert row['return_'] == old['return_']
                            rows.append(row)
                    print(name, task, update, flush=True)
                del trials
            # Same vector inference and checkpoint must reproduce the earlier frozen panel.
            assert error <= 1e-6
            raw = a.output / f'{name}-queries.jsonl.gz'
            with gzip.open(raw, 'wt') as f:
                for row in rows:
                    f.write(json.dumps(row, allow_nan=False) + '\n')
            summary = {}
            for task in prior.TASKS[:size]:
                summary[task] = {}
                for panel in ('before_success', 'first_query'):
                    select = [r for r in rows if r['task'] == task and r['before_success'] and (panel != 'first_query' or r['query'] == 0)]
                    predictions = ('q', 'v', 'q_donor_action', 'q_donor_history', 'v_donor_history',
                                   'q_blind_prior_reward', 'v_blind_prior_reward', 'reset_q', 'reset_v')
                    mse = {key: float(np.mean([(r[key] - r['return_'])**2 for r in select])) for key in predictions}
                    deltas = {}
                    for altered, base in [('q_donor_action', 'q'), ('q_donor_history', 'q'), ('v_donor_history', 'v'),
                                          ('q_blind_prior_reward', 'q'), ('v_blind_prior_reward', 'v'),
                                          ('reset_q', 'q'), ('reset_v', 'v')]:
                        eligible = select if altered == 'q_donor_action' else [r for r in select if r['update'] > 0]
                        numer = np.zeros(16); denom = np.zeros(16)
                        for r in eligible:
                            numer[r['stream']-112] += (r[altered]-r['return_'])**2-(r[base]-r['return_'])**2
                            denom[r['stream']-112] += 1
                        counts = np.random.default_rng(20261116).multinomial(16, np.full(16, 1/16), size=4000)
                        bnum, bden = counts @ numer, counts @ denom
                        bs = np.divide(bnum, bden, out=np.full(4000, np.nan), where=bden > 0)
                        deltas[altered] = dict(queries=len(eligible), altered_minus_real_mse=float(numer.sum()/denom.sum()),
                            bootstrap95=np.nanquantile(bs, [.025, .975]).tolist(),
                            prediction_rms_change=float(np.mean([(r[altered]-r[base])**2 for r in eligible])**.5))
                    summary[task][panel] = dict(queries=len(select), mse=mse, interventions=deltas)
            state['models'][name] = dict(checkpoint=ref, raw_file=raw.name, raw_sha256=prior.sha(raw),
                                         reproduced_previous_max_abs_error=error, summary=summary)
            prior.write(a.output / 'RESULT.json', state)
            del model, rows
            gc.collect()
    state.update(completed=True, elapsed_seconds=time.time()-began)
    prior.write(a.output / 'RESULT.json', state)


if __name__ == '__main__':
    main()
