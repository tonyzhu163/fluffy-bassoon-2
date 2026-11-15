"""Within-task/update/query discrimination with whole-stream uncertainty."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert sys.platform == 'linux'
    import numpy as np
    state = json.loads((a.input / 'STATE.json').read_text())
    assert state['completed']
    for name,digest in state['files'].items():
        assert hashlib.sha256((a.input/name).read_bytes()).hexdigest() == digest
    a.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261115)
    draws = 4000

    def interval(values):
        good = np.asarray(values)
        good = good[np.isfinite(good)]
        return np.quantile(good, [.025,.975]).tolist() if len(good) else None

    def auc(score, success, counts):
        # Pair only distinct streams at the same teacher update and query index.
        score = score.transpose(1,2,0)
        y = success.transpose(1,2,0)
        comparable = ((y[..., :, None] == 1) & (y[..., None, :] == 0))
        wins = (score[..., :, None] > score[..., None, :]) + .5 * (score[..., :, None] == score[..., None, :])
        num = (comparable*wins).sum(axis=(0,1))
        den = comparable.sum(axis=(0,1))
        bnum = np.einsum('bi,ij,bj->b', counts, num, counts)
        bden = np.einsum('bi,ij,bj->b', counts, den, counts)
        bs = np.divide(bnum,bden,out=np.full(draws,np.nan),where=bden>0)
        return dict(auc=float(num.sum()/den.sum()) if den.sum() else None,
                    bootstrap95=interval(bs), comparable_pairs=int(den.sum()),
                    bootstrap_draws_with_pairs=int((bden>0).sum()))

    result = dict(source_pin=state['pin'], input_state=state, bootstrap_draws=draws,
        bootstrap_seed=20261115, bootstrap_unit='whole validation stream within task',
        discrimination='successful versus unsuccessful episode, matched teacher update and query position',
        limitation='Observational outcome association; no counterfactual action-quality labels. One fitted seed per model.',
        models={})
    for path in sorted(a.input.glob('*-queries.jsonl.gz')):
        with gzip.open(path,'rt') as f:
            rows = [json.loads(line) for line in f]
        model = rows[0]['model']
        result['models'][model] = {}
        for task in sorted({r['task'] for r in rows}):
            selected = sorted([r for r in rows if r['task']==task], key=lambda r:(r['stream'],r['update'],r['query']))
            assert len(selected)==16*4*12
            arrays = {key:np.array([r[key] for r in selected]).reshape(16,4,12)
                      for key in ('q','v','weight','reset_q','reset_v','reset_weight','success','return_')}
            counts = rng.multinomial(16, np.full(16,1/16), size=draws)
            summaries = {}
            for view, sl in [('first_query',slice(0,1)),('all_queries',slice(None))]:
                v = {k:x[:,:,sl] for k,x in arrays.items()}
                y,w = v['success'],v['weight']
                adv = v['q']-v['v']
                ra = v['reset_q']-v['reset_v']
                n = y.size
                per_stream_n = y[0].size
                wy = (w*y).sum(axis=(1,2)); ws = w.sum(axis=(1,2)); ys=y.sum(axis=(1,2))
                lift = float((w*y).sum()/w.sum()-y.mean())
                blift = (counts@wy)/(counts@ws) - (counts@ys)/(16*per_stream_n)
                summaries[view] = dict(queries=n, physical_episodes=64, success_fraction=float(y.mean()),
                    weight_quantiles=dict(zip(['min','p05','p50','p95','max'],np.quantile(w,[0,.05,.5,.95,1]).tolist())),
                    weight_mean=float(w.mean()), weight_std=float(w.std()),
                    effective_sample_fraction=float(w.sum()**2/(n*(w*w).sum())),
                    mean_weight_success=float(w[y==1].mean()) if y.any() else None,
                    mean_weight_failure=float(w[y==0].mean()) if (y==0).any() else None,
                    weighted_success_lift_pp=100*lift,
                    weighted_success_lift_bootstrap95_pp=(100*np.asarray(interval(blift))).tolist(),
                    advantage_success=auc(adv,y,counts), q_success=auc(v['q'],y,counts),
                    reset_advantage_success=auc(ra,y,counts),
                    advantage_history_rms=float(np.sqrt(np.mean((adv-ra)**2))),
                    q_return_mse=float(np.mean((v['q']-v['return_'])**2)),
                    v_return_mse=float(np.mean((v['v']-v['return_'])**2)),
                    return_variance=float(v['return_'].var()))
            result['models'][model][task] = summaries
    (a.output/'RESULT.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({m:{t:{'auc':s['first_query']['advantage_success'],
        'lift_pp':s['all_queries']['weighted_success_lift_pp'],
        'ess_fraction':s['all_queries']['effective_sample_fraction']}
        for t,s in tasks.items()} for m,tasks in result['models'].items()},indent=2))


if __name__ == '__main__':
    main()
