"""Frozen-checkpoint feedback interventions and history-policy diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

VIEWS = ('real', 'masked', 'shuffled', 'demo_only')


def metrics(pred, truth, select):
    hit = pred[select] == truth[select, None]
    base = hit[:, 0]
    n = len(hit)
    comparisons = {}
    for j, view in enumerate(VIEWS[1:], 1):
        delta = base.astype(float) - hit[:, j]
        se = float(delta.std(ddof=1) / np.sqrt(n))
        comparisons[view] = dict(real_only=int((base & ~hit[:, j]).sum()),
            intervention_only=int((~base & hit[:, j]).sum()),
            prediction_changed=int((pred[select, 0] != pred[select, j]).sum()),
            difference=float(delta.mean()), paired_normal95=[float(delta.mean()-1.96*se), float(delta.mean()+1.96*se)])
    return dict(n=n, accuracy={v: float(hit[:, j].mean()) for j, v in enumerate(VIEWS)}, comparisons=comparisons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--code', type=Path, required=True)
    ap.add_argument('--grid', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--n', type=int, default=8192)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    sys.path.insert(0, str(args.code/'exploration/learning-histories'))
    import sessions as S
    from train import Learner
    K, q = 6, 7
    checkpoint = args.grid/f'attempts-s{args.seed}'/'attempts.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['step'] == 20000
    assert all(saved['args'][key] == value for key, value in dict(cond='attempts', K=K, seed=args.seed, batch=256, steps=20000).items())
    model = Learner(K+2).cuda().eval()
    model.load_state_dict(saved['model'])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out/'result.json').exists()
    result = dict(seed=args.seed, n=args.n, K=K, checkpoint=str(checkpoint), checkpoint_sha256=digest, source_step=saved['step'], views=VIEWS, replay={}, panels={})

    @torch.no_grad()
    def infer(s, c, o, k, query, view='real'):
        pred, nll = [], []
        for lo in range(0, len(s.rule), args.batch):
            sl = slice(lo, lo+args.batch)
            chunk = S.Sessions(s.rule[sl], s.scenes[sl], s.tgt[sl], s.true[sl])
            cond = 'demo' if view == 'demo_only' else 'attempts'
            x, pad = S.encode(chunk, cond, c[sl], o[sl], k, query)
            if view == 'masked':
                x[..., S.A+2:S.A+4] = 0
                xb, pb = S.encode(chunk, 'blind', c[sl], o[sl], k, query)
                assert np.array_equal(x, xb) and np.array_equal(pad, pb)
            assert not x[:, query, :, S.A:S.A+4].any()
            logits = model(torch.from_numpy(x).cuda(), torch.from_numpy(pad).cuda(), query)
            pred.append(logits.argmax(1).cpu().numpy())
            nll.append(torch.nn.functional.cross_entropy(logits, torch.from_numpy(chunk.true[:, query]).cuda(), reduction='none').cpu().numpy())
        return np.concatenate(pred), np.concatenate(nll)

    def own(s):
        c = np.zeros((len(s.rule), K+2), int)
        o = np.zeros_like(c, bool)
        for t in range(1, K+1):
            c[:, t], _ = infer(s, c, o, t-1, t)
            o[:, t] = c[:, t] == s.true[:, t]
        return c, o

    # Exact original evaluation reproduction before using any fresh panel.
    previous = json.loads((args.grid/f'attempts-s{args.seed}'/'attempts.json').read_text())
    for split in ('train', 'heldout'):
        s = S.sample(np.random.default_rng(10000+args.seed), 8192, K, split)
        c, o = own(s)
        accuracy = [float((infer(s,c,o,k,q)[0] == s.true[:,q]).mean()) for k in range(K+1)]
        assert np.array_equal(accuracy, previous[split]['learner']), (split, accuracy, previous[split]['learner'])
        result['replay'][split] = dict(status='exact', accuracy=accuracy)
        print('REPLAY', split, 'exact', flush=True)

    for split in ('train', 'heldout'):
        s = S.sample(np.random.default_rng(30000+args.seed), args.n, K, split)
        groups = dict(all=np.ones(args.n, bool), two_attribute=S.ARITY[s.rule] == 2)
        allowed = S.TRAIN if split == 'train' else S.HELDOUT
        excluded = np.setdiff1d(np.arange(S.R), allowed)
        for source in ('own', 'training_actor', 'random_actor'):
            if source == 'own':
                c, o = own(s)
            elif source == 'training_actor':
                c, o = S.behave(np.random.default_rng(40000+args.seed), s, K)
            else:
                c = np.zeros_like(s.true)
                c[:, 1:K+1] = np.random.default_rng(50000+args.seed).integers(0, S.N, (args.n,K))
                o = c == s.true
                o[:, [0,q]] = False
            # Shuffle outcomes across sessions separately at each attempt index.
            shuffled = o.copy()
            rng = np.random.default_rng(60000+args.seed)
            for t in range(1,K+1):
                shuffled[:, t] = o[rng.permutation(args.n), t]
                assert shuffled[:, t].sum() == o[:, t].sum()
            panel, predictions, references = [], [], []
            for k in range(K+1):
                ps, losses = [], []
                for view in VIEWS:
                    pred, loss = infer(s,c,shuffled if view == 'shuffled' else o,k,q,view)
                    ps.append(pred); losses.append(loss)
                ps, losses = np.stack(ps,1), np.stack(losses,1)
                if k == 0:
                    assert (ps == ps[:, :1]).all()
                    assert np.allclose(losses, losses[:, :1], atol=1e-6, rtol=0)
                lw = S.history_logw(s, 'attempts', c, o, k)
                full = S.predict(s, lw, q)[0].argmax(1)
                lw[:, excluded] = -np.inf
                matched = S.predict(s, lw, q)[0].argmax(1)
                record = dict(k=k, groups={})
                for name, select in groups.items():
                    m = metrics(ps, s.true[:,q], select)
                    m['nll'] = {v: float(losses[select,j].mean()) for j,v in enumerate(VIEWS)}
                    m['reference_full_prior'] = float((full[select] == s.true[select,q]).mean())
                    m['reference_split_prior'] = float((matched[select] == s.true[select,q]).mean())
                    record['groups'][name] = m
                panel.append(record); predictions.append(ps); references.append(np.stack([full,matched],1))
            key = split+'/'+source
            result['panels'][key] = panel
            np.savez_compressed(args.out/f'{split}-{source}.npz', rule=s.rule, scenes=s.scenes,
                truth=s.true, choice=c, outcome=o, shuffled_outcome=shuffled,
                prediction=np.stack(predictions,1), reference=np.stack(references,1))
            (args.out/'progress.json').write_text(json.dumps(dict(completed=list(result['panels']), elapsed_seconds=time.time()-started),indent=2)+'\n')
            print(key, json.dumps(panel[-1]['groups']['all']), flush=True)
    result['elapsed_seconds'] = time.time()-started
    result['qualification'] = dict(original_evaluation='exact_both_splits', masked_encoding='equals_blind', k0_views='identical', shuffle='preserves_each_attempt_marginal', device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__)
    (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE', result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
