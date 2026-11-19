"""Source-only fitting and paired evaluation of role-relative history readers."""
from itertools import combinations, permutations
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'libero-scene-probes-20261119-r1'))
from report import KNOWN, load_panel, next_scene
from selector import Reachability, distance

SEEDS = (11, 29, 47)


def source_scores(source, panel, model):
    files = sorted((source / 'fitting').glob('p*/*/g*/result.json'))
    rows = [json.loads(f.read_text()) for f in files]
    poses = np.array([r['scene'] for r in rows])
    for i, row in enumerate(rows):
        goal = row['goal']
        assert goal == model.goals[i]
        pool = np.flatnonzero((poses != poses[i]) & (model.goals == goal))
        d = (distance(model.images[pool], model.images[i]) / model.scale[0]
             + distance(model.plans[pool], model.plans[i]) / model.scale[1])
        near = pool[np.argsort(d, kind='stable')[:3]]
        panel[row['scene'], row['layout']]['meta']['scores'][str(goal)] = model.targets[near].mean(axis=0).tolist()


def pose_orders(panel, all_poses):
    for pose, layout in sorted(panel):
        if all_poses:
            other = sorted(p for p, l in panel if l == layout and p != pose)
            for second, query in permutations(other, 2):
                yield pose, second, query, layout
        else:
            second = next_scene(panel, pose)
            yield pose, second, next_scene(panel, second), layout


def tapes(panel, triples, *, all_poses=False):
    result = []
    for pose, p2, p3, layout in pose_orders(panel, all_poses):
        assert len({pose, p2, p3}) == 3
        first, second, query = (panel[p, layout] for p in (pose, p2, p3))
        for goals in triples:
            for order in permutations(goals):
                reach = [block['meta']['scores'][str(g)][0] for block, g in zip((first, second), order[:2])]
                reach += [query['meta']['scores'][str(g)][0] for g in order]
                for hidden, target in enumerate(order):
                    def observed(t):
                        return [int(block['records'][g][str(t)] is not None)
                                for block, g in zip((first, second), order[:2])]
                    result.append((pose, layout, target, hidden, reach, observed(target),
                                   observed(order[(hidden+1) % 3]),
                                   [int(query['records'][g][str(target)] is not None) for g in order]))
    names = ('pose', 'layout', 'target', 'hidden', 'reach', 'reward', 'donor', 'native')
    return {k: np.array(v) for k, v in zip(names, zip(*result))}


def inputs(data, feedback):
    n = len(data['pose'])
    reach = data['reach']
    columns = []
    for slot in range(3):
        cols = [np.full(n, slot == i) for i in range(2)]
        cols += [feedback[:, 0], feedback[:, 1], reach[:, 0], reach[:, 1], reach[:, 2 + slot]]
        columns.append(np.stack(cols, axis=1))
    return torch.tensor(np.stack(columns, axis=1), dtype=torch.float32)


def network():
    return nn.Sequential(nn.Linear(7, 24), nn.Tanh(), nn.Linear(24, 1), nn.Flatten(1))


def fit(source, out):
    panel = load_panel(source / 'fitting')
    source_scores(source, panel, Reachability(source / 'frozen.npz'))
    data = tapes(panel, list(combinations(KNOWN, 3)))
    reward, native = data['reward'], data['native']
    features = torch.cat([inputs(data, reward), inputs(data, np.full_like(reward, -1))])
    targets = torch.tensor(np.tile(native, (2, 1)), dtype=torch.float32)
    table = {}
    for key in map(tuple, np.unique(reward, axis=0)):
        idx = np.all(reward == key, axis=1)
        table[''.join(map(str, key))] = native[idx].mean(axis=0).tolist()
    off = [int(t is not None) for block in panel.values() for g, rec in block['records'].items()
           for h, t in rec.items() if int(h) != g]
    config = dict(table=table, marginal=native.mean(axis=0).tolist(), off=(sum(off)+0.5)/(len(off)+1),
                  source_rows=len(reward), off_hits=sum(off), off_trials=len(off))
    out.mkdir()
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = network()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(240):
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(features), targets)
            loss.backward()
            opt.step()
        torch.save(model.state_dict(), out / f'seed{seed}.pt')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')


def distributions(values):
    ties = np.isclose(values, values.max(axis=1, keepdims=True), atol=1e-9, rtol=0)
    return ties / ties.sum(axis=1, keepdims=True)


def evaluate(panel, triples, frozen, *, all_poses=False):
    data = tapes(panel, triples, all_poses=all_poses)
    native, bits = data['native'], data['reward']
    config = json.loads((frozen / 'config.json').read_text())
    predictions = {}
    with torch.no_grad():
        for condition, rewards in [('matched', bits), ('cleared', np.full_like(bits, -1)), ('donor', data['donor'])]:
            scores = []
            for seed in SEEDS:
                model = network()
                model.load_state_dict(torch.load(frozen / f'seed{seed}.pt', map_location='cpu', weights_only=True))
                value = torch.sigmoid(model(inputs(data, rewards))).numpy()
                if condition == 'matched':
                    predictions[f'seed{seed}'] = distributions(value)
                scores.append(value)
            predictions[condition] = distributions(np.mean(scores, axis=0))
    table = config['table']
    predictions['table'] = distributions(np.array([table.get(''.join(map(str, b)), config['marginal']) for b in bits]))
    predictions['uniform'] = np.full_like(native, 1/3, dtype=float)
    reach = (data['reach'] * 3 + 0.5) / 4
    posterior = np.ones_like(native, dtype=float)
    for i in range(2):
        hit = np.full_like(posterior, config['off'])
        hit[:, i] = reach[:, i]
        posterior *= np.where(bits[:, i, None], hit, 1-hit)
    predictions['bayes'] = distributions(posterior * reach[:, 2:] + (posterior.sum(axis=1, keepdims=True)-posterior)*config['off'])
    matched = np.eye(3)[data['hidden']]
    def totals(mask):
        return dict(rows=int(mask.sum()), oracle=int(native[mask].max(axis=1).sum()),
                    named_motor=int((native[mask]*matched[mask]).sum()),
                    native={k:float((p[mask]*native[mask]).sum()) for k,p in predictions.items()},
                    correct={k:float((p[mask]*matched[mask]).sum()) for k,p in predictions.items()})
    return dict(total=totals(np.ones(len(bits), dtype=bool)),
                pose={str(p):totals(data['pose']==p) for p in np.unique(data['pose'])},
                goal={str(g):totals(data['target']==g) for g in np.unique(data['target'])},
                reward={str(b):totals(np.all(bits==b,axis=1)) for b in np.unique(bits,axis=0)})


def main(command, source, out):
    torch.set_num_threads(1)
    if command == 'fit':
        fit(source, out / 'reader')
        return
    model = Reachability(source / 'frozen.npz')
    if command == 'primary':
        panel = load_panel(out / 'episodes', model)
        triples = [(0, 1, 5)]
    elif command == 'mixed':
        panel = load_panel(source / 'fresh', model)
        triples = [(0, 1, g) for g in KNOWN]
    else:
        raise ValueError(command)
    report = evaluate(panel, triples, out / 'reader')
    (out / f'{command}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(command, report['total'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1], *map(Path, sys.argv[2:]))
