"""Read the two archived menus, aligning the original common query prefixes."""
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'libero-first-attempt-20261119-r1'))
from design import lag24, native_record

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
BOWL = ROOT / 'libero-bowl-decisions-20261118-r1'
TRANSFER = ROOT / 'libero-first-attempt-20261119-r1/transfer02/episodes'
TIMING = ('BASE', 'S20')
ACTIONS = list(product(range(2), (False, True)))


def tape_paths(family, index, timing):
    if family == 'bowl':
        old = BOWL / 'livepair01/episodes/paced_support' / f'b{index}-{timing}'
        supports = list(old.glob('goal*/s*-paced'))
        stem = f's{16+index}-{timing}-g'
        queries = [(BOWL/'pacing01/episodes'/f'{stem}{g}-paced', True) for g in range(2)]
        queries += [(TRANSFER/'bowl'/f'{stem}{g}-switch', False) for g in range(2)]
    else:
        block = TRANSFER/'bottle'/f's{index}-{timing}'
        supports = list(block.glob('query-g*'))
        queries = [(p, json.loads((p/'result.json').read_text())['paced']) for p in block.glob('support-*')]
    return sorted(supports), queries


def motion(actions, states):
    closed = actions[:, -1] > 0
    starts = np.flatnonzero(closed & ~np.r_[False, closed[:-1]])
    stops = np.flatnonzero(closed & ~np.r_[closed[1:], False]) + 1
    rise, width, travel = 0., 0., 0.
    for start, stop in zip(starts, stops):
        xyz = states[start:stop, :3].astype(float)
        lift = xyz[:, 2] - np.minimum.accumulate(xyz[:, 2])
        peak = int(np.argmax(lift))
        if lift[peak] > rise:
            rise = float(lift[peak])
            width = float(abs(np.diff(states[start+peak, -2:])[0]))
        travel = max(travel, float(np.linalg.norm(xyz[:, :2]-xyz[0, :2], axis=1).max()))
    return [rise, width, travel]


def read_panel(family):
    tapes = []
    for index, timing in product(range(4 if family == 'bowl' else 6), TIMING):
        supports, queries = tape_paths(family, index, timing)
        query = {}
        initial, prefix = set(), {}
        for path, paced in queries:
            rec = native_record(path)
            g = rec['goal']
            assert (g, paced) not in query
            query[g, paced] = [x is not None for x in rec['first']]
            initial.add((rec['scene'], rec['initial_physics'], json.dumps(rec['initial_public'], sort_keys=True)))
            with np.load(path/'public.npz') as data:
                first = [data[k][:24] for k in ('issued', 'state_after')]
            if g in prefix:
                for a, b in zip(first, prefix[g]):
                    np.testing.assert_array_equal(a, b)
            prefix[g] = first
        assert len(initial) == 1 and set(query) == set(ACTIONS)
        query_pose, query_physics, _ = initial.pop()
        support_reset, opts = set(), []
        for path in supports:
            rec = native_record(path)
            assert rec['variant'] == timing and rec['paced'] and rec['scene'] != query_pose
            assert rec['initial_physics'] != query_physics
            support_reset.add((rec['scene'], rec['initial_physics'], json.dumps(rec['initial_public'], sort_keys=True)))
            opts.append(rec['goal'])
            with np.load(path/'public.npz') as data:
                act, st = data['issued'], data['state_after']
            assert act.shape == (300, 7) and st.shape == (300, 9)
            assert np.isfinite(act).all() and np.isfinite(st).all()
            support_hit = [t is not None for t in rec['first']]
            tapes.append(dict(index=index, timing=timing, pose=rec['scene'], query_pose=query_pose,
                              target=rec['goal'], lag=lag24(act, st), motion=motion(act, st),
                              hit=support_hit, query=[query[a] for a in ACTIONS],
                              path=str(path.relative_to(ROOT))))
        assert len(support_reset) == 1 and sorted(opts) == [0, 1]
    return tapes
