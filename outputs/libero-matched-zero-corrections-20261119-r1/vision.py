"""Predict completion from public tapes with a fitted nearest-neighbor kernel."""
import json
from pathlib import Path

import numpy as np

GOALS = (4, 8)
MODES = {'reset': ('reset',), 'commands': ('reset','commands'),
         'motion': ('reset','commands','states'), 'full': ('reset','commands','states','images')}


def public_features(tape):
    frames = tape['frames'].astype(np.float32)/255.
    pooled = frames.reshape(31, 2, 7, 4, 7, 4, 3).mean(axis=(3,5))
    states = tape['states']
    positions = states[9::10, :3] - states[0, :3]
    widths = np.abs(np.diff(states[9::10,-2:], axis=1))
    return dict(reset=pooled[0], images=(pooled[1:] - pooled[0]),
                commands=tape['actions'].reshape(30,10,7).mean(axis=1),
                states=np.column_stack([positions, widths]).astype(np.float32))


def read_split(root):
    practice, queries = {}, {}
    for block in sorted(root.glob('p*')):
        pose = int(block.name[1:])
        start = json.loads((block/'initial.json').read_text())
        assert start['physics'] and len(start['prompts']) == 2
        initial_view = None
        for path in sorted(block.glob('g*')):
            info = json.loads((path/'result.json').read_text())
            assert info['pose'] == pose
            slot, role = GOALS.index(info['goal']), info['role']
            with np.load(path/'tape.npz') as tape:
                image = tape['frames'][0]
                if initial_view is None:
                    initial_view = image
                np.testing.assert_array_equal(image, initial_view)
                native = tape['native']
                assert native.shape == (300,2)
                if role.startswith('practice'):
                    i = int(role.removeprefix('practice'))
                    practice[pose,slot,i] = dict(native=native.any(axis=0), goal=slot, pose=pose,
                                                 features=public_features(tape))
                else:
                    queries[pose,slot,info['cadence']] = dict(native=native.any(axis=0), final=native[-1],
                                                               first=tape['actions'][:5], seed=info['seed'])
        for slot in (0,1):
            slow, fast = (queries[pose,slot,c] for c in (10,5))
            assert slow['seed'] == fast['seed']
            np.testing.assert_array_equal(slow['first'],fast['first'])
    assert len(practice)==32 and len(queries)==16
    return practice, queries


def squared_dist(a, b, feature):
    x = a['features'][feature]
    y = b['features'][feature]
    return float(np.mean(np.square(x-y)))


def fit(source):
    result = {}
    for slot in (0,1):
        items = [t for t in source.values() if t['goal']==slot]
        labels = np.array([t['native'][slot] for t in items], dtype=float)
        poses = np.array([t['pose'] for t in items])
        distances = {}
        scales = {}
        for key in MODES['full']:
            d = np.array([[squared_dist(a,b,key) for b in items] for a in items])
            # The scale uses distances across fitting poses, where validation occurs.
            vals = d[(poses[:,None]!=poses[None,:]) & (d>0)]
            scales[key] = np.median(vals) if len(vals) else 1.
            distances[key] = d/scales[key]
        for mode, keys in MODES.items():
            ds = np.mean([distances[k] for k in keys], axis=0)
            losses = []
            for bandwidth in (.25, 1., 4.):
                predicted = [kernel(ds[i,poses!=p],labels[poses!=p],bandwidth) for i,p in enumerate(poses)]
                losses.append(float(np.mean((labels-predicted)**2)))
            bandwidth = (.25, 1., 4.)[int(np.argmin(losses))]
            result[slot,mode] = (items, labels, {k:scales[k] for k in keys}, bandwidth, losses)
    return result


def kernel(distances, labels, bandwidth):
    w = np.exp(-(distances-distances.min())/bandwidth)
    return float((np.dot(w, labels)+.5)/(w.sum()+1))


def predict(model, tape, mode):
    items, labels, scales, bandwidth, _ = model[tape['goal'],mode]
    distances = np.array([sum(squared_dist(tape,b,k)/s for k,s in scales.items())/len(scales) for b in items])
    return kernel(distances,labels,bandwidth)
