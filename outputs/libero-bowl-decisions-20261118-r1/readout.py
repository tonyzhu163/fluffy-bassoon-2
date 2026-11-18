"""Summarize completed branches and estimate timing from a fixed public prefix."""
import argparse
import json
from pathlib import Path

import numpy as np


def estimate_lag(data):
    # Use the same first 80 controls irrespective of contact, success or episode length.
    # For lag L correlate an issued translation with the later observed displacement.
    command = data['issued'][:80, :3].astype(float)
    position = data['state_after'][:80, :3].astype(float)
    movement = np.diff(position, axis=0)
    scores = []
    for lag in range(21):
        # Compare the same observed steps for every tested lag.
        x = command[21-lag:80-lag]
        y = movement[20:79]
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        scores.append(float((x*y).sum()/denom) if denom else 0.)
    return int(np.argmax(scores))


def summarize(directory):
    branches = []
    expected = json.loads((directory / 'complete.json').read_text())['episodes']
    for path in sorted(directory.glob('s*/result.json')):
        record = json.loads(path.read_text())
        with np.load(path.parent / 'public.npz') as data:
            lag = estimate_lag(data)
        hits = np.load(path.parent / 'evaluator-predicates.npy')
        assert hits.shape == (record['controls'], 2) and record['controls'] == 300
        first = [int(np.flatnonzero(h)[0])+1 if h.any() else None for h in hits.T]
        assert first == record['first'] and bool(hits[:, record['goal']].any()) == record['intended']
        branches.append(dict(scene=record['scene'], variant=record['variant'], goal=record['goal'],
                             gain=record['gain'], first=first, final=hits[-1].tolist(),
                             intended=record['intended'], public_lag=lag))
    assert len(branches) == expected
    return branches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.episodes),indent=2))
