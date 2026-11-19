"""Bound achievable native success when different hidden assignments give identical feedback."""
import json
from pathlib import Path
import sys

import numpy as np

from reader import distributions, load_panel, tapes


def bounds(panel):
    data = tapes(panel, [(0, 1, 5)])
    # Each physical tape/order is scored for all three hidden assignments consecutively.
    hidden = data['hidden'].reshape(-1, 3)
    assert np.all(hidden == np.arange(3))
    native = data['native'].reshape(-1, 3, 3)
    rewards = data['reward'].reshape(-1, 3, 2)
    reach = data['reach'][:, 2:]
    by_reach = float((distributions(reach) * data['native']).sum())
    totals = np.zeros(3)
    for outcomes, feedback in zip(native, rewards):
        for k in range(3):
            _, groups = np.unique(feedback[:, :k], axis=0, return_inverse=True) if k else (None, np.zeros(3))
            for group in np.unique(groups):
                totals[k] += outcomes[groups == group].sum(axis=0).max()
    return dict(assignments=int(native.shape[0]*3), physical_orders=int(native.shape[0]),
                query_only_reachability=by_reach, oracle_by_available_reward_count=totals.tolist(),
                hidden_known_oracle=int(native.max(axis=2).sum()))


if __name__ == '__main__':
    out = Path(sys.argv[1])
    result = bounds(load_panel(out / 'episodes'))
    (out / 'bounds.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result)
