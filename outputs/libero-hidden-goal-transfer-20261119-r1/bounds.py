"""Bound achievable native success when different hidden assignments give identical feedback."""
import json
from pathlib import Path
import sys

import numpy as np

from reader import distributions, load_panel, tapes


def bounds(panel, triples):
    data = tapes(panel, triples)
    # Each physical tape/order is scored for all three hidden assignments consecutively.
    hidden = data['hidden'].reshape(-1, 3)
    assert np.all(hidden == np.arange(3))
    native = data['native'].reshape(-1, 3, 3)
    rewards = data['reward'].reshape(-1, 3, 2)
    reach = data['reach'][:, 2:]
    by_reach = float((distributions(reach) * data['native']).sum())
    totals = np.zeros(3)
    completion_balanced = 0.
    completion_reach = 0.
    off_target_positive = 0
    for index, (outcomes, feedback) in enumerate(zip(native, rewards)):
        # The diagonal uses the evaluator's own-goal hit for each probe. This
        # privileged diagnostic asks what a perfect completion detector buys.
        finished = feedback[np.arange(2), np.arange(2)]
        off_target_positive += int(feedback.sum() - finished.sum())
        for h in range(3):
            evidence = feedback[h]
            if evidence.any():
                compatible = np.r_[evidence.astype(bool), False]
            else:
                compatible = np.r_[~finished.astype(bool), True]
            weights = compatible / compatible.sum()
            completion_balanced += float(weights @ outcomes[h])
            rank = np.where(compatible, reach[3*index+h], -1)
            completion_reach += float(distributions(rank[None, :])[0] @ outcomes[h])
        for k in range(3):
            _, groups = np.unique(feedback[:, :k], axis=0, return_inverse=True) if k else (None, np.zeros(3))
            for group in np.unique(groups):
                totals[k] += outcomes[groups == group].sum(axis=0).max()
    return dict(assignments=int(native.shape[0]*3), physical_orders=int(native.shape[0]),
                query_only_reachability=by_reach, oracle_by_available_reward_count=totals.tolist(),
                privileged_completion_uniform=completion_balanced, privileged_completion_reach=completion_reach,
                off_target_positive_support_bits=off_target_positive,
                hidden_known_oracle=int(native.max(axis=2).sum()))


if __name__ == '__main__':
    out = Path(sys.argv[1])
    panel, triples = (load_panel(Path(sys.argv[2])), [(0, 1, g) for g in (4, 8, 2, 9)]) if len(sys.argv) == 3 else (load_panel(out / 'episodes'), [(0, 1, 5)])
    result = bounds(panel, triples)
    (out / ('bounds-mixed.json' if len(sys.argv) == 3 else 'bounds.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(result)
