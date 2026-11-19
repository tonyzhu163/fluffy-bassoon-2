"""Recount the native-source panel and test for a marginal direction shortcut."""
import json
from pathlib import Path
import sys

import numpy as np

from factor import SIGNS, support_grid


def signatures(panel):
    """Multiset of 2-probe observations over the four reflected directions."""
    return sorted(row.tobytes() for row in np.where(panel == 0, 0, panel))


def main():
    out = Path(sys.argv[1])
    counts = json.loads((out / "summary.json").read_text())["counts"]
    checked = {}
    with np.load(out / "predictions.npz", allow_pickle=False) as data:
        rows = 0
        for model, arms in counts.items():
            for arm, labels in arms.items():
                pred = data[f"{model}/{arm}/command_prediction"]
                correct = np.sign(pred) == data["command_target_sign"]
                for i, name in enumerate(data["convention"]):
                    c = correct[:, i].reshape(-1, 2)
                    saved = labels[str(name)]
                    assert [saved[k] for k in ("n", "joint", "x", "y")] == [len(c), c.all(1).sum(), *c.sum(0)]
                    rows += len(c)
        for split in ("fitting", "development"):
            conventions = SIGNS[:3] if split == "fitting" else SIGNS
            grid = support_grid(data[f"{split}/nominal_support"], conventions)
            cases = 0
            for trial in range(grid.shape[2]):
                for columns in (slice(0, 2), slice(2, 4)):
                    baseline = signatures(grid[:, 0, trial, :, columns])
                    for i in range(1, len(conventions)):
                        assert signatures(grid[:, i, trial, :, columns]) == baseline
                    cases += 1
            checked[split] = dict(source_pairs=grid.shape[2], feature_marginals=cases,
                                  conventions=len(conventions), all_equal=True)
    print(json.dumps(dict(verified_predictions=rows, single_input_distributions=checked), indent=1))


if __name__ == "__main__":
    main()
