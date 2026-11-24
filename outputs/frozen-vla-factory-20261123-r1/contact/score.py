"""Leave each entire task out of scaling, action-value estimation and fixed-program selection."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FEATURES = {"command_only": ("command",), "contact_gate": ("command", "contact")}
SEED = 41


def vector(features, groups):
    return np.concatenate([features[k] for k in groups])


def select(source, query_features, groups):
    if query_features is None:
        return 0
    active = [c for c in source if c["features"] is not None]
    x = np.array([vector(c["features"], groups) for c in active])
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    distances = np.linalg.norm((x - vector(query_features, groups)) / scale, axis=1)
    source_tasks = np.array([c["task"] for c in active])
    outcomes = np.array([c["native_hits"] for c in active])
    nearest = []
    for task in np.unique(source_tasks):
        members = source_tasks == task
        distance = distances[members].min()
        # Equal command sequences across materials remain tied for the command-only gate.
        values = outcomes[members & (distances == distance)].mean(axis=0)
        nearest.append((distance, values))
    nearest.sort(key=lambda p: p[0])
    limit = nearest[2][0]
    values = np.mean([v for d, v in nearest if d <= limit], axis=0)
    return int(np.argmax(values))


def predict(cells, tasks):
    picks = [None] * len(cells)
    for task in tasks:
        source = [c for c in cells if c["task"] != task]
        assert len(source) == 28 and len({c["task"] for c in source}) == 7
        fixed = int(np.argmax(np.sum([c["native_hits"] for c in source], axis=0)))
        for i, c in enumerate(cells):
            if c["task"] == task:
                picks[i] = dict(continue_tape=0, fixed_slow=1, fixed_regrasp=2, fold_fixed=fixed)
                for name, groups in FEATURES.items():
                    picks[i][name] = select(source, c["features"], groups)
    return picks


def score(panel):
    cells = panel["cells"]
    tasks = np.array(sorted({c["task"] for c in cells}))
    ids = np.array([c["task"] for c in cells])
    assert len(tasks) == 8 and len(cells) == 32
    assert all(len({(c["cadence"], c["material"]) for c in cells if c["task"] == t}) == 4 for t in tasks)
    assert all(len(set(c["native_hits"])) == 1 for c in cells if c["cut"] is None)
    picks = predict(cells, tasks)
    realized = {k: np.array([c["native_hits"][p[k]] for c, p in zip(cells, picks)])
                for k in picks[0]}
    realized["oracle"] = np.max([c["native_hits"] for c in cells], axis=1)

    rng = np.random.default_rng(SEED)
    bootstrap = rng.integers(len(tasks), size=(50_000, len(tasks)))

    def compare(a, b):
        diff = realized[a] - realized[b]
        by_task = np.array([diff[ids == t].mean() for t in tasks])
        interval = np.quantile(by_task[bootstrap].mean(axis=1), [.025, .975])
        return dict(wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                    delta_pp=100 * diff.mean(), task_bootstrap_95_pp=(100 * interval).tolist())

    strata = {"all": np.ones(len(cells), dtype=bool)}
    for key in ("cadence", "material", "task"):
        for value in sorted({c[key] for c in cells}):
            strata[value] = np.array([c[key] == value for c in cells])
    scores = {g: dict(n=int(mask.sum()), **{k: int(y[mask].sum()) for k, y in realized.items()})
              for g, mask in strata.items()}
    differences = {b: compare("contact_gate", b) for b in realized if b != "contact_gate"}
    program_names = panel["receipt"]["branch_order"]
    interventions = {k: {p: sum(c["cut"] is not None and d[k] == j for c, d in zip(cells, picks))
                         for j, p in enumerate(program_names)} for k in FEATURES}
    active = [c for c in cells if c["cut"] is not None]
    archive_agree = sum(c["archive_success"] == int(c["native_hits"][0] and
                        c["stops"][0] <= c["tape_length"]) for c in cells)
    checks = dict(matched_prefix_cells=len(cells), triggered=len(active),
                  archive_first_finger_at_or_before_cut=sum(c["archive_first_finger"] is not None and
                      c["archive_first_finger"] <= c["cut"] for c in active),
                  archive_reset_hash_matches=sum(c["archive_reset_match"] for c in cells),
                  archive_continue_hits_within_tape_match=archive_agree,
                  archive_stop_controls_match=sum(c["archive_controls"] ==
                      min(c["stops"][0], c["tape_length"]) for c in cells),
                  archive_max_abs_public_trace_error={k: max(c["archive_max_abs"][k] for c in cells)
                                                     for k in cells[0]["archive_max_abs"]})
    return dict(checks=checks, scores=scores, gate_minus_comparator=differences,
                oracle_minus_continue=compare("oracle", "continue_tape"),
                choices_on_triggered_cells=interventions,
                bootstrap=dict(unit="task", draws=50_000, seed=SEED),
                rows=[dict(task=c["task"], cadence=c["cadence"], material=c["material"],
                           gate=p["contact_gate"], command_only=p["command_only"], fold_fixed=p["fold_fixed"])
                      for c, p in zip(cells, picks)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=HERE / "panel.json")
    parser.add_argument("--out", type=Path, default=HERE / "metrics.json")
    args = parser.parse_args()
    metrics = score(json.loads(args.panel.read_text()))
    args.out.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics["scores"]["all"]))
