"""Audit completed physical branches and summarize development retry outcomes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero-pro2-demonstration-pairs-20261118-r1"))
from read_bowls import lag_of_prefix


def read(path):
    return json.loads(path.read_text())


def archive(directory, record):
    assert read(directory / "result.json") == record
    with np.load(directory / "public.npz") as tape:
        data = {key: tape[key] for key in tape.files}
    actions = data["issued"]
    waits = data["waits"]
    opening = data["opening"]
    count = record["controls"]
    assert 0 < count <= record["identity"]["horizon"]
    assert actions.shape == (count, 7) and data["positions"].shape == (count, 3)
    assert np.isfinite(actions).all() and np.isfinite(data["positions"]).all()
    assert data["rewards"].shape == waits.shape == (count,)
    assert np.isfinite(data["rewards"]).all()
    assert np.any(data["rewards"] > 0) == record["success"]
    move, hold = (5, 0) if record["mode"] == "default" else (10, 14)
    assert np.array_equal(data["decision_offsets"], np.arange(0, count, move + hold))
    assert np.array_equal(waits, np.arange(count) % (move + hold) >= move)
    assert waits.sum() == record["waits"] and not actions[waits, :6].any()
    preceding_move = np.flatnonzero(waits) // (move + hold) * (move + hold) + move - 1
    assert np.array_equal(actions[waits, -1], actions[preceding_move, -1])
    assert opening.shape == (10, 7) and np.array_equal(actions[:move], opening[:min(move, count)])
    digest = hashlib.sha256(str((opening.shape, opening.dtype.str)).encode() + opening.tobytes()).hexdigest()
    assert digest == record["opening_sha256"]
    return data


def tabulate(rows):
    counts = Counter()
    for row in rows:
        counts["trials"] += 1
        for key in ("support_success", "default", "paced", "retained", "known_timing", "any_mode",
                    "both_modes", "neither_mode", "retained_only_vs_paced", "paced_only_vs_retained"):
            counts[key] += row[key]
        counts["abstained"] += row["lag"] is None
        counts["timing_decoded"] += (row["lag"] is not None and row["selected"] == row["timing_mode"])
    return dict(counts)


def summarize(output):
    design = read(output / "design.json")
    rows = read(output / "progress.json")
    expected = {(suite, task, repeat, variant)
                for suite, tasks in design["splits"].items() for task in tasks
                for repeat in range(2) for variant in ("BASE", "S20")}
    assert read(output / "complete.json") == {"trials": len(expected), "episodes": 3 * len(expected)}
    assert len(rows) == len(expected)
    seen = set()
    detail = []
    by_task, by_timing = defaultdict(list), defaultdict(list)
    prompt_by_task = {}
    for row in rows:
        directory = output / row["scenario"]
        support = row["support"]
        suite, task, variant = (support["case"][field] for field in ("suite", "task_id", "variant"))
        repeat = support["case"]["init_state_id"] // 3
        key = suite, task, repeat, variant
        assert key in expected and key not in seen
        seen.add(key)
        query = row["query"]
        assert set(query) == {"default", "paced"}
        default, paced = query["default"], query["paced"]
        prompt = support["identity"]["prompt"]
        assert prompt_by_task.setdefault((suite, task), prompt) == prompt
        assert support["case"]["init_state_id"] == repeat * 3
        assert default["case"] == paced["case"] == dict(
            support["case"], init_state_id=3 * repeat + 1, seed=support["case"]["seed"] + 1)
        assert default["identity"] == paced["identity"] and default["identity"]["prompt"] == prompt
        earlier = archive(directory / "support", support)
        lag = lag_of_prefix(earlier["issued"], earlier["positions"])
        anchors = design["lag_anchors"]
        selected = "default" if lag is not None and abs(lag - anchors["BASE"]) < abs(lag - anchors["S20"]) else "paced"
        decision_path = directory / "decision.json"
        assert read(decision_path) == {"lag": lag, "selected": selected}
        assert decision_path.stat().st_mtime_ns <= (directory / selected / "result.json").stat().st_mtime_ns
        assert row["selected"] == selected and row["lag"] == lag
        tapes = {mode: archive(directory / mode, record) for mode, record in query.items()}
        assert np.array_equal(tapes["default"]["opening"], tapes["paced"]["opening"])
        timing = "default" if variant == "BASE" else "paced"
        d, p = (q["success"] for q in (default, paced))
        retained = query[selected]["success"]
        trial = dict(suite=suite, task_id=task, repeat=repeat, variant=variant, lag=lag,
                     selected=selected, timing_mode=timing, support_success=support["success"],
                     default=d, paced=p, retained=retained, known_timing=query[timing]["success"],
                     any_mode=d or p, both_modes=d and p, neither_mode=not (d or p),
                     retained_only_vs_paced=retained and not p, paced_only_vs_retained=p and not retained,
                     default_controls=default["controls"], paced_controls=paced["controls"])
        detail.append(trial)
        by_task[f"{suite}/{task}"].append(trial)
        by_timing[variant].append(trial)
    return dict(overall=tabulate(detail), timing={k: tabulate(v) for k, v in by_timing.items()},
                task={k: tabulate(v) for k, v in by_task.items()}, trials=detail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    print(json.dumps(summarize(parser.parse_args().episodes), indent=2))
