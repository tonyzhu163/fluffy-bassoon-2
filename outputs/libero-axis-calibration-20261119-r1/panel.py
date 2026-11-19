"""Build one axis-calibration example per BASE support/default reset pair."""

import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1" / "SPLITS.json"
PREFIX = 80
MIN_ABS_COMMAND = 1e-5


def _tasks(by_suite):
    return {(suite, task) for suite, ids in by_suite.items() for task in ids}


def _digest(array):
    # Same shape/dtype framing used for raw observations and proposals in collect.py.
    header = str((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _read_controls(folder, record, *, support):
    saved = json.loads((folder / "result.json").read_text())
    for field in ("case", "identity", "controls", "waits", "opening_sha256"):
        assert saved[field] == record[field], (folder, field)
    count = record["controls"]
    identity = record["identity"]
    assert 0 < count <= identity["horizon"], folder
    with np.load(folder / "public.npz", allow_pickle=False) as tape:
        issued, opening = tape["issued"], tape["opening"]
        assert issued.shape == (count, 7) and opening.shape == (10, 7), folder
        assert np.isfinite(issued).all() and np.isfinite(opening).all(), folder
        assert _digest(opening) == record["opening_sha256"], folder
        # Default execution issues five controls before requesting a new proposal.
        burst = min(5, count)
        assert np.array_equal(issued[:burst], opening[:burst]), folder
        waits = tape["waits"]
        assert waits.shape == (count,) and not waits.any() and record["waits"] == 0, folder
        initial = tape["initial_position"]
        assert initial.shape == (3,) and np.isfinite(initial).all(), folder
        assert _digest(initial) == identity["public"]["robot0_eef_pos"], folder
        if not support:
            return issued[0, :2].copy()

        positions = tape["positions"]
        assert positions.shape == (count, 3) and np.isfinite(positions).all(), folder
        # positions[t] is measured after issuing control t, including at t == 0.
        xy = np.vstack((initial[:2], positions[:PREFIX, :2]))
        return issued[:PREFIX, :2], np.diff(xy, axis=0)


def _probe_indices(commands):
    magnitude = np.abs(commands)
    indices = []
    for axis in (0, 1):
        eligible = np.flatnonzero(
            (magnitude[:, axis] > magnitude[:, 1 - axis])
            & (magnitude[:, axis] > MIN_ABS_COMMAND)
        )
        # Earliest control wins ties within an axis.
        index = int(eligible[np.argmax(magnitude[eligible, axis])]) if eligible.size else None
        indices.append(index)
    return indices


def load_panel(episodes: Path, split: str) -> dict:
    """Read the collector's episodes root for either fitting or development.

    Support tokens are ordered x probe, y probe. The query is the first issued
    (x, y) command on the paired reset. Audit indices are zero-based.
    """
    if split not in ("fitting", "development"):
        raise ValueError(f"Expected fitting or development, got {split!r}")
    manifest = json.loads(SPLITS.read_text())
    fitting = _tasks(manifest["development_scaling"]["task_subsets"]["8"])
    development = _tasks(manifest["development"])
    assert len(fitting) == len(development) == 8
    assert fitting <= _tasks(manifest["train"]) and fitting.isdisjoint(development)
    assert (fitting | development).isdisjoint(_tasks(manifest["test"]))
    tasks = fitting if split == "fitting" else development
    expected = {(suite, task, reset) for suite, task in tasks for reset in (0, 3)}

    design = json.loads((episodes / "design.json").read_text())
    assert Path(design["motor"]).name == "pi05-libero-openpi-20261027-r1"
    assert "default" in design["modes"] and "BASE" in design["timings"]
    directory = episodes / split
    rows = json.loads((directory / "progress.json").read_text())
    seen = set()
    keys, supports, queries = [], [], []
    probe_indices, excluded = {}, {}
    for row in rows:
        support = row["support"]
        case = support["case"]
        if case["variant"] != "BASE":
            continue
        suite, task, reset = case["suite"], case["task_id"], case["init_state_id"]
        pair = (suite, task, reset)
        assert pair in expected and pair not in seen, pair
        seen.add(pair)
        key = f"{suite}-{task}-r{reset // 3}-BASE"
        assert row["trial"] == key, key
        query = row["queries"]["default"]
        assert query["case"] == dict(case, init_state_id=reset + 1, seed=case["seed"] + 1), key
        older, current = support["identity"], query["identity"]
        assert older["physics"] != current["physics"], key
        assert older["prompt"] == current["prompt"] and older["horizon"] == current["horizon"], key

        folder = directory / key
        commands, displacements = _read_controls(folder / "support", support, support=True)
        indices = _probe_indices(commands)
        probe_indices[key] = indices
        missing = [f"missing_{axis}_probe" for axis, i in zip("xy", indices) if i is None]
        if not missing:
            assert indices[0] != indices[1], key
            tokens = np.column_stack((commands[indices], displacements[indices]))

        opening = _read_controls(folder / "default", query, support=False)
        for axis, value in zip("xy", opening):
            if abs(value) <= MIN_ABS_COMMAND:
                missing.append(f"near_zero_query_{axis}")
        if missing:
            excluded[key] = missing
            continue
        keys.append(key)
        supports.append(tokens)
        queries.append(opening)

    assert seen == expected, (expected - seen, seen - expected)
    n = len(keys)
    assert n == len(set(keys)) and n + len(excluded) == len(expected) == 16
    support_array = np.asarray(supports, dtype=np.float32).reshape(n, 2, 4)
    query_array = np.asarray(queries, dtype=np.float32).reshape(n, 2)
    assert np.isfinite(support_array).all() and np.isfinite(query_array).all()
    return dict(
        support=support_array, query=query_array, keys=keys,
        audit=dict(split=split, motor=design["motor"], expected_trials=len(expected),
                   retained_trials=n, prefix_controls=PREFIX, min_abs_command=MIN_ABS_COMMAND,
                   probe_indices=probe_indices, excluded=excluded),
    )
