"""Input identities and strict custody for the original mapreader factors."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
READER = HERE.parent / "symmeta-active-map-reader-20261124-r1"
sys.path.insert(0, str(READER))
from observations import SCALE, case_rows, correct, digest, interval, law, metrics, ridge

REGISTRATION = "41e4fda4d4eeef4572e4baef47f7ed4b52c7967d85d8e9cb13cc49041866cc65"
LAW = "acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73"
READER_PIN = "589873c103e46b3ced7bda4ca855b3401b171b2d"
SEEDS = (41, 42, 43)
SOURCE_ARMS = ("Panda", "Sawyer")
TARGET_ARMS = ("UR5e", "IIWA")
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
TARGET_TASKS = dict(zip(SUITES, (8, 6, 8, 8)))
MODES = ("passive", "orth+", "orth-", "along+", "along-")
VAL_MAPS = ("native", "family", "swap", "aniso", "shear", "general+", "general-", "general")
TARGET_MAPS = ("native", "family", "swap", "general+", "general-")
ROW_KEYS = ("arm", "task", "family", "mode")


def read(path):
    return json.loads(path.read_bytes())


def write(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def key(row):
    return tuple(row[k] for k in ROW_KEYS)


def token_hash(tok):
    return hashlib.sha256(np.ascontiguousarray(tok, dtype="<f4").tobytes()).hexdigest()


def copy_verified(src, dest, checksum):
    if digest(src) != checksum:
        raise ValueError(f"Input hash mismatch: {src}")
    shutil.copyfile(src, dest)
    assert digest(dest) == checksum


def historical():
    f = HERE.parent / "frozen-vla-motion-models-20261123-r1/evidence/source-fit.json"
    assert digest(f) == LAW and digest(HERE / "PLAN.md") == REGISTRATION
    fit = read(f)
    return fit


def validation_tasks(fit):
    ids = {}
    for suite in SUITES:
        tasks = sorted({int(r["task"].split("/")[1]) for r in fit["sources"]
                        if r["task"].split("/")[0] == suite})
        assert len(tasks) == 6
        ids[suite] = tasks[4]
    return ids


def original_rows(bank, split, fit):
    arms, tasks, families = ((SOURCE_ARMS, validation_tasks(fit), VAL_MAPS) if split == "val"
                             else (TARGET_ARMS, TARGET_TASKS, TARGET_MAPS))
    assert split in ("val", "target")
    paths, rows = {}, []
    for arm in arms:
        runtime = bank / f"runtime-{arm}.json"
        rt = read(runtime)
        assert rt["arm"] == arm and rt["source_hash"] == LAW
        assert rt["residual"] == fit["residual_power"] and rt["prior"] == fit["prior"]
        paths[str(runtime)] = digest(runtime)
        for suite, task in tasks.items():
            folder = bank / f"{arm}-{suite}-{task}"
            meta = read(folder / "case.json")
            assert (meta["arm"], meta["suite"], meta["task"], meta["split"]) == (arm, suite, task, split)
            assert set(meta["maps"]) == set(families) and meta["complete"]
            physical = case_rows(folder, np.asarray(rt["prior"]), rt["residual"])
            expected = {(arm, f"{suite}/{task}", f, m) for f in families for m in MODES}
            assert len(physical) == len(expected) and {key(r) for r in physical} == expected
            rows.extend(physical)
            for path in sorted(folder.iterdir()):
                if path.suffix in (".npz", ".json"):
                    paths[str(path)] = digest(path)
    return rows, paths


def source_identity(model, bank, require_seal=False):
    fit = historical()
    select = read(model / "selected.json")
    assert (model / "launch.txt").read_text().split()[0] == READER_PIN
    assert select["seeds"] == list(SEEDS) and select["feature_width"] == 100
    assert select["prior"] == fit["prior"] and select["residual"] == fit["residual_power"]
    ck = {f"seed-{s}.pkl": digest(model / f"seed-{s}.pkl") for s in SEEDS}
    assert ck == select["checkpoints"]
    if require_seal:
        seal = read(model / "sealed.json")
        assert seal["selected"] == digest(model / "selected.json")
        assert ck == seal["checkpoints"] and Path(seal["bank"]).resolve() == bank.resolve()
    return select


def archive_cases(path, rows):
    """Small standalone numeric archive sufficient to recompute each estimator and endpoint."""
    arrays = {}
    for i, row in enumerate(rows):
        for name in ("tok", "x", "y", "issued", "fk", "clock", "baseline", "a", "q", "hc", "hd"):
            arrays[f"{i}_{name}"] = row[name]
    np.savez_compressed(path, **arrays)
