"""Pinned inputs and reference transport shared by preparation, fitting and evaluation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REF_CODE = HERE.parent / "reground-rotation-reference-20261123-r1"
sys.path.insert(0, str(REF_CODE))
import reference as ref

SOURCE_STORE = ref.ROOT / "reference-native"
WORK = ref.ROOT.parent / "reground-low-budget-20261123-r1"
CELL = "rot_z/90"
BUDGETS = (15, 100, 1000)
REF_FITS = ref.read(REF_CODE / "evidence/fitted-models.json")


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def subset_hash(episodes):
    """Frame episode boundaries by uint64-le length, then X and A little-endian f32 C bytes."""
    h = hashlib.sha256()
    for e in episodes:
        n = len(e["A"])
        ref.require(e["X"].shape == (n + 1, 25) and e["A"].shape == (n, 7) and n > 0, "Invalid tape")
        h.update(n.to_bytes(8, "little"))
        for k in ("X", "A"):
            ref.require(e[k].dtype.name == "float32", "Unexpected source dtype")
            h.update(e[k].astype("<f4", copy=False).tobytes(order="C"))
    return h.hexdigest()


def configure():
    ref.configure()
    ref.require(ref.ROOT.is_absolute() and ref.ROOT.is_dir(), "Source the shared runtime")
    # All default native write paths are private. Source data are opened explicitly during preparation.
    from reground import utils
    utils.RUNS = WORK / "private-runs"


def cell_dir(seed, budget):
    return WORK / "inputs" / f"seed{seed}" / f"b{budget}"


def reference_model(seed, role, kind):
    return next(m for m in REF_FITS[seed]["models"] if (m["role"], m["kind"]) == (role, kind))


def initial_state(seed, kind):
    m = reference_model(seed, "dep", kind)
    return SOURCE_STORE / "fit-starts" / (Path(m["dir"]).name + ".pt")


def recipe(seed, budget):
    cfg = ref.read(ref.SOURCE / "configs/paper.json")[CELL]
    j = ref.grid()[seed]
    ref.require((j["cfg"], j["cell"], j["policy"], j["seed"]) == (cfg, CELL, "pi05", seed)
                and cfg["anchor"] == "idm" and not cfg.get("gate"), "Unexpected release recipe")
    return {**cfg, "budget": budget}
