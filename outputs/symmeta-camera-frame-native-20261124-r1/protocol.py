"""Registered cases and the real PR7 motor, release bank and support identities."""
import csv
import os
from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve().parent
PR7 = HERE.parent / "symmeta-perceiver-retry-20261123-r1"
sys.path.insert(0, str(PR7))
from paired_records import atomic, checked, digest, encode, parse, require

ROOT = Path("/mowenjie/fluffy-bassoon-2")
PR7_PANEL = "72ea6883cc9eca662041e645c0f31a8e5482f231507b0a1559bdebf44beec6f0"
BANKS = HERE.parent / "frozen-transfer-progress-20261124-r1/evidence/release-banks.json"
BANK_SHA = "b0270a0d9ecb6ab8a467e702197f33427f0c654107eac5c7b5f9cf97d97dc799"
RELEASE_PIN = "e5500c690c1498eca1b7050350654964190cdf70"
MANIFEST_SHA = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
IDENTITY = dict(yaw_deg=0.0, xy_scale=1.0, gripper_sign=1)


def cases():
    with (HERE / "cases.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for k in ("index", "task_shard", "slot", "task_id", "support_layout", "support_seed",
                  "orbit_deg", "digital_roll_deg", "query_layout", "query_seed"):
            row[k] = int(row[k])
    require([r["index"] for r in rows] == [12*t+s for t in range(8) for s in range(4, 12)], "case coverage")
    for col in ("task_shard", "embodiment"):
        for k in {r[col] for r in rows}:
            group = [r for r in rows if r[col] == k]
            require(len(group) == 8 and sum(r["orbit_deg"] for r in group) == 0 and
                    sum(r["query_layout"] == 26 for r in group) == 4, "unbalanced registration")
    return rows


def release_rows():
    import torch
    release = Path(os.environ["CROSSARM_RELEASE"])
    manifest = parse(checked(release / "assets/task_manifest.json", MANIFEST_SHA))
    hasher = runpy.run_path(str(release / "libero_remote/reproducibility.py"))["numerical_state_fingerprint"]
    result = {}
    for b in parse(checked(BANKS, BANK_SHA))["tasks"]:
        key = b["suite"], b["task_id"]
        entry, = (x for x in manifest["tasks"] if (x["suite"], x["base_task_id"], x["case_id"]) == (*key, "BASE"))
        require(entry["max_steps"] == b["native_horizon"] and entry["bddl_sha256"] == b["bddl_sha256"], key)
        checked(release / "assets" / entry["bddl_path"], b["bddl_sha256"])
        installed = Path(b["state_bank"])
        bank = release / "simulator-source/libero/libero/init_files" / b["suite"] / installed.name
        for p in (installed, bank):
            checked(p, b["bank_sha256"])
        states = torch.load(bank, map_location="cpu", weights_only=False)
        require(list(states.shape) == b["shape"] and hasher(states[31]) == b["row31_fingerprint"], key)
        fingerprints = {i: hasher(states[i]) for i in (11, 13, 26, 27)}
        require(len(set(fingerprints.values())) == 4, f"Release rows coincide for {key}")
        for i in (26, 27):
            result[key + (i,)] = dict(bank_sha256=b["bank_sha256"], released_row_fingerprint=fingerprints[i],
                                      bddl_sha256=b["bddl_sha256"], horizon=b["native_horizon"])
    return result


def historical():
    import numpy as np
    from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
    from paired_inputs import FLOW, GOAL_PIN, check_tree, module, native

    ev = native()
    src = Path(os.environ["PAIRED_DEPS"])
    check_tree(src, GOAL_PIN)
    flow = module("matched_flow", FLOW/"live.py")
    runtime = flow.qualify_runtime(flow.load_fk())
    panel = parse(checked(ROOT/"outputs"/PR7.name/"prepare01pro/gpu09test/panel.json", PR7_PANEL))
    require(panel["runtime"]["weights"] == runtime and panel["runtime"]["pin"] == GOAL_PIN, "motor provenance")
    _, entries = ev.build_test_plan()
    banks = release_rows()
    by_index = {r["index"]: r for r in panel["rows"] if r["seed"] == 41}
    result = []
    for r in cases():
        old = by_index[r["index"]]
        entry = entries[r["index"]]
        require(all(old[k] == entry[k] == r[k] for k in ("index", "suite", "task_id", "regime", "embodiment"))
                and old["support_layout"] == r["support_layout"] == 11 and old["tape_sha256"] == r["support_sha256"]
                and old["support_seed"] == r["support_seed"] and old["shard"] == r["task_shard"], r)
        query = dict(old["query"], init_state_id=r["query_layout"], seed=r["query_seed"])
        require(query["arm"] == "UR5e" and query["variant"] == "BASE" and
                old["query"]["seed"] - 13 == query["seed"] - r["query_layout"], "query")
        tape = Path(old["tape"])
        checked(tape, r["support_sha256"])
        with np.load(tape, allow_pickle=False) as z:
            require(z["issued_mean"].shape == (3, 7) and z["valid"].tolist() == [True]*3, tape)
        spec = get_embodiment_spec(r["embodiment"])
        bank = banks[r["suite"], r["task_id"], r["query_layout"]]
        ref = panel["refs"][r["task_shard"]]
        require(bank["horizon"] == ref["horizon"] and spec.regime == r["regime"], "official case")
        result.append(dict(**r, query=query, true_map={k: getattr(spec, k) for k in IDENTITY},
                           original_tape=str(tape), nominal=old["tuned"], historical_symmeta=old["symmeta"], reference=bank,
                           support_physics=old["support_physics"], prompt=ref["prompt"],
                           action_schema=ref["action_schema"], model_class=ref["model_class"]))
    return result, runtime
