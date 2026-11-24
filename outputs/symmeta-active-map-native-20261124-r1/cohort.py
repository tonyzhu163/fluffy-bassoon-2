"""The prospectively assigned native cohort and official release-bank identities."""
import itertools
import os
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "symmeta-perceiver-retry-20261123-r1"))
from paired_records import atomic, checked, digest, encode, parse, require

TASKS = (("libero_10", 8), ("libero_goal", 6), ("libero_object", 8), ("libero_spatial", 8))
ROBOTS = ("UR5e", "IIWA")
FAMILIES = ("general+", "general-")
ESSENTIAL = ("raw", "true", "structured_passive", "direct_passive", "structured_lateral", "direct_lateral")
ALONG = ("structured_along", "direct_along")
BANK_SHA = "b0270a0d9ecb6ab8a467e702197f33427f0c654107eac5c7b5f9cf97d97dc799"
MANIFEST_SHA = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
RELEASE_PIN = "e5500c690c1498eca1b7050350654964190cdf70"


def assignments():
    rows = []
    for t, a, m, l in itertools.product(range(4), range(2), range(2), range(2)):
        suite, task = TASKS[t]
        layout = 28 + l
        rows.append(dict(index=len(rows), task_index=t, arm=ROBOTS[a], suite=suite, task=task,
                         family=FAMILIES[m], sign="+" if (t+a+m+l)%2 == 0 else "-",
                         map_seed=24112504+100*t+m, layout=layout, seed=41800000+10000*t+100*task+layout))
    return rows


def official_banks():
    import torch

    release = Path(os.environ["CROSSARM_RELEASE"])
    manifest = parse(checked(release/"assets/task_manifest.json", MANIFEST_SHA))
    fingerprint = runpy.run_path(str(release/"libero_remote/reproducibility.py"))["numerical_state_fingerprint"]
    bankfile = HERE.parent/"frozen-transfer-progress-20261124-r1/evidence/release-banks.json"
    refs = {}
    for b in parse(checked(bankfile, BANK_SHA))["tasks"]:
        key = b["suite"], b["task_id"]
        if key not in TASKS:
            continue
        entry, = (e for e in manifest["tasks"] if (e["suite"], e["base_task_id"], e["case_id"]) == (*key, "BASE"))
        require(entry["max_steps"] == b["native_horizon"] and entry["bddl_sha256"] == b["bddl_sha256"], key)
        checked(release/"assets"/entry["bddl_path"], b["bddl_sha256"])
        bank = Path(b["state_bank"])
        checked(bank, b["bank_sha256"])
        verified = release/"simulator-source/libero/libero/init_files"/b["suite"]/bank.name
        checked(verified, b["bank_sha256"])
        states = torch.load(bank, map_location="cpu", weights_only=False)
        hashes = {i:fingerprint(states[i]) for i in (11, 17, 28, 29, 31)}
        require(list(states.shape) == b["shape"] and hashes[31] == b["row31_fingerprint"]
                and len(set(hashes.values())) == len(hashes), key)
        for layout in (28, 29):
            refs[key+(layout,)] = dict(bank_sha256=b["bank_sha256"], released_row_fingerprint=hashes[layout],
                                       support_row_fingerprint=hashes[17], bddl_sha256=b["bddl_sha256"],
                                       horizon=entry["max_steps"])
    require(len(refs) == 8, "official task banks")
    return refs


def bounded_issue(nominal, estimate):
    """The motor already bounded the nominal proposal; use mapreader's q and A dtypes."""
    require(nominal.dtype == np.float32 and np.isfinite(nominal).all() and np.abs(nominal).max() <= 1, "nominal wire")
    result = nominal.copy()
    if estimate is not None:
        require(estimate.shape == (2,2) and np.isfinite(estimate).all(), "estimated map")
        q = nominal[:, :2].astype(np.float64)
        result[:, :2] = np.clip(q @ np.linalg.pinv(estimate).T, -1, 1)
    return result


def array_sha(a):
    a = np.ascontiguousarray(a)
    return digest(repr((a.shape, a.dtype.str)).encode()+a.tobytes())


def query(row):
    return dict(arm=row["arm"], suite=row["suite"], task_id=row["task"], variant="BASE",
                init_state_id=row["layout"], seed=row["seed"])


def order(row, methods):
    shift = row["index"] % len(methods)
    return methods[shift:] + methods[:shift]
