"""Archive authentication. Oracle receipt and held-displacement data stay outside the gate."""
from collections import Counter, defaultdict
import hashlib
import io
import json
from itertools import product
from pathlib import Path

import numpy as np
from mechanism import HERE, Templates, support
from observations import model_rows, STRATEGY, ridge
from design import SCHEDULES
from score import sensor_audit

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
UP = ROOT / "symmeta-active-map-reader-20261124-r1"
BANK = UP / "cpu01"
OLD = ROOT / "symmeta-query-direction-probe-20261124-r1/cpu03"
LAWFILE = HERE.parent / "frozen-vla-motion-models-20261123-r1/evidence/source-fit.json"
LAW_HASH = "acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73"
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
FRESH = (8, 6, 8, 8)
PRIOR = (5, 3, 1, 0)
MODES = ("passive", "along+", "along-", "orth+", "orth-")
SIGNS = ("general+", "general-")
SOURCE_ARMS = ("Panda", "Sawyer")
TARGET_ARMS = ("UR5e", "IIWA")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, sort_keys=True) + "\n")


def read(path):
    return json.loads(path.read_text())


def data_ref(path, manifest):
    raw = path.read_bytes()
    manifest[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw


def load_np(path, manifest, fields):
    with np.load(io.BytesIO(data_ref(path, manifest)), allow_pickle=False) as data:
        return {k:data[k] for k in fields}


def source_law(manifest):
    fit = json.loads(data_ref(LAWFILE, manifest))
    assert manifest[str(LAWFILE)] == LAW_HASH
    law, residual = np.array(fit["prior"]), float(fit["residual_power"])
    for arm in SOURCE_ARMS:
        rt = json.loads(data_ref(BANK/f"runtime-{arm}.json", manifest))
        np.testing.assert_array_equal(law, rt["prior"])
        assert rt["residual"] == residual and rt["source_hash"] == LAW_HASH
    return law, residual


def verify_query(meta, q, manifest):
    ref = meta.get("query_record", meta.get("query"))
    if isinstance(ref, list):
        copies = []
        for r in ref:
            p = Path(r["path"])
            z = load_np(p, manifest, ["issued_mean", "valid"])
            assert manifest[str(p)] == r["sha256"]
            assert r["start"] != meta["source_record"]["start"]
            copies.append(z["issued_mean"][z["valid"]])
        raw = np.repeat(np.concatenate(copies), 5, axis=0)
    else:
        p = Path(ref["path"])
        obj = json.loads(data_ref(p, manifest))
        assert manifest[str(p)] == ref["sha256"] and obj["method"] == "raw"
        assert obj["case"]["init_state_id"] == 14 and obj["case"]["arm"] == meta["arm"]
        assert meta["support_reset"]["released_row_fingerprint"] != meta["held_reset"]["released_row_fingerprint"]
        raw = np.concatenate([c["cmd"] for c in obj["chunks"]]).astype(np.float32)
    np.testing.assert_array_equal(q, raw)


def group(name):
    return name.rstrip("+-")


def folders(split):
    if split in ("fresh", "developmental"):
        bank, ids = (BANK, FRESH) if split == "fresh" else (OLD, PRIOR)
        return [bank/f"{arm}-{suite}-{i}" for arm in TARGET_ARMS for suite,i in zip(SUITES, ids)]
    found = []
    for arm in SOURCE_ARMS:
        for suite in SUITES:
            dirs = sorted(BANK.glob(f"{arm}-{suite}-*"), key=lambda p:int(p.name.rsplit("-",1)[1]))
            assert len(dirs) == 6
            found.extend(dirs[:4] if split=="train" else dirs[4:5] if split=="val" else dirs[5:])
    return found


def cases(split, law, residual, manifest):
    for folder in folders(split):
        m = json.loads(data_ref(folder/"case.json", manifest))
        expected = "target" if split=="fresh" else split
        assert m["complete"] and m.get("split", "developmental") == expected
        arch = load_np(folder/"archive-controls.npz", manifest, ["query", "support_chunk_means"])
        q = arch["query"]
        verify_query(m, q, manifest)
        for fam, record in sorted(m["maps"].items()):
            A = np.array(record["A"], float)
            assert np.linalg.norm(A, np.inf) <= 1+1e-12
            hx, hy = np.empty((0, 4)), np.array([])
            if split in ("test", "val"):
                hpath = folder/f"{fam}-held.npz"
                if hpath.exists():
                    h = load_np(hpath, manifest, ["fk","issued","physical","goal_m"])
                    assert sensor_audit(h, A) < 1e-8
                    end = len(h["issued"]) // 5 * 5
                    np.testing.assert_array_equal(h["issued"], q[m["prefix"]:m["prefix"]+len(h["issued"])].astype(np.float32))
                    if end >= 10:
                        cmd = h["issued"][:end,:3].astype(float).reshape(-1,5,3).mean(axis=1)
                        raw_x, offset = model_rows(cmd, law)
                        hx = raw_x[2:]
                        hy = (np.diff(h["fk"][:end+1:5,:2],axis=0)/.05-offset)[1:].ravel()
            rows, issued, goal = [], [], []
            for mode in MODES:
                path = folder/f"{fam}-{mode}.npz"
                z = load_np(path, manifest, ["issued","fk","time","physical","goal_m"])
                receipt_error = sensor_audit(z, A)
                assert receipt_error < 1e-8
                x, y = support(z["issued"], z["fk"], z["time"], law, residual)
                u = z["issued"][:,:2].astype(float)
                s = np.linalg.svd(u.reshape(3,5,2).mean(axis=1), compute_uv=False)
                if mode=="passive":
                    np.testing.assert_array_equal(z["issued"], np.repeat(arch["support_chunk_means"],5,axis=0).astype(np.float32))
                issued.append(u)
                goal.append(z["goal_m"][:,:2]/.05)
                rows.append(dict(task=f'{m["suite"]}/{m["task"]}', arm=m["arm"], family=fam, mode=mode,
                                 strategy=STRATEGY[mode], rank_ratio=s[1]/s[0] if s[0] else 0.,
                                 support_effort=float(np.square(u).sum()), A=A, q=q[:,:2].astype(float),
                                 x=x, y=y, hx=hx, hy=hy, u=u, physical=z["physical"], R=ridge(x,y,residual),
                                 max_receipt_error_m=receipt_error, parent=str(path)))
            # Pool only immutable receipt checks across schedules, never gate information.
            U = np.concatenate(issued)
            fit, _, rank, _ = np.linalg.lstsq(U, np.concatenate(goal), rcond=None)
            assert rank==2
            np.testing.assert_allclose(fit.T, A, rtol=0, atol=5e-6)
            for row in rows:
                row["receipt_fit_max_error"] = float(abs(fit.T-A).max())
                yield row


def key(r):
    return tuple(r[k] for k in ("task","arm","family","mode"))


def balanced(rows):
    """Give each branch of task/body/class/strategy/draw/pulse-sign tree equal mass."""
    levels = [(r["task"],r["arm"],group(r["family"]),r["strategy"],r["family"],r["mode"]) for r in rows]
    assert len(set(levels))==len(rows)
    children = defaultdict(set)
    for path in levels:
        for i, item in enumerate(path):
            children[path[:i]].add(item)
    w = np.array([np.prod([1 / len(children[path[:i]]) for i in range(len(path))]) for path in levels])
    np.testing.assert_allclose(w.sum(),1)
    return w


def template_bank(train):
    originals = [r for r in train if r["mode"]=="passive"]
    weights = balanced(originals)
    maps, mass, lineage = {}, Counter(), []
    for row, prior in zip(originals, weights, strict=True):
        for flip, s, t in product((False,True),(-1,1),(-1,1)):
            P = np.diag([s,t])[:, ::-1 if flip else 1]
            A = row["A"] @ P.T
            u = row["u"] @ P.T
            assert abs(u).max()<=1 and np.linalg.norm(A,np.inf)<=1+1e-12
            np.testing.assert_array_equal((u @ A.T).astype(np.float32),row["physical"][:,:2])
            k = tuple(A.round(12).ravel())
            maps.setdefault(k,A)
            mass[k] += prior/8
            lineage.append(dict(parent=row["parent"], transform=P.tolist(), key=list(k), mass=prior/8))
    keys = sorted(maps)
    return Templates(np.stack([maps[k] for k in keys]), np.array([mass[k] for k in keys])), lineage
