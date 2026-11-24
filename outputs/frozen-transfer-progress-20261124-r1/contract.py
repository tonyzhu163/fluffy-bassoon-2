"""Immutable case keys, source law and first PR7 prediction export."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parent
SHARED = Path("/mowenjie/fluffy-bassoon-2/outputs")
TASKS = (("libero_10", 5), ("libero_10", 8), ("libero_goal", 3), ("libero_goal", 6),
         ("libero_object", 1), ("libero_object", 8), ("libero_spatial", 0), ("libero_spatial", 8))
ARMS = {"IIWA": ("gpu09iiwa", "c38a7951d22bc469f845950850191763f5803281f9480bc144d8957d52ac8e04"),
        "Kinova3": ("gpu06kinova3", "74587eaaf87ce69d733083b068f27b46fc62619c3a7ffe2ffe8248816949783e")}
MODES = ("raw", "tuned", "fixed", "progress")
IDENTITY = dict(yaw_deg=0.0, xy_scale=1.0, gripper_sign=1)
PRED_SHA = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
SOURCE_SHA = "acf6150ecd7937733d77516cabfaca4a7648531009e3d7c3264c025170279f73"
WEIGHT_SHA = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
RELEASE_SHA = "e5500c690c1498eca1b7050350654964190cdf70"
MANIFEST_SHA = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"


def require(ok, message):
    if not ok:
        raise ValueError(str(message))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(data, expected):
    require(digest(data) == expected, f"Input digest differs from {expected}")
    return data


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("x") as stream:
        json.dump(obj, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    require(not path.exists(), path)
    tmp.replace(path)


def source_law():
    path = OUTPUTS / "frozen-vla-motion-models-20261123-r1/evidence/source-fit.json"
    obj = json.loads(checked(path.read_bytes(), SOURCE_SHA))
    require(obj["counts"] == {"Panda": 96, "Sawyer": 96} and
            all(r["arm"] in obj["counts"] and r["task"] not in {f"{s}/{i}" for s, i in TASKS}
                for r in obj["sources"]), "source-only nominal model")
    result = np.asarray(obj["prior"], dtype=float)
    require(result.shape == (6, 3) and np.isfinite(result).all(), "source law")
    return result


def cases(arm, *, verify_tapes=False):
    folder = OUTPUTS / "symmeta-perceiver-retry-20261123-r1/evidence/train02pro"
    pred = json.loads(checked((folder/"predictions.json").read_bytes(), PRED_SHA))
    rows = pred["rows"]
    require(pred["schema"] == 1 and len({(r["panel"], r["index"], r["seed"]) for r in rows}) ==
            len(rows) == 1152, "unique prediction keys")
    panel, sha = ARMS[arm]
    archive = OUTPUTS / f"symmeta-fk-crossarm-20261123-r1/evidence/{panel}-cases.json.gz"
    data = json.loads(gzip.decompress(checked(archive.read_bytes(), sha)))
    require(data["arm"] == arm and data["panel"] == panel and data["policy"] == "pi05", "wrong archive")
    chosen = [r for r in rows if (r["panel"], r["embodiment"], r["seed"]) == (panel, "fam_nom", 41)]
    require(len(chosen) == len(TASKS), "ambiguous nominal predictions")
    result = []
    for i, (suite, task) in enumerate(TASKS):
        p, = (r for r in chosen if (r["suite"], r["task_id"]) == (suite, task))
        r, = (r for r in data["rows"] if r["index"] == p["index"])
        seed_offset = 10000 * (i // 2) + 100 * task
        require(all(p[k] == r[k] for k in ("index", "suite", "task_id", "embodiment", "regime")) and
                p["index"] == 12*i and p["tape_sha256"] == r["support_sha256"] and
                r["support_layout"] == 11 and r["support_seed"] == 31800011 + seed_offset,
                f"support join {panel}/{suite}/{task}")
        c = p["tuned"]
        require(set(c) == set(IDENTITY) and np.isfinite(list(c.values())).all() and c["xy_scale"] > 0
                and c["gripper_sign"] in (-1, 1), "invalid PR7 correction")
        if verify_tapes:
            shard = SHARED / "xemb-icml-campaign-20261121-r1" / panel / "live" / f"shard-{i}"
            tape = shard/"support/00-fam_nom-passive.npz"
            meta, = (m for m in json.loads((shard/"support-meta.json").read_bytes())
                     if m["embodiment"] == "fam_nom")
            require(meta["passive_npz"] == str(tape) and meta["physics"] == r["physics"], "wrong passive tape")
            checked(tape.read_bytes(), p["tape_sha256"])
            with np.load(tape, allow_pickle=False) as z:
                require(np.array_equal(z["valid"], np.arange(24) < 3), "invalid three passive blocks")
                if "chunk_steps" in z:
                    require(np.array_equal(z["chunk_steps"], z["valid"]*5), "incomplete support")
        result.append(dict(query=dict(arm=arm, suite=suite, task_id=task, variant="BASE", init_state_id=31,
                                      seed=41800031 + seed_offset),
                           support=dict(panel=panel, index=p["index"], reader_seed=41, suite=suite, task_id=task,
                                        shift="fam_nom", layout=11, seed=r["support_seed"], controls=15,
                                        tape_sha256=p["tape_sha256"], physics=r["physics"],
                                        predictions_sha256=PRED_SHA, tuned=c)))
    return result


def check_weights(path):
    meta = json.loads(checked((path/"READY.json").read_bytes(), WEIGHT_SHA))
    require(meta["status"] == "verified" and Path(meta["checkpoint_path"]) == path, "checkpoint identity")
    for entry in meta["files"]:
        file = path / entry["name"]
        require(file.stat().st_size == entry["bytes"], file)
        h = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(8 << 20), b""):
                h.update(block)
        require(h.hexdigest() == entry["sha256"], file)
