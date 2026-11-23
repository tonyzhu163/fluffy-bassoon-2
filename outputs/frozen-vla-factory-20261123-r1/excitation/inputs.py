"""Read the pinned command and kinematics exports; no policy loading."""
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
RELEASE = "e5500c690c1498eca1b7050350654964190cdf70"
SEED = 20261123
WORLD = OUTPUTS / "symmeta-camera-command-20261123-r1/evidence/world.json.gz"
PI0 = OUTPUTS / "symmeta-fk-crossarm-20261123-r1/evidence/gpu08pi0fix-cases.json.gz"


def load(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def supports():
    world, pi0 = load(WORLD), load(PI0)
    assert pi0["arm"] == "UR5e" and pi0["policy"] == "pi0" and pi0["fk_qualified"]
    assert world["method"] == "same_archived_passive_joint_fk"
    rows = []
    for motor, records, x_key, y_key in (
        ("pi05", world["rows"], "issued_mean", "delta_world_m"),
        ("pi0", pi0["rows"], "issued_xy", "world_delta_xy_m"),
    ):
        assert len(records) == 96
        for r in records:
            x = np.asarray(r[x_key])[:, :2]
            y = np.asarray(r[y_key])[:, :2]
            assert x.shape == y.shape == (3, 2) and np.isfinite([x, y]).all()
            rows.append(dict(motor=motor, task=f"{r['suite']}/{r['task_id']}",
                             index=r["index"], x=x, y=y, label=r["embodiment"],
                             tape_sha=r.get("support_sha256", r.get("archive_sha256"))))
    assert len({r["tape_sha"] for r in rows}) == 192
    return rows, {(r["suite"], r["task_id"]): r for r in pi0["resets"]}


def spread(x):
    s = np.linalg.svd(x, compute_uv=False)
    return float(s[-1] / s[0])


def rot(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def designs(x):
    dirs = x / np.linalg.norm(x, axis=1, keepdims=True)
    return {"one": np.tile(dirs[0], (3, 1)), "passive": dirs,
            "two": np.stack([dirs[0], rot(np.pi / 2) @ dirs[0], dirs[0]])}
