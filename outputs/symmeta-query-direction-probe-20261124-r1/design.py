"""Frozen input identities, causal intervention and source transfer law."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from fluffy_bassoon.xemb.wider_family import interface_matrix, sample_interfaces

HERE = Path(__file__).resolve().parent
ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
NATIVE = "frozen-vla-multiarm-confirmation-20261123-r1"
MOTION = "frozen-vla-motion-models-20261123-r1"
TASKS = [("libero_10", 5), ("libero_goal", 3), ("libero_object", 1), ("libero_spatial", 0)]
FAMILIES = ["native", "family", "swap", "general"]
SCHEDULES = ["passive", "along+", "along-", "orth+", "orth-"]


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def checked(path, digest):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"Archive changed: {path}")
    return raw


def source():
    refs = json.loads((HERE.parent / NATIVE / "evidence/production-native.json").read_bytes())
    fit = json.loads(checked(HERE.parent / MOTION / "evidence/source-fit.json", refs["source_fit_sha256"]))
    increments = []
    for row in fit["sources"]:
        with np.load(io.BytesIO(checked(row["path"], row["sha256"]))) as z:
            cmd = z["issued_mean"][z["valid"], :2].astype(float)
            assert cmd.shape == (3, 2)
            increments.extend(np.linalg.norm(np.diff(cmd, axis=0), axis=1))
    assert len(increments) == 384
    return refs, fit, float(np.median(increments))


def inputs(refs, arm, suite, task):
    key = f"{suite}/{task}"
    ref = refs["arms"][arm]["tasks"][key]
    mat = json.loads(checked(HERE.parent / MOTION / "evidence/matrices.json", refs["matrix_file_sha256"]))
    sup = mat[f"{arm}/{key}"]["support"]
    assert sup["sha256"] == ref["support_sha256"] and sup["start"] == 11 and sup["arm"] == arm
    with np.load(io.BytesIO(checked(sup["path"], sup["sha256"]))) as z:
        x = z["issued_mean"][z["valid"]].astype(np.float64)
        first = z["frame0_state"]
    path = ROOT / NATIVE / f"gpu01-{arm}/{suite}-{task}-14-raw.json"
    digest = ref["methods"]["raw"]["receipt_sha256"]
    obj = json.loads(checked(path, digest))
    cmd = np.concatenate([c["cmd"] for c in obj["chunks"]]).astype(np.float32)
    xyz = np.concatenate([c["xyz"] for c in obj["chunks"]])
    assert obj["method"] == "raw" and obj["case"]["init_state_id"] == 14 and obj["case"]["arm"] == arm
    assert obj["record"]["reset"]["physics"] == ref["physical_reset"] != sup["physics"]
    assert cmd.shape == (ref["methods"]["raw"]["controls"], 7) and x.shape == (3, 7)
    assert np.abs(cmd).max() <= 1 and np.abs(x).max() <= 1
    return sup, x, first, obj, cmd, xyz, dict(path=str(path), sha256=digest)


def maps(task_index):
    yield "native", np.eye(2), 0
    for index, family in enumerate(FAMILIES[1:], 1):
        rng = np.random.default_rng(112400 + 10 * task_index + index)
        for draw in range(1, 10001):
            a = interface_matrix(sample_interfaces(rng, family, 1))[0]
            if np.linalg.norm(a, np.inf) > 1:
                continue
            if family == "general" and np.sign(np.linalg.det(a)) != (-1) ** task_index:
                continue
            yield family, a, draw
            break
        else:
            raise ValueError("No bounded map")


def schedule(chunks, mode, requested):
    """Release one previously archived chunk at a time, with public common bounds."""
    first_mag = 0.
    for i, chunk in enumerate(chunks):
        v = chunk.copy()
        if i == 0:
            yield v
            direction = v[:2]
            norm = np.linalg.norm(direction)
            u = direction / norm if norm else np.zeros(2)
            axes = np.stack((u, [-u[1], u[0]]))
            continue
        if mode != "passive" and norm:
            component = np.max(np.abs(axes), axis=0)
            headroom = ((1 - np.abs(v[:2])) / component).min()
            mag = min(requested if i == 1 else first_mag, headroom)
            if i == 1:
                first_mag = mag
            sign = (1 if mode.endswith("+") else -1) * (1 if i == 1 else -1)
            v[:2] += sign * mag * axes[int(mode.startswith("orth"))]
        assert np.max(np.abs(v)) <= 1 + 1e-14
        yield v
