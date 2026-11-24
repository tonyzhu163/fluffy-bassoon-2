"""Authenticated archived endpoint factors. One returned row per physical case and reader seed."""
import csv
import gzip
import hashlib
import itertools
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path("/mowenjie/fluffy-bassoon-2/outputs")
RETRY = OUT / "symmeta-perceiver-retry-20261123-r1"
SRC = OUT / "symmeta-omitted-chunk-20261124-r1"
OLD = REPO / "outputs/symmeta-yaw-blend-20261124-r1/metrics.json"
LABELS = REPO / "src/fluffy_bassoon/xemb/embodiment.py"
SEEDS = (41, 42, 43)
PANELS = ("gpu09test", "gpu09iiwa", "gpu06kinova3", "gpu08pi0fix")
SRC_TASKS = dict(cal=("libero_10/6", "libero_goal/8", "libero_object/5", "libero_spatial/6"),
                 eval=("libero_10/4", "libero_10/7", "libero_goal/4", "libero_goal/9",
                       "libero_object/4", "libero_object/9", "libero_spatial/5", "libero_spatial/9"))
CHECKED = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    actual = digest(path)
    assert actual == expected, (path, actual, expected)
    CHECKED[str(path)] = actual
    return path


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def labels(kind):
    checked(LABELS, "e0ed86b5fc85bbde7b4dd95b0ec2b520b792325d0e6f135465132cc2dc99c41c")
    return runpy.run_path(str(LABELS))[kind + "_EMBODIMENTS"]


def source(role):
    """Only the named label file is opened. Identity order is independently joined in each input."""
    seal = read_json(checked(SRC / "data01/source-seal.json",
                     "c1442a9c972ee1c7d4255956c98d19186bd174814af2251d451685516ec2bbd1"))
    assert set(seal["roles"][role]) == set(SRC_TASKS[role])
    assert set(seal["roles"]["readerfit"]).isdisjoint(SRC_TASKS["cal"] + SRC_TASKS["eval"])
    meta = read_json(checked(SRC / "data01/source-membership.json", seal["sha256"]["source-membership.json"]))
    member = {r["case_id"]: r for r in meta if r["split"] == role}
    receipt = read_json(checked(SRC / "pred01/prediction-receipt.json",
                        "67d3fb36be573a39b3db61663cfb0261b294ca081d13dd62356c82e2065a32c1"))
    assert receipt["reader_sha256"] == "22ac89611e555ccbc4be31b6f31734e8ad972fa2d3898dd316ff75e90db1d1f9"
    assert receipt["data_seal"] == digest(SRC / "data01/source-seal.json")
    freeze = read_json(checked(SRC / "judge-evidence/freeze01/seal.json",
                       "32626107e00556e010fa67adc6c8ed2a8ea3872d4379fd67ac2b2eb824990356"))
    replay = read_json(checked(SRC / "replay01/replay.json",
                       "37385ea504c15b50743944f1be5ec028a3ec2eb706015733f372a99318ffbaab"))
    assert replay["prediction_sha256"] == receipt["reader_sha256"] and replay["exact_yaws"] == 16128
    ckpt = read_json(checked(SRC / "reader01/checkpoints.json", receipt["checkpoints"]))
    assert {(r["model"], r["seed"]) for r in ckpt["rows"]} == set(itertools.product(("tuned", "geom"), SEEDS))
    name = f"compact-{role}.npz"
    cols = ("case_id", "seed", "task", "arm", "split", "physical_trace_id", "full_yaw", "full_valid",
            "tuned_gain", "tuned_sign")
    with np.load(checked(SRC / "pred01/readers.npz", receipt["reader_sha256"])) as z:
        mask = z["split"] == role
        data = {k: z[k][mask] for k in cols}
    order = np.lexsort((data["seed"], data["case_id"]))
    with np.load(checked(SRC / "judge-evidence/freeze01" / name, freeze["outputs"][name])) as compact:
        other = np.lexsort((compact["seed"], compact["case_id"]))
        for k in cols:
            np.testing.assert_array_equal(data[k][order], compact[k][other])
    name = f"labels-{role}.npz"
    with np.load(checked(SRC / "data01" / name, seal["sha256"][name])) as z:
        by_case = {c: (float(y), float(g), int(s)) for c, y, g, s in
                   zip(*(z[k] for k in ("case_id", "true_yaw", "true_gain", "true_sign")), strict=True)}
        assert len(by_case) == len(z["case_id"]) == len(member) == len(SRC_TASKS[role]) * 56
    assert set(by_case) == set(member)
    specs = {(v.yaw_deg, float(np.float32(v.xy_scale)), v.gripper_sign): v.name for v in labels("TRAIN")}
    out = []
    for n in order:
        cid = str(data["case_id"][n])
        task, panel, trace = (str(data[k][n]) for k in ("task", "arm", "physical_trace_id"))
        m = member[cid]
        assert (m["task"], m["arm"], m["tape_sha256"]) == (task, panel, trace)
        y, g, sign = by_case[cid]
        out.append(dict(case=cid, task=task, panel=panel, trace=trace, shift=specs[y, g, sign],
                        regime=role, start=m["init_state_id"], seed=int(data["seed"][n]), y=y, g=g, s=sign,
                        t=float(data["full_yaw"][n, 0]), h=float(data["full_yaw"][n, 1]),
                        tv=bool(data["full_valid"][n, 0]), hv=bool(data["full_valid"][n, 1]),
                        gain=float(data["tuned_gain"][n]), sign=int(data["tuned_sign"][n])))
    grid = Counter((r["task"], r["panel"], r["shift"], r["start"], r["seed"]) for r in out)
    assert grid == Counter(itertools.product(SRC_TASKS[role], ("Panda","Sawyer"), specs.values(), range(4), SEEDS))
    return out


def target():
    pred = read_json(checked(RETRY / "train02pro/predictions.json",
                     "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"))
    path = checked(RETRY / "native02g4090-independent/pairs.csv.gz",
                   "be37297b1950f4ec3463a55ce9e60617eba2318c42c363504d42844eb3d8cf1d")
    with gzip.open(path, "rt") as f:
        native = {(r["panel"],int(r["index"]),int(r["seed"])): r for r in csv.DictReader(f)}
    specs = {v.name:v for v in labels("EVAL")}
    old = read_json(checked(OLD, "7b5732d8c43536fb1a4c79c210b25ff6b7d53d6fa463922efceb24a04f7e106f"))
    tasks = {f"{suite}/{i}" for suite,i in old["task_clusters"]}
    assert tasks.isdisjoint(SRC_TASKS["cal"] + SRC_TASKS["eval"])
    out = []
    for r in pred["rows"]:
        panel, i, seed = (r[k] for k in ("panel", "index", "seed"))
        n = native.pop((panel,i,seed))
        assert all(n[k] == str(r[k]) for k in ("suite","task_id","embodiment","regime","tape_sha256"))
        v = specs[r["embodiment"]]
        assert v.regime == r["regime"]
        t, h = r["tuned"], r["symmeta"]
        out.append(dict(case=f"{panel}/{i:02d}", panel=panel, task=f"{r['suite']}/{r['task_id']}",
                        trace=r["tape_sha256"], shift=v.name, regime=v.regime, start=11, seed=seed,
                        y=v.yaw_deg, g=v.xy_scale, s=v.gripper_sign, t=t["yaw_deg"], h=h["yaw_deg"],
                        tv=True, hv=True, gain=t["xy_scale"], sign=t["gripper_sign"]))
    assert not native and len(out) == 1152
    assert Counter((r["panel"],r["task"],r["shift"],r["seed"]) for r in out) == Counter(
        itertools.product(PANELS, tasks, specs, SEEDS))
    return out
