"""Count identifiable native gripper commands from the original bounded chunk tapes."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/mowenjie/fluffy-bassoon-2/outputs")
SCALE = BASE / "libero-xemb-ur5e-scale-20261120-r1"
CAMPAIGN = BASE / "xemb-icml-campaign-20261121-r1"
SOURCE_PIN = "d597ee0f1e99937bbeb2705132d709b514a4b5c2"
ARCHIVE_HASH = "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076"
PANELS = {
    "UR5e": (SCALE / "gpu09test", "test_live", "5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b"),
    "IIWA": (CAMPAIGN / "gpu09iiwa", "live", "490f25c5716aa067199c5b65e32caf6cd15421a4"),
    "Kinova3": (CAMPAIGN / "gpu06kinova3", "live", "465f33dc657ceab341dbceaf6b98355b94810059"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    with path.open("x") as file:
        json.dump(obj, file, indent=2, allow_nan=False)
        file.write("\n")


def read_tape(path, mode):
    with np.load(path, allow_pickle=False) as tape:
        valid = tape["valid"]
        assert valid.dtype == bool and valid[:3].all() and not valid[3:].any()
        g = tape["issued_mean"][valid, 6].astype(float)
        jaw = tape["delta_jaw"][valid].astype(float)
        first = tape["frame0_state"]
    assert np.isfinite(jaw).all() and np.abs(g).max() <= 1
    # Frame zero retains the two signed slides; the endpoints are signed aperture in metres.
    np.testing.assert_allclose(jaw[:, 1] - jaw[:, 0],
                               [first[-2] - first[-1], *jaw[:-1, 1]], atol=1e-8, rtol=0)
    if mode == "active":
        np.testing.assert_array_equal(g, [1, -1, 1])
    return dict(path=str(path), sha256=sha(path), g=g.tolist(), delta_mm=(jaw[:, 0]*1000).tolist(),
                width_mm=(jaw[:, 1]*1000).tolist())


def run(out):
    splits = json.loads((Path(__file__).resolve().parent.parent /
                         "libero-meta-learning-20261117-r1/SPLITS.json").read_text())
    source = SCALE / "gpu01/archive/archive.json"
    assert sha(source) == ARCHIVE_HASH
    assert (source.parents[1] / "launch.txt").read_text().split()[0] == SOURCE_PIN
    expected = {(a, s, t, i) for a in ("Panda", "Sawyer") for s, tasks in splits["train"].items()
                for t in tasks for i in range(4)}
    rows = []
    for r in json.loads(source.read_bytes())["trials"]:
        if r["split"] != "train" or r["embodiment"] != "nom":
            continue
        ident = tuple(r[k] for k in ("arm", "suite", "task_id", "init_state_id"))
        expected.remove(ident)
        assert (r["yaw_deg"], r["xy_scale"], r["gripper_sign"], r["passive_controls"], r["active_controls"]) == (0, 1, 1, 15, 15)
        rows.append(dict(arm=r["arm"], suite=r["suite"], task=r["task_id"], start=r["init_state_id"],
                         seed=r["seed"], physics=r["physics"], pin=SOURCE_PIN,
                         **{m:read_tape(source.parent / r[m+"_path"], m) for m in ("active", "passive")}))
    assert not expected and len(rows) == 192
    suites = sorted(splits["test"])
    tasks = [(s, t) for s in suites for t in splits["test"][s]]
    for arm, (directory, subdir, pin) in PANELS.items():
        assert (directory / "launch.txt").read_text().split()[0] == pin
        for i, (s, t) in enumerate(tasks):
            shard = directory / subdir / f"shard-{i}"
            meta = json.loads((shard / "support-meta.json").read_text())[0]
            assert meta["embodiment"] == "fam_nom"
            tapes = {}
            for mode in ("active", "passive"):
                path = shard / "support" / f"00-fam_nom-{mode}.npz"
                assert Path(meta[mode+"_npz"]) == path
                tapes[mode] = read_tape(path, mode)
            rows.append(dict(arm=arm, suite=s, task=t, start=11,
                             seed=31800000+10000*suites.index(s)+100*t+11, physics=meta["physics"], pin=pin, **tapes))
    write(out / "archives.json", rows)
    report = {}
    for arm in ("Panda", "Sawyer", *PANELS):
        rr = [r for r in rows if r["arm"] == arm]
        stats = {}
        for mode in ("active", "passive"):
            g = np.array([r[mode]["g"] for r in rr])
            d = np.array([r[mode]["delta_mm"] for r in rr])
            stats[mode] = dict(tapes=len(rr), chunks=g.size, positive_mean_chunks=int((g>0).sum()),
                               guaranteed_all_negative=int((g < -.8 - 1e-6).sum()),
                               ambiguous_chunks=int((np.abs(g) <= .8+1e-6).sum()),
                               command_interval=[g.min(), g.max()], shrinking_chunks=int((d<0).sum()),
                               delta_mm_min=d.min(axis=0).tolist(), delta_mm_median=np.median(d,axis=0).tolist(),
                               delta_mm_max=d.max(axis=0).tolist())
        report[arm] = stats
    # The source and target tasks differ. Across held-out arms compare the same eight tasks.
    widths = {a:np.array([r["active"]["width_mm"] for r in rows if r["arm"]==a]) for a in PANELS}
    report["heldout_same_task_max_active_endpoint_range_mm"] = np.ptp(list(widths.values()), axis=0).max(axis=0).tolist()
    write(out / "archive-summary.json", report)
    print(json.dumps(report), flush=True)
