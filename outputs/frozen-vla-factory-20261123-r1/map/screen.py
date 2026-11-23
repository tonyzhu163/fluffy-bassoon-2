"""Offline XY commanded-direction screen on frozen pi0/pi0.5 traces. NumPy only."""

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.wider_family import command_spread, ridge_linear_map, rot2_batch


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
FRICTION = OUTPUTS / "libero-xemb-friction-dev-20261119-r1" / "cpu01"
CAMERA = OUTPUTS / "symmeta-camera-command-20261123-r1" / "evidence"
CROSSARM = OUTPUTS / "symmeta-fk-crossarm-20261123-r1" / "evidence"
PIN = "757a44cd4374833cad790852df8b87ddf412e638"
PENALTIES = (0., .0001, .001, .01, .1, 1., 10.)
CAPS = (.25, .5, .9, 1.5, 2.5)
METHODS = ("identity", "procrustes", "ridge", "bounded", "orthogonal", "orthogonal_bounded")
RAYS = rot2_batch(np.arange(128) * (360 / 128))[:, :, 0]
I = np.eye(2)


@dataclass
class Trace:
    panel: str
    task: str
    key: str
    recorded_interface: str
    issued: np.ndarray
    motion: np.ndarray


def unpack(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def source_traces():
    traces = []
    files = sorted(FRICTION.glob("*-default-ordinary.npz"))
    assert len(files) == 8
    for file in files:
        task = file.name.removesuffix("-default-ordinary.npz")
        with np.load(file, allow_pickle=False) as tape:
            actions = tape["issued"][:15, :2]
            assert actions.shape == (15, 2) and np.max(np.abs(actions)) <= 1
            issued = actions.reshape(3, 5, 2).mean(axis=1)
            motion = np.diff(tape["robot0_eef_pos"][:16:5, :2], axis=0)
        traces.append(Trace("development", task, task, "nom", issued, motion))
    return traces, files


def target_traces():
    archive = CAMERA / "world.json.gz"
    visual = CAMERA / "roll.json.gz"
    world = unpack(archive)
    assert world["status"] == "complete" and len(world["rows"]) == 96
    upright = [r for r in unpack(visual)["rows"] if r["roll_deg"] == 0]
    visual_by_sha = {r["support_sha256"]: r for r in upright}
    assert len(visual_by_sha) == len(upright) == 96
    fk, optical = [], []
    for r in world["rows"]:
        key = r["archive_sha256"]
        task = f"{r['suite']}-{r['task_id']}"
        x = np.asarray(r["issued_mean"])[:, :2]
        args = ("UR5e_pi05", task, key, r["embodiment"], x)
        fk.append(Trace(*args, np.asarray(r["delta_world_m"])[:, :2]))
        v = visual_by_sha[key]
        assert (r["suite"], r["task_id"], r["embodiment"]) == (v["suite"], v["task_id"], v["embodiment"])
        optical.append(Trace(*args, np.asarray(v["vectors"]["hs_projected_xy"])))
    files = sorted(CROSSARM.glob("*-cases.json.gz"))
    assert len(files) == 3
    for path in files:
        data = unpack(path)
        panel = data["arm"] + "_" + data["policy"]
        assert len(data["rows"]) == 96
        for r in data["rows"]:
            if r["fk_available"]:
                fk.append(Trace(panel, f"{r['suite']}-{r['task_id']}", r["support_sha256"],
                                r["embodiment"], np.asarray(r["issued_xy"]), np.asarray(r["world_delta_xy_m"])))
    assert len(fk) == len({r.key for r in fk}) == 360 and len(optical) == 96
    return fk, optical, [archive, visual, *files]


def interventions(family, n, rng):
    """B maps synthetic issued coordinates to recorded issued coordinates."""
    b = np.tile(I, (n, 1, 1))
    if family == "recorded":
        return b
    if family in ("shear", "mirror_shear"):
        b[:, 0, 1] = rng.uniform(.2, .7, n) * rng.choice([-1, 1], n)
    if family in ("mirror", "mirror_shear"):
        b[:, :, 1] *= -1
    # Two independent rotations vary shear and mirror axes. 2-norm and parity are preserved.
    b = rot2_batch(rng.uniform(-180, 180, n)) @ b @ rot2_batch(rng.uniform(-180, 180, n))
    # For every primitive x in the original unit box, ||B^-1 x||_inf <= 1.
    return b * np.linalg.norm(np.linalg.inv(b), ord=np.inf, axis=(1, 2))[:, None, None]


def pairs(traces, changes):
    x = np.stack([r.issued for r in traces])
    d = np.stack([r.motion for r in traces])
    assert x.shape == d.shape == (len(traces), 3, 2) and np.isfinite(x).all() and np.isfinite(d).all()
    # Batch order is trace then transform. No recorded or counterfactual target label enters.
    x = (x[:, None] @ np.linalg.inv(changes).transpose(0, 2, 1)).reshape(-1, 3, 2)
    d = np.repeat(d, len(changes), axis=0)
    assert np.max(np.abs(x)) <= 1 + 1e-10
    for arr in (x, d):
        norm = np.linalg.norm(arr, axis=(1, 2), keepdims=True)
        assert (norm > 0).all()
        arr /= norm  # predictions are scored for direction only
    return x, d


def similarity(x, d, proper):
    """Inverse map mapping world motion d to issued command x, up to a scalar."""
    u, s, vh = np.linalg.svd(x.transpose(0, 2, 1) @ d)
    signs = np.ones_like(s)
    if proper:
        signs[:, -1] = np.sign(np.linalg.det(u @ vh))
    gain = np.sum(s * signs, axis=1) / np.sum(d * d, axis=(1, 2))
    return gain[:, None, None] * ((u * signs[:, None]) @ vh), gain


def fit(x, d, method, penalty=0., cap=None):
    if method == "identity":
        return np.broadcast_to(I, (len(d), 2, 2))
    base, gain = similarity(x, d, proper=not method.startswith("orthogonal"))
    if method in ("procrustes", "orthogonal"):
        return base
    # Same normalized ridge around each support's similarity, with and without the cap.
    ridge = ridge_linear_map(d, x, np.ones(d.shape[:2]), base, penalty)
    if cap is None:
        return ridge
    u, s, vh = np.linalg.svd(ridge - base)
    s = np.minimum(s, (cap * gain)[:, None])
    return base + (u * s[:, None]) @ vh


def truth(traces, b):
    recorded = []
    for r in traces:
        spec = get_embodiment_spec(r.recorded_interface)
        recorded.append(spec.xy_scale * rot2_batch(spec.yaw_deg))
    return (np.stack(recorded)[:, None] @ b).reshape(-1, 2, 2)


def score(correction, label):
    transfer = label @ correction
    achieved = RAYS @ transfer.transpose(0, 2, 1)
    # A command with zero XY output has undefined direction. Fail visibly if one occurs.
    assert np.all(np.linalg.norm(achieved, axis=-1) > 1e-13)
    dot = np.sum(achieved * RAYS, axis=-1)
    cross = RAYS[:, 0] * achieved[:, :, 1] - RAYS[:, 1] * achieved[:, :, 0]
    error = np.rad2deg(np.arctan2(np.abs(cross), dot)).mean(axis=1)
    return error, (np.linalg.det(transfer) > 0).astype(float)


def tune(source):
    rng = np.random.default_rng(4101)
    source_pairs, source_labels = [], []
    for family in ("similarity", "shear", "mirror"):
        changes = interventions(family, 32, rng)
        source_pairs.append(pairs(source, changes))
        source_labels.append(truth(source, changes))
    x, d = (np.concatenate(a) for a in zip(*source_pairs))
    target = np.concatenate(source_labels)
    best = {}
    for method in ("ridge", "bounded", "orthogonal_bounded"):
        candidates = []
        for penalty in PENALTIES:
            for cap in ([None] if method == "ridge" else CAPS):
                error, _ = score(fit(x, d, method, penalty, cap), target)
                candidates.append((float(error.mean()), penalty, cap))
        optimum = min(loss for loss, _, _ in candidates)
        # Tie-break only numerical equality in degrees, then prefer the smallest penalty/cap.
        tied = [(penalty, cap) for loss, penalty, cap in candidates if abs(loss - optimum) < 1e-10]
        penalty, cap = min(tied)
        best[method] = dict(penalty=penalty, cap=cap, source_angle_deg=optimum,
                            equivalent_grid_points=[list(pair) for pair in tied])
    return best


def task_summary(errors, retained):
    """Paired task means. Relabel draws and views of a tape are dependent."""
    tasks = sorted({r.task for r in retained})
    task_rows = np.stack([errors[[r.task == task for r in retained]].mean(axis=0) for task in tasks])
    indices = np.random.default_rng(4137).integers(len(tasks), size=(10000, len(tasks)))
    deltas = {}
    methods = {name: i for i, name in enumerate(METHODS)}
    for candidate, control in (("bounded", "identity"), ("bounded", "procrustes"), ("bounded", "ridge"),
                               ("orthogonal", "procrustes"), ("orthogonal_bounded", "bounded"),
                               ("orthogonal_bounded", "ridge"), ("orthogonal_bounded", "orthogonal")):
        delta = task_rows[:, methods[candidate]] - task_rows[:, methods[control]]
        lo, hi = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        deltas[f"{candidate}_minus_{control}"] = [delta.mean(), lo, hi]
    return dict(n_supports=len(retained), angle_deg=task_rows.mean(axis=0),
                task_angles_deg=dict(zip(tasks, task_rows)), paired_delta_mean_lo_hi_deg=deltas)


def evaluate(traces, best):
    rng = np.random.default_rng(4199)
    out = {}
    for family in ("recorded", "shear", "mirror", "mirror_shear"):
        changes = interventions(family, 1 if family == "recorded" else 8, rng)
        x, d = pairs(traces, changes)
        predictions = []
        for method in METHODS:
            cfg = best.get(method, {})
            predictions.append(fit(x, d, method, cfg.get("penalty", 0.), cfg.get("cap")))
        # Show the full source cap grid at the locked penalty, including source-equivalent caps.
        # This diagnostic never selects a setting on the target family.
        bounded_sweep = [fit(x, d, "bounded", best["bounded"]["penalty"], cap) for cap in CAPS]
        # Labels first accessed after *every* support-only prediction has been made.
        label = truth(traces, changes)
        scored = [score(prediction, label) for prediction in predictions]
        angle, parity = (np.stack(v, axis=-1).reshape(len(traces), len(changes), len(METHODS)).mean(axis=1)
                         for v in zip(*scored))
        summary = task_summary(angle, traces)
        task_names = sorted({r.task for r in traces})
        cap_sensitivity = {}
        for cap, prediction in zip(CAPS, bounded_sweep):
            error = score(prediction, label)[0].reshape(len(traces), len(changes)).mean(axis=1)
            cap_sensitivity[str(cap)] = np.mean([error[[r.task == task for r in traces]].mean()
                                                  for task in task_names])
        summary["bounded_angle_deg_by_cap_at_source_penalty"] = cap_sensitivity
        summary["parity_correct_fraction_by_support"] = parity.mean(axis=0)
        summary["command_spread_quartiles"] = np.quantile(command_spread(x, np.ones(x.shape[:2])), [.25, .5, .75])
        summary["panels"] = {}
        for panel in sorted({r.panel for r in traces}):
            keep = np.array([r.panel == panel for r in traces])
            stats = task_summary(angle[keep], [r for r in traces if r.panel == panel])
            summary["panels"][panel] = {k: stats[k] for k in ("n_supports", "angle_deg")}
        out[family] = summary
    return out


def analytic_check():
    # One independent, well-excited, noiseless fixture validates matrix order, score and parity.
    x = np.array([[[.5, 0], [0, .5], [.2, .1]]])
    for op in (rot2_batch(59) @ np.array([[1., .5], [0., 1.]]), rot2_batch(64) @ np.diag([1., -1.])):
        d = x @ op.T
        np.testing.assert_allclose(score(fit(x, d, "ridge"), op)[0], 0., atol=1e-8)
        np.testing.assert_allclose(score(fit(x, d, "orthogonal_bounded", 0., 2.5), op)[0], 0., atol=1e-8)
        assert np.linalg.det(fit(x, d, "bounded", 0., .9)).item() > 0


def compact(value):
    if isinstance(value, dict):
        return {k: compact(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, list)):
        return [compact(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    return value


def main():
    analytic_check()
    source, source_files = source_traces()
    best = tune(source)
    fk, visual, target_files = target_traces()
    assert {r.task for r in source}.isdisjoint(r.task for r in fk)
    results = dict(input_revision=PIN, numpy=np.__version__, gpu_hours=0,
                   source_tasks=[r.task for r in source], methods=list(METHODS), source_selection=best,
                   input_sha256={str(f.relative_to(OUTPUTS.parent)): hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in [*source_files, *target_files]},
                   trusted_fk=evaluate(fk, best), upright_brightness_flow=evaluate(visual, best))
    (HERE / "metrics.json").write_text(json.dumps(compact(results), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
