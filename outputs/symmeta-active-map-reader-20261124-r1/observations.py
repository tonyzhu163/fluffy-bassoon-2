"""Physical support -> common public features. Oracle measurement fields stay in the audit."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "symmeta-query-direction-probe-20261124-r1"))
from design import SCHEDULES, schedule
from score import law, physical_fit, sensor_audit

IDENTITY = np.eye(2)
SCALE = .05
BASIS = np.eye(4).reshape(4,2,2)
STRATEGY = dict(passive="passive", **{"orth+":"lateral", "orth-":"lateral", "along+":"along", "along-":"along"})


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tape(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in ("issued", "fk", "time", "goal_m", "physical")}


def model_rows(cmd, prior):
    """x times row-major vec(A) plus base equals the physical source law."""
    zero = law(cmd, np.zeros((2,2)), prior)
    x = np.stack([(law(cmd, a, prior)-zero).reshape(-1) for a in BASIS], axis=-1)
    return x / SCALE, zero / SCALE


def features(issued, fk, clock, prior, residual):
    """All terms are public at the end of the 15 controls. World FK is never rotated."""
    assert issued.shape == (15,7) and fk.shape == (16,3) and clock.shape == (16,)
    np.testing.assert_allclose(np.diff(clock), .05, rtol=0, atol=1e-6)
    u = issued.reshape(3,5,7)
    cmd = u.mean(1)[:,:3]
    observed = np.diff(fk[::5,:2], axis=0) / SCALE
    x, base = model_rows(cmd, prior)
    y = (observed - base).reshape(-1)
    k = residual / SCALE**2
    gram = x.T @ x
    rhs = x.T @ y
    common = np.r_[gram.ravel(), rhs, prior[:,:2].ravel()/SCALE, np.sqrt(k)]
    # Include all 15 controls and intra-chunk causal encoder-derived motion.
    # Each token corresponds to five consecutive controls. Global factors summarize the completed support.
    tokens = np.column_stack((u.reshape(3,-1), np.diff(fk, axis=0).reshape(3,-1)/SCALE,
                              fk[0:15:5], observed, base, x.reshape(3,8),
                              (x @ IDENTITY.ravel()).reshape(3,2)+base,
                              np.broadcast_to(common, (3,len(common)))))
    assert tokens.shape == (3,100) and np.isfinite(tokens).all()
    return tokens.astype(np.float32), x, y


def ridge(x, y, residual, multiplier=1.):
    k = multiplier * residual / SCALE**2
    return np.linalg.solve(x.T@x + k*np.eye(4), x.T@y + k*IDENTITY.ravel()).reshape(2,2)


def correct(q, a):
    return np.clip(q @ np.linalg.pinv(a).T, -1, 1)


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=-1))))


def case_rows(folder, prior, residual):
    meta = json.loads((folder/"case.json").read_text())
    if not meta["complete"]:
        raise ValueError(f"Incomplete {folder}")
    with np.load(folder/"archive-controls.npz") as z:
        query = z["query"][:,:2].astype(float)
        base = z["support_chunk_means"].astype(float)
        archived_xyz = z["archived_query_xyz"]
    prefix_n = meta.get("prefix", meta.get("midpoint_start"))
    with np.load(folder/"native-prefix.npz") as z:
        m = len(z["issued"])
        np.testing.assert_array_equal(z["issued"][:,:2], query[:m].astype(np.float32))
        prefix_drift = rms(z["public_xyz"][1:]-archived_xyz[:m])*1000 if m and len(archived_xyz) else None
    rows = []
    for name, draw in meta["maps"].items():
        a = np.asarray(draw["A"])
        assert np.linalg.norm(a, np.inf) <= 1+1e-12 and abs(np.linalg.det(a)) > 0
        held_path = folder/f"{name}-held.npz"
        hc, hd = np.empty((0,3)), np.empty((0,2))
        if held_path.exists():
            h = load_tape(held_path)
            assert sensor_audit(h,a) < 1e-8
            n = len(h["issued"])
            np.testing.assert_array_equal(h["issued"][:,:2], query[prefix_n:prefix_n+n].astype(np.float32))
            end = n//5*5
            hc = h["issued"][:end,:3].astype(float).reshape(-1,5,3).mean(1)
            hd = np.diff(h["fk"][:end+1:5,:2], axis=0)[1:]
        for mode in SCHEDULES:
            z = load_tape(folder/f"{name}-{mode}.npz")
            u = z["issued"].astype(float)
            assert sensor_audit(z,a) < 1e-8
            regenerated = np.repeat(list(schedule(base,mode,meta["source_amplitude"])), 5, axis=0)
            np.testing.assert_array_equal(u, regenerated.astype(np.float32))
            tok, x, y = features(u, z["fk"], z["time"], prior, residual)
            mean = u.reshape(3,5,7).mean(1)[:,:3]
            ref = physical_fit(mean, np.diff(z["fk"][::5,:2],axis=0), prior, residual)
            np.testing.assert_allclose(ref, ridge(x,y,residual), atol=5e-12, rtol=0)
            np.testing.assert_allclose((x@a.ravel()).reshape(3,2)*SCALE + law(mean,np.zeros((2,2)),prior),
                                       law(mean,a,prior), atol=5e-15)
            ideal = np.linalg.lstsq(mean[:,:2],z["goal_m"].reshape(3,5,3).mean(1)[:,:2]/SCALE,rcond=None)[0].T
            s = np.linalg.svd(mean[:,:2],compute_uv=False)
            rank_ratio = s[1]/s[0] if s[0] else 0.
            rows.append(dict(task=f'{meta["suite"]}/{meta["task"]}', arm=meta["arm"],
                             split=meta.get("split","developmental"), family=name, mode=mode, strategy=STRATEGY[mode],
                             tok=tok,x=x,y=y,issued=u,fk=z["fk"],clock=z["time"],baseline=base,
                             amplitude=meta["source_amplitude"],a=a,q=query,hc=hc,hd=hd,ideal=ideal,
                             prefix_drift=prefix_drift,rank_ratio=rank_ratio, support_id=str(folder),
                             effort=float(np.square(u[:,:2]).sum()), bound_hits=int((abs(u[:,:2])==1).sum())))
    return rows


def metrics(row, estimate, prior):
    q, a = row["q"], row["a"]
    pred, exact = correct(q,estimate), correct(q,a)
    motion = None
    if len(row["hd"]):
        motion = rms(law(row["hc"], estimate, prior)[1:]-row["hd"])*1000
    return dict(command=50*rms(pred-exact), motion=motion, map=float(np.linalg.norm(estimate-a)),
                setpoint=50*rms((pred-exact)@a.T), box=50*rms(exact@a.T-q),
                support=1000*SCALE*rms((row["x"]@estimate.ravel()-row["y"]).reshape(3,2)))


def family_group(name):
    return name.rstrip("+-")


def grouped(rows, metric, keys):
    """Average signed branches, map draws, seeds, bodies and tasks with equal nesting."""
    grouped_data = {}
    for r in rows:
        value = r[metric]
        if value is None:
            continue
        key = tuple(r[k] for k in keys)
        grouped_data.setdefault(key, []).append(value)
    return {k:float(np.mean(v)) for k,v in grouped_data.items()}


def mean_report(rows, metric):
    # Full factorial. Every admitted arm-task has the same numbers of draws and signs.
    values = grouped(rows, metric, ("task","arm","family_group","strategy"))
    return float(np.mean(list(values.values()))) if values else None


def interval(values):
    x = np.asarray(values)
    resamples = np.asarray(list(itertools.product(range(len(x)),repeat=len(x))))
    return np.quantile(x[resamples].mean(1), [.025,.975]).tolist()


def train_views(row, prior, residual, rng):
    """Re-express a *real* trajectory. Admit only actual same-schedule, nonclipped transformations."""
    base, mode, a = row["baseline"], row["mode"], row["a"]
    # The eight signed permutations preserve box/headroom exactly. Continuous O(2)
    # enriches feasible unsaturated orientations without changing the physical world.
    angles = np.r_[np.tile(np.arange(4)*np.pi/2,2),rng.uniform(-np.pi,np.pi,32)]
    signs = np.r_[np.repeat([1.,-1.],4),rng.choice([-1.,1.],32)]
    for theta, sign in zip(angles, signs):
        c,s = np.cos(theta),np.sin(theta)
        rot = np.array([[c,-s*sign],[s,c*sign]])
        transform = a @ rot.T
        if np.linalg.norm(transform,np.inf) > 1+1e-12:
            continue
        us = row["issued"].copy()
        us[:,:2] = us[:,:2] @ rot.T
        base_new = base.copy()
        base_new[:,:2] = base_new[:,:2] @ rot.T
        if max(abs(base_new).max(), abs(us).max()) > 1+1e-14:
            continue
        named = mode
        if mode.startswith("orth") and sign < 0:
            named = "orth-" if mode.endswith("+") else "orth+"
        actual = np.repeat(list(schedule(base_new,named,row["amplitude"])),5,axis=0)
        # The genuine controls were issued through float32. Tolerance only covers that cast.
        if abs(actual-us).max() > 8e-8:
            continue
        np.testing.assert_allclose(us[:,:2]@transform.T, row["issued"][:,:2]@a.T, atol=1e-14,rtol=0)
        tok,x,y = features(us,row["fk"],row["clock"],prior,residual)
        yield tok, x, y, transform
