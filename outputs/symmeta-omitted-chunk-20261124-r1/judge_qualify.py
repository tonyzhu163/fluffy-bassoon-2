"""CPU-only known-answer tests and an independent real judge-fit tape check; no eval labels."""
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import platform

import cv2
import numpy as np

import judge as j
import score


def toy(task, n, truth):
    count = n * len(task)
    rng = np.random.default_rng(723)
    angle = rng.uniform(-180., 180., size=(count, 1)) + np.array([0., 110., 230.])
    cmd = np.stack((np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))), axis=-1)
    geom_wins = np.arange(count) % 2 == 0
    yaw = np.asarray(truth) + rng.uniform(-179., 179., count)
    ident = np.array([f"{task[0]}-{i}" for i in range(count)])
    full = j.wrap(yaw[:, None] + np.c_[-40. * geom_wins, 40. * ~geom_wins])
    return dict(task=np.repeat(task, n), physical_trace_id=ident, case_id=ident,
                arm=np.where(geom_wins, "Panda", "Sawyer"), seed=np.resize([41, 42, 43], count),
                valid=np.ones((count, 3), bool), cmd=cmd, true_yaw=yaw,
                flow=j.rotate(cmd, yaw[:, None] - 31.), full_yaw=full,
                loo_yaw=np.repeat(full[:, None, :], 3, axis=1), full_valid=np.ones_like(full, bool),
                loo_valid=np.ones((count, 3, 2), bool), tuned_gain=np.linspace(.7, 1.3, count),
                tuned_sign=np.where(geom_wins, 1, -1), true_gain=np.ones(count),
                true_sign=np.where(geom_wins, 1, -1))


def known_answer():
    f = toy(["f0", "f1", "f2"], 6, 3.)
    c = toy(["c0", "c1", "c2"], 4, 5.)
    v = toy(["v0", "v1", "v2"], 8, -72.)
    parts = dict(fit=f, cal=c, eval=v)
    j.assert_disjoint(parts)
    terms = j.fit_terms(f)
    w = j.projection(terms, np.ones(3))
    np.testing.assert_allclose(w, j.rotate(np.eye(2), 31.), rtol=0., atol=1e-12)
    good = np.arange(len(v["task"])) % 2 == 0
    e, d = j.residuals(v, w)
    np.testing.assert_allclose(d, np.where(good, 40., -40.), atol=1e-10)
    # Independently derive all 3x2 angular errors via complex arguments.
    cmd = v["cmd"].view(np.complex128).squeeze(-1)
    img = np.ascontiguousarray(v["flow"] @ w).view(np.complex128).squeeze(-1)
    predicted = cmd[..., None] * np.exp(1j * np.deg2rad(v["loo_yaw"]))
    reference = np.abs(np.angle(predicted / img[..., None], deg=True))
    np.testing.assert_allclose(e, reference, atol=1e-10, rtol=0.)
    _, dc = j.residuals(c, w)
    gate = j.calibrate(j.cal_cache(c), dc, j.equal_units(c)[2])
    p = j.decide(v, d, gate)
    assert gate["b"] > 0. and np.max(np.abs(j.wrap(p["adaptive"] - v["true_yaw"]))) < 1e-10
    np.testing.assert_array_equal(p["tuned_gain"], v["tuned_gain"])
    np.testing.assert_array_equal(p["tuned_sign"], v["tuned_sign"])
    probes = j.diagnostics(v, w, gate, e, d, p)
    for k in ("rotated_command", "next_chunk_pixels", "ignored_flow", "ignored_command"):
        assert probes[k]["emitted_yaw"]["n_changed_gt_1e_10"] > 0
    assert probes["ignored_flow"]["advantage"]["n_validity_changed"] == len(d)
    # Mutate just one omitted observation, checking other two residuals remain untouched.
    wrong = v["cmd"].copy()
    wrong[:, 1] = j.rotate(wrong[:, 1], 90.)
    e1, _ = j.residuals({**v, "cmd": wrong}, w)
    np.testing.assert_array_equal(e1[:, [0, 2]], e[:, [0, 2]])
    assert np.max(np.abs(e[:, 1] - e1[:, 1])) > 1.
    valid = v["loo_valid"].copy()
    valid[0, 2, 1] = False
    ee, dd = j.residuals({**v, "loo_valid": valid}, w)
    invalid = j.decide(v, dd, gate)
    assert np.isnan(ee[0, 2, 1]) and np.isnan(dd[0]) and invalid["weight"][0] == gate["c"]
    np.testing.assert_array_equal(ee[1:], e[1:])
    assert invalid["hard"][0] == gate["hard_default"]
    # Antipodal full yaws and missing directions go to tuned; +179/-179 interpolate via ±180.
    restricted = {**v, "full_yaw": v["full_yaw"].copy(), "full_valid": v["full_valid"].copy()}
    restricted["full_yaw"][:3] = [[0., 180.], [179., -179.], [4., 50.]]
    restricted["full_valid"][2, 1] = False
    pr = j.decide(restricted, d, gate)
    np.testing.assert_array_equal(pr["adaptive"][[0, 2]], [0., 4.])
    assert pr["midpoint"][1] == -180.
    c0 = dict(gate, b=0.)
    np.testing.assert_array_equal(j.decide(v, d, c0)["adaptive"], p["constant"])
    # A completely uninformative held-out flow should select the best constant exactly.
    gate0 = j.calibrate(j.cal_cache(c), np.full_like(dc, np.nan), j.equal_units(c)[2])
    assert gate0["b"] == 0. and gate0["s"] == 0.
    synthetic = score.statistic(parts, {**p, "advantage": d}, terms, dc, v, gate, draws=32)
    assert synthetic["joint_refit"]["mae_adaptive"]["ci95"][1] < 1e-10
    assert abs(synthetic["joint_refit"]["hard_concordance"]["mean"] - 1.) < 1e-10
    assert abs(synthetic["joint_refit"][score.HARD_DELTA]["mean"] - .5) < 1e-10
    assert synthetic["registered_advancement_supported"]
    return dict(gate=gate, matrix=w.tolist(), n=len(d), independent_fold_error_checks=int(e.size),
                probes=probes, toy_bootstrap_draws=32, toy_mae=synthetic["joint_refit"]["mae_adaptive"],
                toy_concordance_delta=synthetic["joint_refit"][score.HARD_DELTA])


def real_fit_only(folder):
    manifest = folder / "archive.json"
    sha = j.digest(manifest)
    assert sha == "65c079f1051e347d781d950643113be9d4de6b92e952e373503d78d0eeb3b82e"
    rows = [r for r in json.loads(manifest.read_bytes())["trials"] if r["split"] == "train"]
    flow, cmd, yaw, task, ids, arms, pair_shared, ignored = [], [], [], [], [], [], [], []
    valid_rows = []
    for r in rows:
        assert r["arm"] in ("Panda", "Sawyer") and r["passive_controls"] == 15
        t = f"{r['suite']}/{r['task_id']}"
        assert t in j.EXPECTED["fit"] and t not in j.EXPECTED["cal"] + j.EXPECTED["eval"]
        path = folder / r["passive_path"]
        with np.load(path, allow_pickle=False) as tape:
            a, b, v = (tape[k] for k in ("img_before", "img_after", "valid"))
            xy = tape["issued_mean"][:, :2].astype(float)
            assert a.shape == b.shape == (3, 64, 64, 3) and xy.shape == (3, 2) and v.all()
            flow.append(j.dense_flow_image_motion(a, b, v))
            ignored.append(j.dense_flow_image_motion(a, a, v))
            pair_shared.append(np.array_equal(a[1:], b[:-1]))
            cmd.append(xy)
            valid_rows.append(v)
        task.append(t)
        yaw.append(r["yaw_deg"])
        ids.append(j.digest(path))
        arms.append(r["arm"])
    fit = dict(task=np.array(task), arm=np.array(arms), physical_trace_id=np.array(ids), case_id=np.array(ids),
               flow=np.array(flow), cmd=np.array(cmd), valid=np.array(valid_rows), true_yaw=np.array(yaw))
    assert len(set(ids)) == len(rows) == 224 and set(task) == set(j.EXPECTED["fit"])
    assert max(np.abs(ignored).ravel()) == 0.
    terms = j.fit_terms(fit)
    w = j.projection(terms, np.ones(8))
    # Cross-check against unaggregated least squares. Every observed group/case is equally sized.
    x = j.unit(fit["flow"])[0]
    y = j.rotate(j.unit(fit["cmd"])[0], fit["true_yaw"][:, None])
    valid = terms[2]
    weight = j.equal_units(fit)[2][:, None] / np.maximum(1, valid.sum(axis=1))[:, None]
    sw = np.sqrt(np.broadcast_to(weight, valid.shape)[valid, None])
    direct, _, rank, _ = np.linalg.lstsq(sw * x[valid], sw * y[valid], rcond=None)
    np.testing.assert_allclose(w, direct, atol=1e-12, rtol=0.)
    assert rank == 2
    return dict(archive_sha256=sha, ordered_tapes_sha256=j.hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                fit_physical_traces=len(rows), fit_tasks=sorted(set(task)), by_arm=dict(Counter(arms)),
                recorded_chunks=int(valid.size), valid_fit_chunks=int(valid.sum()), ols_rank=rank.item() if hasattr(rank,"item") else rank,
                max_ols_difference=float(np.abs(w-direct).max()), matrix=w.tolist(),
                before_before_nonzero=int(np.count_nonzero(ignored)), traces_sharing_adjacent_boundary=sum(pair_shared),
                note="Only archived judge-fit tapes opened. No candidate or calibration/evaluation outcomes.")


def main():
    if not __debug__:
        raise RuntimeError("Qualification requires Python assertions enabled")
    j.require(platform.system() == "Linux" and os.getenv("CUDA_VISIBLE_DEVICES") == "" and
              os.getenv("OPENBLAS_NUM_THREADS") == "1", "Use a single Linux CPU thread")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    root = Path("/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-ur5e-sysid-20261120-r1/gpu02col/archive")
    result = dict(host=platform.node(), platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, opencv=cv2.__version__, opencv_threads=cv2.getNumThreads(),
                  pooling_module_sha256=j.digest(Path(inspect.getfile(j.dense_flow_image_motion))),
                  fixture=known_answer(), source_fit_qualification=real_fit_only(root))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
