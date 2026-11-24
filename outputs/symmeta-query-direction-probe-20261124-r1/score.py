"""Paired held-command corrections and physical five-control forecasts."""
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from design import FAMILIES, SCHEDULES, TASKS, save


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=-1))))


def law(cmd, matrix, prior):
    world = cmd.copy()
    world[:,:2] = cmd[:,:2] @ matrix.T
    lag = np.vstack((np.zeros(3), world[:-1]))
    return np.column_stack((world, lag)) @ prior[:,:2]


def physical_fit(cmd, delta, prior, residual):
    base = law(cmd, np.zeros((2, 2)), prior)
    x = np.stack([(law(cmd, b.reshape(2, 2), prior) - base).ravel() for b in np.eye(4)], axis=1)
    aug = np.vstack((x, np.sqrt(residual) * np.eye(4)))
    y = np.r_[(delta - base).ravel(), np.sqrt(residual) * np.eye(2).ravel()]
    return np.linalg.lstsq(aug, y, rcond=None)[0].reshape(2, 2)


def assess(q, true, estimated):
    exact = np.clip(q @ np.linalg.inv(true).T, -1, 1)
    pred = np.clip(q @ np.linalg.pinv(estimated).T, -1, 1)
    return pred, dict(command_mm=50*rms(pred-exact), setpoint_mm=50*rms((pred-exact)@true.T),
                      desired_mm=50*rms(pred@true.T-q), oracle_box_mm=50*rms(exact@true.T-q),
                      oracle_bound_share=float(np.mean(np.any(np.abs(exact) == 1, axis=1))),
                      fitted_bound_share=float(np.mean(np.any(np.abs(pred) == 1, axis=1))))


def sensor_audit(z, a):
    np.testing.assert_array_equal(z["issued"][:,2:], z["physical"][:,2:])
    np.testing.assert_allclose(z["physical"][:,:2], z["issued"][:,:2] @ a.T, atol=8e-8, rtol=0)
    assert np.max(np.abs(z["issued"])) <= 1 and np.max(np.abs(z["physical"])) <= 1
    return float(np.max(np.abs(z["goal_m"][:,:2] - .05*z["issued"][:,:2] @ a.T)))


def read(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def case(folder, prior, residual):
    meta = json.loads((folder/"case.json").read_text())
    assert meta["complete"]
    arch = read(folder/"archive-controls.npz")
    q = arch["query"][:,:2].astype(float)
    x0 = np.repeat(arch["support_chunk_means"],5,axis=0)
    prefix = read(folder/"native-prefix.npz")
    n = len(prefix["issued"])
    np.testing.assert_array_equal(prefix["issued"], arch["query"][:n])
    drift = 1000*np.linalg.norm(prefix["public_xyz"][1:] - arch["archived_query_xyz"][:n], axis=1)
    audit = dict(midpoint=meta["midpoint_start"], native_prefix_steps=n, native_prefix_max_drift_mm=float(drift.max()),
                 native_prefix_rms_drift_mm=float(np.sqrt(np.mean(drift**2))), families={})
    result = []
    predictions = {}
    for family in FAMILIES:
        a = np.array(meta["maps"][family]["A"])
        hpath = folder/f"{family}-held.npz"
        h = read(hpath) if hpath.exists() else None
        observed = np.zeros((0,2))
        command = np.zeros((0,3))
        maxgoal = 0.
        if h is not None:
            t = len(h["issued"])
            np.testing.assert_array_equal(h["issued"], arch["query"][n:n+t])
            maxgoal = sensor_audit(h,a)
            usable = 5*(t//5)
            command = h["issued"][:usable,:3].astype(float).reshape(-1,5,3).mean(1)
            observed = np.diff(h["fk"][:usable+1:5,:2],axis=0)[1:]
        pulses = {}
        for label in ("x+", "x-", "y+", "y-", "zero"):
            d = read(folder/f"{family}-check-{label}.npz")
            maxgoal = max(maxgoal, sensor_audit(d,a))
            pulses[label] = 1000*(d["fk"][-1,:2]-d["fk"][0,:2])
        secant = np.stack([pulses[c+"+"]-pulses[c+"-"] for c in "xy"], axis=1)
        asym = np.stack([.5*(pulses[c+"+"]+pulses[c+"-"])-pulses["zero"] for c in "xy"],axis=1)
        ref_motion = law(command,a,prior)[1:] if len(command) else np.empty((0,2))
        drift_mm = 1000*rms(ref_motion-observed) if len(observed) else None
        audit["families"][family] = dict(evaluated_held_chunks=len(observed), injected_map=a.tolist(),
                pulse_endpoint_mm={k:v.tolist() for k,v in pulses.items()}, secant_mm_per_unit=secant.tolist(),
                even_residual_mm=asym.tolist(), source_law_true_map_motion_mm=drift_mm)
        if family == "native" and h is not None:
            audit["native_midpoint_max_drift_mm"] = float(np.max(1000*np.linalg.norm(h["public_xyz"][1:]-arch["archived_query_xyz"][n:n+t],axis=1)))
        traces = {s:read(folder/f"{family}-{s}.npz") for s in SCHEDULES}
        nominal = traces["passive"]
        for name, z in traces.items():
            x = z["issued"].astype(float)
            maxgoal = max(maxgoal, sensor_audit(z,a))
            assert len(x) == 15 and np.array_equal(x[:,2:],x0[:,2:])
            np.testing.assert_array_equal(z["physical"][:5], nominal["physical"][:5])
            np.testing.assert_array_equal(z["camera_steps"], [0,5,10,15])
            assert not np.array_equal(z["external"][0], z["external"][-1])
            delta = np.diff(z["fk"][::5,:2],axis=0)
            avg = x[:,:3].reshape(3,5,3).mean(1)
            fit = physical_fit(avg, delta, prior, residual)
            ideal = np.linalg.lstsq(avg[:,:2], z["goal_m"].reshape(3,5,3).mean(1)[:,:2]/.05, rcond=None)[0].T
            pred, stats = assess(q, a, fit)
            predictions[family+"_"+name+"_inverse"] = pred
            predictions[family+"_"+name+"_held_xy_m"] = law(command,fit,prior)[1:] if len(command) else np.empty((0,2))
            _, ideal_stat = assess(q, a, ideal)
            p = x[:,:2]-x0[:,:2]
            spec = np.linalg.svd(avg[:,:2],compute_uv=False)
            mean = p.mean(0)
            delta2 = np.linalg.norm(p[5])
            delta3 = np.linalg.norm(p[10])
            row = dict(arm=meta["arm"],task=meta["suite"]+"/"+str(meta["task"]),family=family,schedule=name,
                       **stats, ideal_command_mm=ideal_stat["command_mm"], true_law_motion_mm=drift_mm,
                       motion_mm=1000*rms(predictions[family+"_"+name+"_held_xy_m"]-observed) if len(observed) else None,
                       singular_ratio=float(spec[-1]/spec[0]), mean_change=mean.tolist(),
                       balanced=bool(np.linalg.norm(mean)<1e-7), amplitude2=float(delta2), amplitude3=float(delta3),
                       xy_effort=float(np.sum(x[:,:2]**2)), perturbation_effort=float(np.sum(p**2)),
                       bound_controls=int(np.any(np.abs(x[:,:2])==1,axis=1).sum()),
                       physical_effort=float(np.sum(z["physical"][:,:2].astype(float)**2)),
                       map=fit.tolist(), ideal_map=ideal.tolist(), response=delta.tolist(),
                       fk_sensor_max_mm=float(np.max(1000*np.linalg.norm(z["fk"]-z["public_xyz"],axis=1))))
            result.append(row)
        for kind in ("orth","along"):
            plus, minus = [traces[kind+s]["issued"][:,:2].astype(float) for s in "+-"]
            np.testing.assert_allclose((plus+minus)/2,x0[:,:2],atol=5e-8,rtol=0)
        for s in "+-":
            ort=p_for(traces["orth"+s],x0); along=p_for(traces["along"+s],x0)
            np.testing.assert_allclose(np.linalg.norm(ort,axis=1),np.linalg.norm(along,axis=1),atol=1e-7,rtol=0)
            np.testing.assert_allclose((ort*along).sum(axis=1),0,atol=5e-8,rtol=0)
        audit["families"][family]["max_measured_goal_error_m"]=maxgoal
    np.savez_compressed(folder/"held-predictions.npz", **predictions)
    return result, audit


def p_for(z,x0):
    return z["issued"][:,:2].astype(float)-x0[:,:2]


def interval(x):
    # Exact ordered whole-task bootstrap. Four task clusters imply only 4^4 draws.
    x=np.array(x)
    samples=np.array(list(itertools.product(range(len(x)), repeat=len(x))))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[samples].mean(1),[.025,.975]).tolist(),
                n_tasks=len(x),per_task=x.tolist())


def summarize(rows, fams, arms, balance):
    task_data={}
    for suite, tid in TASKS:
        task=f"{suite}/{tid}"
        grouped=[]
        for arm in arms:
            for fam in fams:
                case=[r for r in rows if (r["arm"],r["family"],r["task"])==(arm,fam,task)]
                assert len(case)==5
                if balance is not None and all(r["balanced"] for r in case)!=balance:
                    continue
                grouped.append(case)
        if not grouped:
            continue
        task_data[task]={}
        for k in ("command_mm","setpoint_mm","motion_mm","desired_mm","oracle_box_mm","true_law_motion_mm","ideal_command_mm","singular_ratio","xy_effort","physical_effort"):
            vals=[]
            for case in grouped:
                modes=[[r[k] for r in case if r["schedule"].startswith(p)] for p in ("passive","along","orth")]
                if all(v is not None for mode in modes for v in mode):
                    vals.append([np.mean(mode) for mode in modes])
            if vals:
                p,a,o=np.mean(vals,axis=0)
                task_data[task][k]=[p,a,o,o-p,o-a]
    if not task_data:
        return None
    out=dict(tasks=list(task_data))
    labels=("passive","along","orth","orth_minus_passive","orth_minus_along")
    for k in next(iter(task_data.values())):
        rows2=[v[k] for v in task_data.values() if k in v]
        out[k]={label:interval(col) for label,col in zip(labels,zip(*rows2))}
    return out


def main(out):
    rows=[]; audit={}
    arms=["UR5e","IIWA"]
    for arm in arms:
        runtime=json.loads((out/f"runtime-{arm}.json").read_text())
        for suite, tid in TASKS:
            folder=out/f"{arm}-{suite}-{tid}"
            cases, notes=case(folder,np.array(runtime["prior"]),runtime["residual"])
            rows.extend(cases); audit[folder.name]=notes
    save(out/"readback.json",audit)
    save(out/"scores.json",rows)
    summary={}
    for group in [*FAMILIES,"held"]:
        fams=["general","swap"] if group=="held" else [group]
        summary[group]={k:summarize(rows,fams,v,None) for k,v in [("both",arms),("UR5e",["UR5e"]),("IIWA",["IIWA"])]}
        for name,val in (("balanced",True),("imbalanced",False)):
            summary[group][name]=summarize(rows,fams,arms,val)
    save(out/"summary.json",summary)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
