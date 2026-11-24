"""Join a blinded, frozen release to real support tapes and official query banks on one CPU."""
import argparse
from functools import lru_cache
import io
import os
from pathlib import Path
import re
import sys

import numpy as np
from cohort import (HERE, RELEASE_PIN, MANIFEST_SHA, assignments, atomic, bounded_issue, checked, digest, encode, official_banks, parse, require)

sys.path[:0] = [str(HERE.parents[1]/"src"), str(HERE.parent/"symmeta-active-map-reader-20261124-r1")]
from observations import correct, features, ridge
from design import schedule
from fluffy_bassoon.xemb.wider_family import interface_matrix, sample_interfaces


@lru_cache(maxsize=512)
def read(path, sha256):
    path = Path(path)
    require(path.is_absolute() and re.fullmatch("[0-9a-f]{64}",sha256), "archive reference")
    return checked(path, sha256)


def unpack(ref):
    return read(ref["path"],ref["sha256"])


def expected_draw(seed, family):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        for a in interface_matrix(sample_interfaces(rng,"general",64)):
            if np.linalg.norm(a,np.inf)<=1 and (np.linalg.det(a)>0)==(family=="general+"):
                return a
    raise ValueError("registered bounded draw missing")


def inverse_convention(fixtures):
    dtypes, singular, reflects = set(), False, False
    for f in fixtures:
        dtype = f.get("dtype","float64")
        require(dtype in ("float32","float64"), "operator dtype")
        a = np.array(f["A"],dtype=dtype)
        q = np.array(f["q"],np.float64)
        require(a.shape==(2,2) and q.shape[1]==2 and np.isfinite(q).all() and np.abs(q).max()<=1, "reference fixture")
        upstream = np.asarray(f["issued"],np.float64)
        # Compare the exported evaluation with independently run mapreader code in the native deployment.
        np.testing.assert_allclose(correct(q,a),upstream,rtol=8*np.finfo(a.dtype).eps,atol=8*np.finfo(a.dtype).eps)
        wire = np.zeros((len(q),7),np.float32)
        wire[:,:2] = q
        expected = correct(wire[:,:2].astype(float),a).astype(np.float32)
        np.testing.assert_array_equal(bounded_issue(wire,a)[:,:2],expected)
        dtypes.add(dtype)
        singular |= np.linalg.matrix_rank(a)==1
        reflects |= (np.linalg.det(a)<0 and not np.allclose(a,a.T) and np.any(np.abs(upstream)==1))
    require(dtypes=={"float32","float64"} and singular and reflects, "rank/reflection/bounds/dtype convention coverage")


def map_metadata(r, row, ref, archives):
    meta = parse(unpack(row["case"]))
    key = r["suite"]+"/"+str(r["task"])
    require(Path(row["case"]["path"]).name=="case.json" and meta["complete"] and meta["split"]=="target" and
            (meta["arm"],meta["suite"],meta["task"])==(r["arm"],r["suite"],r["task"]), "physical case identity")
    source = meta["source_record"]
    require(source==archives[r["arm"]+"/"+key]["support"] and source["start"]==11 and
            (source["arm"],source["task"])==(r["arm"],key) and
            row["source_archive"]=={k:source[k] for k in ("path","sha256")}, "layout-11 source archive join")
    require(meta["support_layout"]==17 and meta["support_seed"]==24112600+r["task_index"], "calibration layout and seed")
    sr = meta["support_reset"]
    require(sr["source_pin"]==RELEASE_PIN and sr["manifest_sha256"]==MANIFEST_SHA and sr["perturbation_plan"]["case_id"]=="BASE" and
            sr["suite"]==r["suite"] and sr["task_id"]==r["task"] and sr["seed"]==meta["support_seed"] and
            sr["init_state_id"]==17 and sr["released_row_fingerprint"]==ref["support_row_fingerprint"] and
            sr["bddl_sha256"]==ref["bddl_sha256"] and meta["model"]["arm"]==r["arm"], "actual physical support reset")
    draw = meta["maps"][r["family"]]
    a = np.array(draw["A"],np.float64)
    require(draw["seed"]==r["map_seed"] and np.linalg.norm(a,np.inf)<=1, "physical draw seed or bounds")
    # Different NumPy/BLAS versions may round the trig and matrix products in generation at float64 precision.
    np.testing.assert_allclose(a,expected_draw(r["map_seed"],r["family"]),atol=2e-14,rtol=0)
    return meta, a


def verified_support(row, meta, a, mode, law, release):
    directory = Path(row["case"]["path"]).parent
    require(Path(row["support_tape"]["path"])==directory/f'{row["family"]}-{mode}.npz', "support tape join")
    require(row["source_law_sha256"]==release["source_law"]["sha256"] and
            row["checkpoint_sha256"]==release["checkpoint41"]["sha256"], "source fit/reader identity")
    with np.load(io.BytesIO(unpack(row["source_archive"])),allow_pickle=False) as z:
        baseline = z["issued_mean"][z["valid"]].astype(float)
    require(baseline.shape==(3,7), "source proposals")
    with np.load(io.BytesIO(unpack(row["support_tape"])),allow_pickle=False) as z:
        u, fk, time = (z[k].astype(float) for k in ("issued","fk","time"))
        regenerated = np.repeat(list(schedule(baseline,mode,meta["source_amplitude"])),5,axis=0)
        np.testing.assert_array_equal(u,regenerated.astype(np.float32))
        physical = z["issued"].copy()
        for i,command in enumerate(physical.copy()):
            physical[i,:2] = a @ command[:2]
        np.testing.assert_array_equal(z["physical"],physical)
        np.testing.assert_allclose(z["goal_m"],.05*physical[:,:3].astype(float),atol=1e-8,rtol=0)
        require(z["q"].shape==(16,meta["model"]["joint_count"]) and np.isfinite(z["q"]).all() and
                z["terminal"].shape==(15,) and not z["terminal"][:-1].any(), "support control/encoder count")
    h = digest(u.astype("<f8").tobytes(order="C")+fk.astype("<f8").tobytes(order="C"))
    require(h==row["input_sha256"], "actual common estimator input")
    prior = np.array(law["prior"])
    residual = law["residual_power"]
    tok,x,y = features(u,fk,time,prior,residual)
    np.testing.assert_allclose(row["structured"],ridge(x,y,residual),atol=5e-12,rtol=1e-12)
    est = {}
    for name,dt in (("structured","float64"),("direct","float32")):
        dtype = row.get(name+"_dtype",dt)
        require(dtype in ("float32","float64"), "release dtype")
        arr = np.array(row[name],dtype=dtype)
        require(arr.shape==(2,2) and np.isfinite(arr).all(), "finite forward matrix")
        est[name] = dict(A=arr.tolist(),dtype=dtype)
    return est, dict(case=row["case"],source_archive=row["source_archive"],support_tape=row["support_tape"],
                     input_sha256=h, feature_sha256=digest(tok.tobytes()), schedule=mode,source_law_sha256=row["source_law_sha256"],
                     checkpoint_sha256=row["checkpoint_sha256"],support_reset=meta["support_reset"])


def recorded_inference(folder):
    """Read only provenance/forward estimates from the final released prediction archive."""
    files = folder/"fresh-files.json"
    outputs = folder/"fresh-rows.json"
    if not (files.is_file() and outputs.is_file()):
        return {}, {}, None
    file_bytes, output_bytes = files.read_bytes(), outputs.read_bytes()
    lookup = {}
    for row in parse(output_bytes):
        if (row["method"],row["seed"]) not in (("direct",41),("incumbent",0)):
            continue
        if row["family"] not in ("general+","general-"):
            continue
        suite,task = row["task"].split("/")
        key = row["arm"],suite,int(task),row["family"],row["mode"]
        model = "direct" if row["method"]=="direct" else "structured"
        entry = lookup.setdefault(key,{})
        require(model not in entry,"duplicate official fitted output")
        entry[model] = row["estimate"]
    return parse(file_bytes),lookup,dict(files_sha256=digest(file_bytes),outputs_sha256=digest(output_bytes))


def prepare(out, path, sha):
    require(os.environ["CUDA_VISIBLE_DEVICES"]=="" and len(os.sched_getaffinity(0))==1,"single CPU release preparation")
    p = parse(checked(path,sha))
    require(p["version"]==1 and re.fullmatch("[0-9a-f]{40}",p["reader_pin"]) and type(p["qualified"]) is bool,"release schema")
    official = official_banks()
    ref_file = HERE.parent/"frozen-vla-multiarm-confirmation-20261123-r1/evidence/production-native.json"
    archive_ref = parse(ref_file.read_bytes())
    m = HERE.parent/"frozen-vla-motion-models-20261123-r1/evidence/matrices.json"
    archives = parse(checked(m,archive_ref["matrix_file_sha256"]))
    require(p["source_law"]["sha256"]==archive_ref["source_fit_sha256"],"prior 192-source law")
    law = parse(unpack(p["source_law"]))
    freeze = parse(unpack(p["source_freeze"]))
    ck_sha = p["checkpoint41"]["sha256"]
    require(freeze["checkpoints"]["seed-41.pkl"]==ck_sha, "checkpoint missing from source freeze")
    unpack(p["checkpoint41"])
    fit = Path(p["source_freeze"]["path"]).parent
    qualification_error = None
    try:
        source_files = parse(checked(fit/"source-files.json",freeze["source"]))
        selected = parse(checked(fit/"selected.json",freeze["selected"]))
    except (OSError,KeyError,ValueError) as e:
        source_files, selected = {}, {}
        qualification_error = str(e)
    source_only = (bool(source_files) and all(Path(f).parent.name.startswith(("Panda-libero","Sawyer-libero"))
                   for f in source_files) and selected.get("seeds")==[41,42,43] and
                   selected.get("checkpoints",{}).get("seed-41.pkl")==ck_sha)
    scored_files, scored_maps, output_refs = recorded_inference(fit)
    qualification, reason = False, qualification_error
    if p.get("source_qualification"):
        try:
            qbytes = unpack(p["source_qualification"])
            # Current source producer seals the test report separately from checkpoint selection.
            if "test" in freeze and freeze["test"]==digest(qbytes):
                q = parse(qbytes)
                qualification = q["general-lateral/command/direct-minus-identity"]["mean"] < 0
            else:
                q = parse(qbytes)
                qualification = q.get("qualified") is True
        except (OSError,ValueError,KeyError) as e:
            reason = str(e)
    qualification = qualification and p["qualified"] and source_only
    if not qualification and not reason:
        reason = "source qualification absent, failed, or unreconciled with source selection"
    inverse_convention(p["inverse_fixture"])
    rows = {}
    for row in p["rows"]:
        k = tuple(row[field] for field in ("arm","suite","task","family","schedule","seed"))
        require(k not in rows, "duplicate released support identity")
        rows[k] = row
    require(parse((HERE/"cases.json").read_bytes())==assignments(), "registered 32-case manifest")
    shared_draws = {}
    support_initial = {}
    cases, quarantine = [], []
    for registered in assignments():
        r = registered.copy()
        ref = official[r["suite"],r["task"],r["layout"]]
        r.update(reference=ref, prompt=archive_ref["arms"][r["arm"]]["tasks"][r["suite"]+"/"+str(r["task"])]["prompt"], estimates={"raw":None}, input_receipts={})
        for strategy,mode in (("passive","passive"),("lateral","orth"+r["sign"]),("along","along"+r["sign"])):
            try:
                key = r["arm"],r["suite"],r["task"],r["family"],mode,41
                row = rows[key]
                if output_refs:
                    require(scored_maps[key[:-1]]=={m:row[m] for m in ("direct","structured")},
                            "matrix differs from frozen inference output")
                    for field in ("case","support_tape"):
                        require(scored_files[row[field]["path"]]==row[field]["sha256"],
                                "physical archive differs from the scoring-time receipt")
                meta,a = map_metadata(r,row,ref,archives)
                draw_key = (r["suite"],r["task"],r["family"])
                require(shared_draws.setdefault(draw_key,a.tolist())==a.tolist(),"physical A differs across arms/layouts")
                support_key = (r["arm"],r["suite"],r["task"])
                require(support_initial.setdefault(support_key,meta["support_reset"])==meta["support_reset"],"support reset differs")
                shared = dict(true_map=a.tolist(),support_physics=meta["support_reset"]["initial_state_fingerprint"],
                              support_model=meta["model"])
                if "true_map" in r:
                    require(all(r[k]==v for k,v in shared.items()), "different physical setup across schedules")
                r.update(shared)
                r["estimates"]["true"] = {"A":a.tolist(),"dtype":"float64"}
                estim, receipt = verified_support(row,meta,a,mode,law,p)
                r["input_receipts"][strategy] = receipt
                for name in estim:
                    r["estimates"][name+"_"+strategy] = estim[name]
            except (OSError, ValueError, KeyError, AssertionError) as e:
                quarantine.append(dict(index=r["index"],schedule=mode,error=str(e)))
        if "true_map" in r:
            cases.append(r)
    require(cases, "no technically verified native cases")
    panel = dict(version=1,registration=digest((HERE/"PLAN.md").read_bytes()),release=dict(path=str(path),sha256=sha),
                 source_qualified=qualification,unresolved_reason=reason,reader_pin=p["reader_pin"],quarantine=quarantine,
                 inverse_fixture=p["inverse_fixture"],scoring_archive=output_refs,source_file_count=len(source_files),source_refs={k:p.get(k) for k in
                     ("source_freeze","source_qualification","source_law","checkpoint41")},
                 assignments=assignments(),cases=cases)
    raw = encode(panel)
    atomic(out/"panel.json",raw)
    atomic(out/"input-receipt.json",encode(dict(panel_sha256=digest(raw),cases=len(cases),invalid=quarantine,
           eligible_episodes=sum(len(r["estimates"]) for r in cases),source_qualified=qualification,
           reader_pin=p["reader_pin"],release_sha256=sha)))
    print("sealed",digest(raw),"cases",len(cases),"quarantined schedules",len(quarantine),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("out",type=Path)
    parser.add_argument("release",type=Path)
    parser.add_argument("sha256")
    args=parser.parse_args()
    prepare(args.out,args.release,args.sha256)
