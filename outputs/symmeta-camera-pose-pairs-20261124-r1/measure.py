"""Geometry-only checks plus original DIS/lumaflow maps. All comparisons use shared support."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from geometry import as_h, dump, load_view, project


def grid_sample(image, homography):
    y, x = np.mgrid[:64,:64]
    xy = project(homography, np.stack([x.ravel(),y.ravel()], axis=1)).reshape(64,64,2)
    # Remove numerical noise at the edge of the identity transform.
    xy[np.abs(xy)<1e-10]=0
    xy[np.abs(xy-63)<1e-10]=63
    mask = ((xy >= 0) & (xy <= 63)).all(axis=-1)
    sample = cv2.remap(image, xy.astype(np.float32), None, cv2.INTER_LINEAR)
    return sample, mask


def measure(row, diag, maps):
    from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
    from fluffy_bassoon.xemb.equiv_v2 import _vis_inputs_v2
    ref = row["views"]["nominal"]
    xyz = np.asarray(diag["world_tcp_m"])
    ref_px=project(ref["P"], xyz)
    t=load_view(row,"nominal")
    rgb_ref=np.concatenate([t["frame0_img"][None],t["img_after"]])
    scores=[]
    for name,v in row["views"].items():
        t=load_view(row,name)
        flow=compute_dis_flow_maps(t["img_before"],t["img_after"],t["valid"])
        batch={k:t[k][None] for k in ("img_before","img_after","valid")}
        mp=np.asarray(_vis_inputs_v2(dict(batch,dis_flow=flow[None]),"dis_lumaflow")[0])
        assert mp.shape==(3,64,64,9) and np.isfinite(mp).all()
        path=maps/row["id"]/f"{name}.npz"
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,mp=mp,dis_flow=flow)
        v["motion_npz"]=str(path)
        observed=np.concatenate([t["frame0_img"][None],t["img_after"]])
        obs_px=project(v["P"],xyz)
        if not ((obs_px>=0)&(obs_px<=63)).all():
            raise ValueError(f"TCP outside {row['id']} {name}")
        opp_name=name if name=="nominal" else name[:-1]+{"+":"-","-":"+"}[name[-1]]
        Hs={"identity":np.eye(3),"correct":np.array(v["H_ref_to_view"]),
            "opposite":np.array(row["views"][opp_name]["H_ref_to_view"])}
        values={k:dict(tcp_error_px=[],rgb_mae=[]) for k in Hs}
        overlap=[]
        for k,H in Hs.items():
            values[k]["tcp_error_px"]=np.linalg.norm(project(as_h(np.linalg.inv(H)),obs_px)-ref_px,axis=-1).tolist()
        for image,reference in zip(observed,rgb_ref,strict=True):
            samples={k:grid_sample(image,H) for k,H in Hs.items()}
            common=np.logical_and.reduce([m for _,m in samples.values()])
            assert np.any(common)
            overlap.append(float(common.mean()))
            for k,(s,_) in samples.items():
                values[k]["rgb_mae"].append(float(np.abs(s[common].astype(float)-reference[common]).mean()))
                if name=="nominal":
                    np.testing.assert_array_equal(s, reference)
        if name=="nominal":
            np.testing.assert_allclose(v["H_ref_to_view"],np.eye(3),atol=1e-10)
        scores.append(dict(id=row["id"],arm=row["arm"],shard=row["shard"],view=name,overlap=overlap,
                           reference_px=ref_px.tolist(),observed_px=obs_px.tolist(),scores=values))
    return scores


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("folders",type=Path,nargs="+")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    rows, diagnostics, results=[],[],[]
    for f in args.folders:
        bank=json.loads((f/"bank.json").read_text())["rows"]
        ds=json.loads((f/"diagnostics.json").read_text())
        assert [r["id"] for r in bank]==[d["id"] for d in ds]
        for r,d in zip(bank,ds,strict=True):
            results.extend(measure(r,d,args.out/"motion"))
        rows.extend(bank)
        diagnostics.extend(ds)
    assert len({r["id"] for r in rows})==len(rows)
    summary={}
    gen=np.random.default_rng(1124)
    for v in rows[0]["views"]:
        items=[r for r in results if r["arm"]=="UR5e" and r["view"]==v]
        if not items:
            continue
        groups=sorted({r["shard"] for r in items})
        stats={}
        for metric in ("tcp_error_px","rgb_mae"):
            start=int(metric=="tcp_error_px")
            task={}
            for k in ("identity","correct","opposite"):
                task[k]=np.array([np.mean([r["scores"][k][metric][start:] for r in items if r["shard"]==g]) for g in groups])
            for k in ("identity","opposite"):
                task["correct_minus_"+k]=task["correct"]-task[k]
            picks=gen.integers(len(groups),size=(10000,len(groups)))
            stats[metric]={k:dict(mean=float(a.mean()),per_task=a.tolist(),range=[float(a.min()),float(a.max())],
                                  ci95=np.quantile(a[picks].mean(axis=1),[.025,.975]).tolist()) for k,a in task.items()}
        summary[v]=dict(n=len(items),shards=groups,mean_overlap=float(np.mean([r["overlap"] for r in items])),**stats)
    dump(args.out/"bank.json",dict(rows=rows))
    dump(args.out/"diagnostics.json",diagnostics)
    dump(args.out/"scores.json",results)
    dump(args.out/"summary.json",summary)


if __name__=="__main__":
    main()
