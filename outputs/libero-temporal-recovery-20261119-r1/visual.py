"""Exploratory terminal-image and image-history readouts on the same physical attempts."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import read
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"libero-witness-encoders-20261119-r1"))
import probe


def encode(frames, model):
    selected=frames[np.asarray(probe.CONTROLS)//25].reshape(10,96,96,3)
    torch=probe.torch
    with torch.inference_mode():
        cells=[]
        for batch in torch.from_numpy(selected).split(4):
            images=probe.F.interpolate(batch.permute(0,3,1,2).float(),size=(224,224),mode="bilinear",
                                       align_corners=False,antialias=True).div_(127.5).sub_(1).to(torch.bfloat16)
            patches=model(pixel_values=images).last_hidden_state
            grid=patches.transpose(1,2).reshape(len(batch),1152,16,16)
            cells.append(probe.F.adaptive_avg_pool2d(grid.float(),(2,2)))
        result=torch.cat(cells).reshape(5,2,-1).numpy().astype(np.float64)
    return {"last":result[-1].ravel(),"history":np.diff(result,axis=0).ravel()}


def evaluate(features, report):
    cases=report["cases"]
    rows=[]
    for test in cases:
        p,q,g,f=[test[k] for k in ("support_pose","query_pose","goal","fault")]
        other=next(o for o in read.FAULTS if o!=f)
        fit=[c for c in cases if c["goal"]==g and {p,q}.isdisjoint((c["support_pose"],c["query_pose"]))]
        assert len(fit)==6
        labels=np.stack([c["query"]["hit"] for c in fit])
        is_second=np.array([c["fault"]==read.FAULTS[1] for c in fit])
        row=dict(p=p,g=g,f=f,strict=next(b["strict"] for b in report["blocks"] if b["support_pose"]==p and b["goal"]==g))
        for mode in ("last","history"):
            x=np.stack([features[c["support_pose"],g,c["fault"]][mode] for c in fit])
            centers=[x[is_second==v].mean(axis=0) for v in (False,True)]
            choices=[int(labels[is_second==v].mean(axis=0).argmax()) for v in (False,True)]
            for tag,src in (("own",f),("swapped",other)):
                obs=features[p,g,src][mode]
                nearest=read.decision(obs,x,labels)
                condition=int(np.argmin([np.linalg.norm(obs-c) for c in centers]))
                row[f"{mode}_{tag}"]=dict(direct=int(test["query"]["hit"][nearest]),
                                          fault_decoded=int((f==read.FAULTS[1])==condition),
                                          fault_then_action=int(test["query"]["hit"][choices[condition]]))
        rows.append(row)
    summary={}
    for group,chosen in (("all",rows),("strict",[r for r in rows if r["strict"]])):
        summary[group]={"n":len(chosen)}
        for mode in ("last","history"):
            for tag in ("own","swapped"):
                key=f"{mode}_{tag}"
                summary[group][key]={m:sum(r[key][m] for r in chosen) for m in ("direct","fault_decoded","fault_then_action")}
    return dict(summary=summary,cases=rows)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data",type=Path)
    ap.add_argument("report",type=Path)
    ap.add_argument("--arm",choices=("vlm","robot"),required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.pt_weights=probe.STAGED_PT/"vision.safetensors"
    args.robot_weights=probe.ROBOT/"model.safetensors"
    probe.torch.set_num_threads(4)
    probe.torch.set_num_interop_threads(1)
    model,weights=probe.load_encoder(*probe.checkpoint(args))
    features={}
    for p in read.POSES:
        for g in read.GOALS:
            for f in read.FAULTS:
                with np.load(args.data/f"p{p}-g{g}-{f}.npz") as z:
                    features[p,g,f]=encode(z["frames"],model)
        print("encoded",p,flush=True)
    result=evaluate(features,json.loads(args.report.read_text()))
    result.update(arm=args.arm,weights=weights)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
