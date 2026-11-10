"""Choose the two checkpoints on development data and summarize their locked confirmation."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
from readout import read

STEPS=(128,512,1024)


def outcomes(folder,arm):
    complete=read(folder/"complete.json")
    identity=read(folder/"identity.json")
    assert arm in complete["arms"] and len(identity["scenes"])==complete["scenes"]
    rows={}
    for seed in identity["scenes"]:
        for goal in (0,1):
            setting=goal if arm in ("own","donor") else None
            r=read(folder/f"{seed}-{arm}-{setting}.json")
            assert r["query"]==seed and r["goal"]==setting and r["arm"]==arm
            assert r["executed"]+r["prefix_controls"]<=500 and sum(r["placed"])<=1
            rows[seed,goal]=r
    return rows


def choose(dev,fit):
    scores={}
    for step in STEPS:
        key=f"step-{step:04d}"
        rows=read(fit/key/"offline.json")
        shared=[r for r in rows if r["matched_decision"]]
        assert shared
        panel=dev/key
        assert read(panel/"identity.json")["step"]==step
        scores[step]={}
        for arm in ("memoryless","own"):
            native=outcomes(panel,arm)
            assert len(native)==16
            scores[step][arm]={"native":sum(r["placed"][g] for (_,g),r in native.items()),
                               "masked_error":float(np.mean([r[arm] for r in shared]))}
    return {"development":scores,**{a:max(STEPS,key=lambda s:(scores[s][a]["native"],-scores[s][a]["masked_error"],-s))
                                    for a in ("memoryless","own")}}


def confirm(study,selected):
    actual={}
    for arm,part in (("own","dpt"),("donor","dpt"),("masked","dpt"),("memoryless","baseline"),("frozen","baseline")):
        folder=study/part
        expected=selected["own" if part=="dpt" else "memoryless"]
        assert read(folder/"identity.json")["step"] == expected
        actual[arm]=outcomes(folder,arm)
    keys=set(actual["own"])
    assert len(keys)==32 and all(set(x)==keys for x in actual.values())
    rows=[]
    for seed,goal in sorted(keys):
        current={a:v[seed,goal] for a,v in actual.items()}
        assert len({(r["prefix_controls"],tuple(sorted(r["decision"].items()))) for r in current.values()})==1
        row={"scene":seed,"goal":goal,"grasped":current["own"]["grasped"],"support":current["own"]["support"],
             **{a:int(r["placed"][goal]) for a,r in current.items()}}
        row["flipped"]=int(actual["own"][seed,1-goal]["placed"][goal])
        rows.append(row)
    arms=(*actual,"flipped")
    contrasts={}
    seeds=sorted({r["scene"] for r in rows})
    for arm in ("donor","flipped","masked","memoryless","frozen"):
        diff=[r["own"]-r[arm] for r in rows]
        groups=[sum(r["own"]-r[arm] for r in rows if r["scene"] in seeds[i:i+4]) for i in range(0,len(seeds),4)]
        boot=[100*sum(draw)/len(rows) for draw in product(groups,repeat=len(groups))]
        contrasts[arm]={"wins":diff.count(1),"losses":diff.count(-1),"ties":diff.count(0),
                        "delta_pp":100*sum(diff)/len(rows),"four_scene_block_95_pp":np.quantile(boot,[.025,.975]).tolist()}
    return {"goal_cases":len(rows),"grasped_goal_cases":sum(r["grasped"] for r in rows),
            "success":{a:sum(r[a] for r in rows) for a in arms},"paired_own_minus":contrasts,"cases":rows}


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="mode",required=True)
    for name,fields in (("select",("dev","fit","output")),("confirm",("study","selection","output"))):
        q=sub.add_parser(name)
        for f in fields:
            q.add_argument(f,type=Path)
    args=ap.parse_args()
    data=choose(args.dev,args.fit) if args.mode=="select" else confirm(args.study,read(args.selection))
    with args.output.open("x") as f:
        json.dump(data,f,indent=2)
    print(json.dumps({k:v for k,v in data.items() if k!="cases"}),flush=True)
