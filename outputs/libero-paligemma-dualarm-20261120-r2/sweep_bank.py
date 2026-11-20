"""Search native source pushes, replay successful command sequences with cameras, and retain actual successes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

import numpy as np

from expert import parse_layouts, write_json
from paddle import Route, trial


def candidates(pose, repeat, n):
    primary = Route("rim", 28, 180, -35)
    recipes = [primary, replace(primary,tilt=48,flank=0), replace(primary,flank=0),
               replace(primary,tilt=22,flank=-45), replace(primary,tilt=36,flank=-23)]
    rng = np.random.default_rng([798002,pose,repeat])
    for _ in range(n):
        recipes.append(Route("rim",float(rng.uniform(12,53)),float(180+rng.uniform(-24,24)),
                             float(rng.uniform(-58,20)),float(rng.uniform(.002,.011))))
    if repeat:
        rng.shuffle(recipes)
    return recipes[:n]


def lane(job):
    out, body, pose, rep, n, positive_bank, seed_base = job
    local = out / "search" / f"{body}-{pose}-{rep}"
    for name in ("records","drafts"):
        (local/name).mkdir(parents=True,exist_ok=True)
    render=out / "render" / f"{body}-{pose}-{rep}"
    for name in ("records",body):
        (render/name).mkdir(parents=True,exist_ok=True)
    seed = seed_base+pose+rep*1000
    attempts = 0
    render_rows=[]
    choices=candidates(pose,rep,n)
    saved=[]
    if positive_bank and body=="Sawyer":
        for f in sorted((positive_bank/"records").glob(f"{pose}-{seed}-*.json")):
            item=json.loads(f.read_text())
            if item["native_success"] and item["terminal_reward"]>0 and item["done"]:
                commands=np.load(positive_bank/"drafts"/(f.stem+".npy"),allow_pickle=False)
                saved.append((Route(**item["params"]),commands))
    for i,route in enumerate([p[0] for p in saved]+choices):
        if i<len(saved):
            action=saved[i][1]
        else:
            hit=trial((local,pose,seed,i,route,False),body=body)
            attempts+=1
            if not hit["native_success"]:
                continue
            action=np.load(local/"drafts"/f"{pose}-{seed}-{i}.npy",allow_pickle=False)
        for suffix,tape in enumerate((action,None)):
            row=trial((render,pose,seed,i*2+suffix,route,True),body=body,replay=tape)
            row["candidate_number"]=i
            row["repeat"]=rep
            row["headless_candidates"]=attempts
            if row["accepted"]:
                src=render/row["file"]
                dest=out/body/f"{pose}-{seed}-{i}-{suffix}.npz"
                src.rename(dest)
                row["file"]=str(dest.relative_to(out))
            render_rows.append(row)
            if row["accepted"]:
                return render_rows, attempts
    return render_rows,attempts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--body",choices=("Panda","Sawyer"),default="Sawyer")
    p.add_argument("--poses",type=parse_layouts,required=True)
    p.add_argument("--repeats",type=int,default=1)
    p.add_argument("--count",type=int,default=45)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--seed",type=int,default=20261120)
    p.add_argument("--positive-bank",type=Path)
    p.add_argument("--out",type=Path,default=Path("bank"))
    args=p.parse_args()
    if min(args.repeats,args.count,args.workers)<1:
        p.error("Positive repeat, candidate and worker counts required")
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/args.body).mkdir()
    index=dict(suite="libero_goal",case="BASE",task_id=5,bodies=[args.body],poses=args.poses,
               seed_base=args.seed,attempts=[],completed=[],source_constructor="fixed_primary_with_seeded_local_search")
    write_json(out/"index.json",index)
    jobs=[(out,args.body,pose,rep,args.count,args.positive_bank,args.seed)
          for rep in range(args.repeats) for pose in args.poses]
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures={pool.submit(lane,j):(j[2],j[3]) for j in jobs}
        for f in as_completed(futures):
            pose,rep=futures[f]
            rows,n=f.result()
            accepted=sum(int(r["accepted"]) for r in rows)
            index["attempts"].extend(rows)
            index["completed"].append(dict(pose=pose,repeat=rep,headless_candidates=n,accepted=accepted))
            write_json(out/"index.json",index)
            print("completed",args.body,pose,rep,"candidates",n,"native camera successes",accepted,flush=True)


if __name__=="__main__":
    main()
