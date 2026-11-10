"""Can a held-out support history distinguish a hidden rewarded side on a fresh query?"""
import argparse
import json
from pathlib import Path

import numpy as np
from readout import read

SIDES = ("left","right")


def public_features(episode, prefix, reward):
    # Lateral motion is observable on the robot. No target identity, teacher phase or object pose enters.
    actions=np.concatenate([prefix["commands"],episode["commands"]])
    track=np.concatenate([prefix["public"][:,1],episode["public"][1:,1]])
    y=track-track[0]
    movement=np.array([y[-1],y.min(),y.max(),y.mean(),actions[:,1].sum()/100,
                       np.maximum(actions[:,1],0).sum()/100,np.minimum(actions[:,1],0).sum()/100])
    return np.r_[movement,reward,movement*(2*reward-1)]


def predict(x, y, held):
    mean=x.mean(0)
    scale=np.maximum(x.std(0),1e-6)
    x=(x-mean)/scale
    held=(held-mean)/scale
    answer=np.linalg.solve(x.T@x + len(y)*.1*np.eye(x.shape[1]),x.T@(y-y.mean()))
    return y.mean()+held@answer


def evaluate(source, admission):
    seeds=read(admission)["admitted"]
    assert len(seeds) >= 4 and len(seeds)%2 == 0
    assignments=np.random.default_rng(110912).permutation(np.repeat([0,1],len(seeds)//2))
    traces=[]
    for index,seed in enumerate(seeds):
        donor=seeds[index^1]
        arm=SIDES[int(assignments[index])]
        place=read(source/str(seed)/f"{arm}.json")["placed"]
        with np.load(source/str(donor)/"left.npz") as query:
            # The current query is the same RGB/proprioceptive observation under each goal.
            query_y=query["public"][0,:3].copy()
        with np.load(source/str(seed)/f"{arm}.npz") as branch, np.load(source/str(seed)/"pickup.npz") as prefix:
            for goal in (0,1):
                reward=int(place[goal])
                traces.append(dict(seed=seed,query=donor,pair=index//2,arm=arm,goal=goal,
                                   completed=bool(any(place)),reward=reward,
                                   full=public_features(branch,prefix,reward),blind=public_features(branch,prefix,0),
                                   current=query_y))
    folds=[]
    for pair in range(len(seeds)//2):
        train=[r for r in traces if r["pair"]!=pair]
        test=[r for r in traces if r["pair"]==pair]
        assert {r["seed"] for r in train}.isdisjoint({r["query"] for r in test})
        y=np.array([r["goal"] for r in train])
        scores={k:predict(np.stack([r[k] for r in train]),y,np.stack([r[k] for r in test]))
                for k in ("full","blind","current")}
        for i,r in enumerate(test):
            folds.append({**{k:r[k] for k in ("seed","query","pair","arm","goal","completed","reward")},
                          **{k:int(s[i]>.5) for k,s in scores.items()}})
    def summary(rows):
        return {"n":len(rows),**{k:sum(r[k]==r["goal"] for r in rows) for k in ("full","blind","current")}}
    return {"all":summary(folds), "placed":summary([r for r in folds if r["completed"]]),
            "timeout":summary([r for r in folds if not r["completed"]]),"trials":folds}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("admission",type=Path)
    args=p.parse_args()
    print(json.dumps(evaluate(args.source,args.admission),indent=2))
