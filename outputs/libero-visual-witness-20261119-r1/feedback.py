"""Cross both possible native reward assignments on each fixed history and check a last-frame reader."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
from score import preference


def main(source, score_file, out):
    report=json.loads(score_file.read_text())
    traces={}
    for split in report["splits"].values():
        for row in split["physical_tapes"]:
            key=row["pose"],row["goal"],row["variant"]
            with np.load(source/f"p{key[0]}-g{key[1]}-{key[2]}.npz") as z:
                x=z["frames"][-1].reshape(2,8,12,8,12,3).mean(axis=(2,4)).ravel()
            traces[key]=dict(x=x,own=row["own_hit"],ever=row["ever"],scores=dict(row["hit_score"]))
    for g in range(2):
        centers=[np.stack([r["x"] for (p,goal,v),r in traces.items() if goal==g and p<16 and r["own"]==h]).mean(axis=0)
                 for h in (False,True)]
        for (p,goal,v),row in traces.items():
            if g==goal:
                miss,hit=[np.linalg.norm(row["x"]-c) for c in centers]
                row["scores"]["last_rgb"]=miss/(miss+hit) if miss+hit else .5
    def reward_choice(bits,pa):
        if any(bits):
            assert bits in ((1,0),(0,1))
            return float(bits[0])
        return pa
    names=["rgb","last_rgb","proprio","commands"]
    output={}
    for split,info in report["splits"].items():
        cyclic=info["panels"]["cyclic"]["triples"]
        strict={(g["A"],g["B"],g["query"]) for g in info["groups"] if g["strict"]}
        totals=defaultdict(float)
        controls=defaultdict(lambda:np.zeros(2))
        records=[]
        for a,b,q in cyclic:
            for va,vb,h in product(("BASE","D18"),("BASE","D18"),(0,1)):
                ha,hb=traces[a,0,va],traces[b,1,vb]
                bits=tuple(int(row["ever"][h]) for row in (ha,hb))
                choices=np.array([traces[q,g,"BASE"]["ever"][h] for g in (0,1)],float)
                controls[(a,b,q,*bits)]+=choices
                totals["assignments"]+=1
                totals["reachable"]+=choices.max()
                totals["two_zero"]+=not any(bits)
                preds={}
                for mode in names:
                    pa=preference(ha["scores"][mode],hb["scores"][mode])
                    policy=reward_choice(bits,pa)
                    preds[mode]=policy
                    totals[mode+"_plus_reward"]+=np.dot([policy,1-policy],choices)
                    totals[mode+"_ignore_reward"]+=np.dot([pa,1-pa],choices)
                    paired=(va,vb,h) in (("D18","BASE",0),("BASE","D18",1))
                    if (a,b,q) in strict and paired:
                        totals[mode+"_strict"]+=np.dot([policy,1-policy],choices)
                records.append(dict(a=a,b=b,q=q,variants=(va,vb),assigned=h,rewards=bits,
                                    continuation_hits=choices.tolist(),probability_retry_cabinet=preds))
        totals["hindsight_pose_order_commands_and_reward_ceiling"]=sum(v.max() for v in controls.values())
        output[split]=dict(totals=dict(totals),cases=records,strict_assignments=info["panels"]["cyclic"]["assignments"])
    output["scope"]="Exploratory crossed hidden assignments on all four physical histories per cyclic triple. One positive identifies a prompted goal because the experiment records zero off-target hits throughout. Both zero uses the prespecified smaller predicted completion score. Ignore-reward controls keep both goals balanced for each fixed history."
    out.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main(*map(Path,sys.argv[1:]))
