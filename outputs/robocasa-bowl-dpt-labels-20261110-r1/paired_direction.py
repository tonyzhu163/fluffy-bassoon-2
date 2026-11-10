"""Measure whether changing only observed support reward moves a native action toward its physical teacher."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corpus import normalize_history, observation, public_history
from eval_bowl import actor
from readout import read


@torch.inference_mode()
def measure(args):
    weights=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    native,student=actor(args.code,weights)
    seeds=read(args.admission)["admitted"]
    labels=read(args.source/"targets/index.json")["labels"]
    by_key={(r["seed"],r["goal"]):r for r in labels}
    probes=np.random.default_rng(110922).permutation(np.repeat([0,1],len(seeds)//2))
    cases=[]
    for i,seed in enumerate(seeds):
        pair=[by_key[seed,g] for g in (0,1)]
        if any(r["label"] is None or r["matched_decision_index"] is None for r in pair):
            continue
        index=pair[0]["matched_decision_index"]
        assert pair[1]["matched_decision_index"]==index
        with np.load(pair[0]["query_archive"]) as left, np.load(pair[1]["query_archive"]) as right:
            frame=left["frames"][index]
            state=left["states"][index]
            np.testing.assert_array_equal(frame,right["frames"][index])
            np.testing.assert_array_equal(state,right["states"][index])
            raw=observation(frame,state)
            offset=int(left["offsets"][index])
            assert right["offsets"][index]==offset
            count=min(50,len(left["commands"])-offset,len(right["commands"])-offset)
            teacher=np.stack([x["commands"][offset:offset+count,:3] for x in (left,right)])
        diff=teacher[0]-teacher[1]
        assert np.square(diff).sum()>0
        support=seeds[i^1]
        probe=int(probes[i])
        context=[public_history(args.source/str(support),probe,g) for g in (0,1)]
        complete=bool(context[0][-1,-3]!=context[1][-1,-3])
        assert np.count_nonzero(context[0]!=context[1])==int(complete)
        summaries=[]
        for rep in range(4):
            gen=torch.Generator(device="cuda").manual_seed(seed+801701+10000*rep)
            noise=torch.randn((1,50,32),device="cuda",generator=gen)
            pred=[]
            for values in context:
                h=normalize_history(values,weights["mean"],weights["scale"])
                student.select("dpt",torch.from_numpy(h[None]).cuda())
                proposal,_=native.predict_action_batch(raw,mode="eval",noise=noise.clone())
                pred.append(proposal[0,:count,:3].detach().float().cpu().numpy())
            pred=np.stack(pred)
            if not complete:
                np.testing.assert_array_equal(pred[0],pred[1])
            delta=pred[0]-pred[1]
            direction=float((delta*diff).sum()/np.square(diff).sum())
            own=np.square(pred-teacher).mean((1,2))
            swapped=np.square(pred-teacher[::-1]).mean((1,2))
            summaries.append({"projection":direction,"own_translation_mse":own.tolist(),"swapped_translation_mse":swapped.tolist()})
        cases.append({"scene":seed,"support":support,"probe":probe,"informative":complete,"controls":count,
                      "mean_projection":float(np.mean([s["projection"] for s in summaries])),"draws":summaries})
    return {"cases":cases,"checkpoint":str(args.checkpoint),"fit_seed":weights["contract"]["seed"],
            "step":weights["step"],"admitted_scenes":len(seeds)}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for field in ("code","checkpoint","source","admission","output"):
        p.add_argument(field,type=Path)
    args=p.parse_args()
    result=measure(args)
    with args.output.open("x") as f:
        json.dump(result,f,indent=2)
