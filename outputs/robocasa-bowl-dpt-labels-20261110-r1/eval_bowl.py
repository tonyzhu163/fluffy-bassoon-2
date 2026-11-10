"""Replay the neutral pickup, then hand the matched query directly to frozen π0.5 or a student."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from rlinf.models.embodiment.openpi import get_model

from collect import CAMERAS, SIDES, convert_action, held_fruit, identity, release, reset_scene, stable_counter_regions, state
from corpus import normalize_history, observation, public_history
from fit import PairedStudents
from readout import read


def actor(code, weights):
    with initialize_config_dir(config_dir=str(code/"benchmarks/robocasa/configs"),version_base=None):
        cfg=compose(config_name="robocasa365_pi05_eval")
    native=get_model(cfg.rollout.model,torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    assert native.num_steps==10 and native.action_horizon==50
    students=PairedStudents(native,len(weights["mean"])).cuda().eval().requires_grad_(False)
    students.load_state_dict(weights["students"])
    return native,students


def plans(seeds, query_index, arms):
    query=seeds[query_index]
    own=seeds[query_index^1]
    donor=seeds[query_index^2]
    assert len({query,own,donor})==3
    probe=int(np.random.default_rng(110922).permutation(np.repeat([0,1],len(seeds)//2))[query_index])
    for arm in arms:
        goals=(0,1) if arm in ("own","donor") else (None,)
        for goal in goals:
            yield {"arm":arm,"goal":goal,"query":query,"support":donor if arm=="donor" else own,
                   "support_goal":1-goal if arm=="donor" else goal, "probe":probe}


@torch.inference_mode()
def episode(native,students,weights,source,plan):
    arm=plan["arm"]
    if arm in ("own","donor","masked"):
        goal=plan["support_goal"] if arm!="masked" else 0
        raw=public_history(source/str(plan["support"]),plan["probe"],goal)
        if arm=="masked":
            raw[:,-3]=0
        h=torch.from_numpy(normalize_history(raw,weights["mean"],weights["scale"])[None]).cuda()
        students.select("dpt",h)
    else:
        students.select("memoryless" if arm=="memoryless" else None,None)
    seed=plan["query"]
    path=source/str(seed)
    prefix=read(path/"pickup.json")
    result={**plan,"prefix_controls":prefix["controls"],"grasped":prefix["fruit"] is not None,
            "decision":prefix["decision"],"placed":[False,False],"executed":0,"native_four_fruit":False}
    wrapped,obs,layout=reset_scene(seed)
    try:
        env=wrapped.unwrapped.env
        assert identity(env,obs)==prefix["initial"]
        with np.load(path/"pickup.npz") as tape:
            for a,after in zip(tape["commands"],tape["post"],strict=True):
                obs,_,_,_,_=wrapped.step(convert_action(a))
                np.testing.assert_array_equal(state(obs),after)
        assert identity(env,obs)==prefix["decision"] and held_fruit(env)==prefix["fruit"]
        if not prefix["fruit"]:
            return result
        rng=torch.Generator(device="cuda").manual_seed(seed+800701)
        for index in range(500-prefix["controls"]):
            if index%5==0:
                raw=observation(np.stack([obs[k] for k in CAMERAS]),state(obs))
                noise=torch.randn((1,50,32),device="cuda",dtype=torch.float32,generator=rng)
                proposal,_=native.predict_action_batch(raw,mode="eval",noise=noise)
                chunk=proposal.detach().float().cpu().numpy()[0]
                assert chunk.shape==(50,12) and np.isfinite(chunk).all()
            obs,_,done,truncated,info=wrapped.step(convert_action(chunk[index%5]))
            result["executed"]+=1
            result["native_four_fruit"] |= bool(info["success"])
            result["placed"]=release(env,layout["bowls"])
            if any(result["placed"]) or done or truncated:
                break
        return result
    finally:
        wrapped.close()


def main(args):
    torch.set_num_threads(3)
    torch.manual_seed(991)
    assert torch.cuda.device_count()==1
    stable_counter_regions()
    weights=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    native,students=actor(args.code,weights)
    seeds=read(args.admission)["admitted"]
    assert len(seeds)>=4 and len(seeds)%4==0
    args.out.mkdir()
    (args.out/"identity.json").write_text(json.dumps({"checkpoint":str(args.checkpoint),"step":weights["step"],
                            "fit":weights["contract"],"scenes":seeds,"arms":args.arms},indent=2))
    for i,seed in enumerate(seeds):
        for plan in plans(seeds,i,args.arms):
            record=episode(native,students,weights,args.source,plan)
            name=f"{seed}-{plan['arm']}-{plan['goal']}.json"
            (args.out/name).write_text(json.dumps(record))
            print("NATIVE",seed,plan["arm"],plan["goal"],record["placed"],record["executed"],flush=True)
    (args.out/"complete.json").write_text(json.dumps({"scenes":len(seeds),"arms":args.arms}))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("code","checkpoint","source","admission","out"):
        p.add_argument(name,type=Path)
    p.add_argument("--arms",nargs="+",choices=("frozen","memoryless","own","donor","masked"),required=True)
    main(p.parse_args())
