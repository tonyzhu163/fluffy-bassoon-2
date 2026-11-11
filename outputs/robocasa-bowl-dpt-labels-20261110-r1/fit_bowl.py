"""Train paired frozen-π0.5 action adapters on validated, masked compact-bowl branches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation
from fluffy_bassoon.robocasa.residual import combine_observations
from corpus import Corpus, given_side_prompt, normalize_history
from fit import velocity
from bowl_students import BowlStudents

STEPS = (128,512,1024)


def batch(native, corpus, examples, mean, scale, noise_seed, *, given_side=False):
    values = [corpus.get(*e,mean,scale) for e in examples]
    raw, target, active, history = zip(*values)
    if given_side:
        for obs, (label, _, _, _) in zip(raw, examples, strict=True):
            obs["task_descriptions"] = [given_side_prompt(label["goal"])]
    target = torch.from_numpy(np.stack(target)).cuda()
    active = torch.from_numpy(np.stack(active)).cuda()
    history = torch.from_numpy(np.stack(history)).cuda()
    with torch.no_grad():
        obs = preprocess_observation(native.env_obs_to_observation(combine_observations(raw)), train=False)
        _, pad, cache = native.build_prefix_cache(obs)
        generator = torch.Generator(device="cuda").manual_seed(noise_seed)
        noise = torch.randn(target.shape, generator=generator,device="cuda")
        t = torch.rand(len(target), generator=generator,device="cuda").pow(1/1.5)*.999+.001
        x = torch.lerp(target,noise,t[:,None,None])
        x = torch.where(active[:,:,None],x,noise)
    return (obs,x,t,cache,pad),(noise-target)[:,:,:12],active,history


def losses(native, inputs, target, active):
    error=(velocity(native,inputs)-target).square().mean(-1)
    return (error*active).sum(1)/active.sum(1)


def opposite(corpus, examples, mean, scale):
    h=[normalize_history(corpus.supports[s,p,1-r["goal"]],mean,scale) for r,_,s,p in examples]
    return torch.from_numpy(np.stack(h)).cuda()


def validation_rows(corpus):
    examples=[]
    for r in corpus.rows:
        i=corpus.scenes.index(r["seed"])
        support=corpus.scenes[i^1]
        for probe in (0,1):
            anchors={0,r["valid_rows"]//2,r["valid_rows"]-1}
            if r["matched_decision_index"] is not None:
                anchors.add(r["matched_decision_index"])
            for a in sorted(anchors):
                examples.append((r,a,support,probe))
    return examples


@torch.no_grad()
def evaluate(native, students, valid, mean, scale, *, given_side=False):
    records=[]
    examples=validation_rows(valid)
    for start in range(0,len(examples),6):
        group=examples[start:start+6]
        students.select(None,None)
        inputs,target,active,own=batch(native,valid,group,mean,scale,90000+start,given_side=given_side)
        if given_side:
            arms = (("frozen_given_side",None,None),("oracle","memoryless",None))
        else:
            swap=opposite(valid,group,mean,scale)
            arms = (("frozen",None,None),("memoryless","memoryless",None),("own","dpt",own),("flipped","dpt",swap))
        scores={}
        for key,arm,history in arms:
            students.select(arm,history)
            scores[key]=losses(native,inputs,target,active).tolist()
        for j,(r,b,s,p) in enumerate(group):
            records.append({"scene":r["seed"],"goal":r["goal"],"query":b,"support":s,"probe":p,
                            "matched_decision":b==r["matched_decision_index"],**{k:v[j] for k,v in scores.items()}})
    return records


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(args):
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    assert torch.cuda.device_count()==1
    rng=np.random.default_rng(args.seed)
    milestones = STEPS
    if args.extend_to is not None:
        assert args.resume and args.extend_to > STEPS[-1]
        milestones = (*STEPS, args.extend_to)
    train=Corpus(args.train,args.train/"targets")
    valid=Corpus(args.valid,args.valid/"targets")
    assert set(train.scenes).isdisjoint(valid.scenes)
    mean,scale=train.statistics()
    with initialize_config_dir(config_dir=str(args.code/"benchmarks/robocasa/configs"),version_base=None):
        cfg=compose(config_name="robocasa365_pi05_eval")
    native=get_model(cfg.rollout.model,torch_dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    students=BowlStudents(native,len(mean)).cuda()
    params={k:list(v.parameters()) for k,v in students.adapters.items()}
    params["dpt"]+=list(students.encoder.parameters())+list(students.project.parameters())
    if args.given_side:
        params = {"memoryless":params["memoryless"]}
    opts={k:torch.optim.AdamW(ps,lr=3e-4,weight_decay=.01) for k,ps in params.items()}
    contract={"seed":args.seed,"steps":milestones,"batch_size":6,"code":os.environ["PILOT_COMMIT"],
              "given_side_in_prompt":args.given_side,
              "history_encoder":None if args.given_side else "2-layer, 4-head causal transformer, width 128, ALiBi",
              "training_labels":fingerprint(args.train/"targets/index.json"),
              "validation_labels":fingerprint(args.valid/"targets/index.json"),
              "trainable":{k:sum(p.numel() for p in ps) for k,ps in params.items()}}
    (args.out/"contract.json").write_text(json.dumps(contract,indent=2))
    start=0
    if args.resume:
        state=torch.load(args.resume,map_location="cpu",weights_only=False)
        if args.extend_to is None:
            assert state["contract"]==contract
        else:
            before = {k:v for k,v in state["contract"].items() if k not in ("steps","code")}
            after = {k:v for k,v in contract.items() if k not in ("steps","code")}
            assert before == after
            previous_steps = tuple(state["contract"]["steps"])
            assert previous_steps in (STEPS, milestones)
            assert STEPS[-1] <= state["step"] < args.extend_to
            if previous_steps == STEPS:
                assert state["step"] == STEPS[-1]
            np.testing.assert_array_equal(mean, state["mean"])
            np.testing.assert_array_equal(scale, state["scale"])
            (args.out/"resume.json").write_text(json.dumps({
                "checkpoint":str(args.resume), "sha256":fingerprint(args.resume),
                "previous_contract":state["contract"], "restored_step":state["step"],
                "continued_to":args.extend_to}, indent=2))
        students.load_state_dict(state["students"])
        for k,opt in opts.items():
            opt.load_state_dict(state["optimizers"][k])
        rng.bit_generator.state=state["draws"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start=state["step"]
    begin=time.monotonic()
    for step in range(start+1,milestones[-1]+1):
        examples=[train.sample(rng) for _ in range(6)]
        students.select(None,None)
        inputs,target,active,own=batch(native,train,examples,mean,scale,args.seed*10000+step,given_side=args.given_side)
        if step==1:
            with torch.no_grad():
                students.select(None,None)
                reference=velocity(native,inputs)
                for arm in opts:
                    students.select(arm,own)
                    assert torch.equal(velocity(native,inputs),reference)
        scores={}
        for arm,opt in opts.items():
            students.select(arm,own)
            opt.zero_grad(set_to_none=True)
            loss=losses(native,inputs,target,active).mean()
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(params[arm],1.,error_if_nonfinite=True)
            if step in (1,2):
                assert norm>0 and any(bank.up.weight.grad.abs().max()>0 for bank in students.adapters[arm].values())
                if arm=="dpt":
                    assert students.project.weight.grad.abs().max()>0
            opt.step()
            scores[arm]=loss.item()
        if step in (1,8) or step%32==0:
            print("UPDATE",step,round(time.monotonic()-begin,1),scores,flush=True)
        if step%64==0 or step in (1,8) or step==milestones[-1]:
            state={"step":step,"contract":contract,"students":students.state_dict(),"optimizers":{k:o.state_dict() for k,o in opts.items()},
                   "draws":rng.bit_generator.state,"cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                   "mean":mean,"scale":scale}
            torch.save(state,args.out/"next.tmp")
            os.replace(args.out/"next.tmp",args.out/"checkpoint.pt")
        if step in milestones:
            phase=args.out/f"step-{step:04d}"
            phase.mkdir()
            os.link(args.out/"checkpoint.pt",phase/"checkpoint.pt")
            measured=evaluate(native,students,valid,mean,scale,given_side=args.given_side)
            (phase/"offline.json").write_text(json.dumps(measured))
            metrics = ("oracle","frozen_given_side") if args.given_side else ("memoryless","own","flipped")
            print("VALIDATION",step,{k:np.mean([r[k] for r in measured]) for k in metrics},flush=True)
    print("COMPLETE",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for label in ("code","train","valid","out"):
        p.add_argument(label,type=Path)
    p.add_argument("--seed",type=int,default=991)
    p.add_argument("--resume",type=Path)
    p.add_argument("--given-side",action="store_true")
    p.add_argument("--extend-to",type=int)
    fit(p.parse_args())
