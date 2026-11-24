"""Replay the selected checkpoints on all source holdouts and measure each factor on CPU."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from data import SENSOR, VIS
from eval_stage2a_live_ur5e import load_v2_checkpoint
from fluffy_bassoon.xemb.equiv_v2 import apply_reader_v2
from model import CFG, decode, infer, restore

SEEDS=(41,42,43)
HIST=Path("/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/train02pro/checkpoints")


def read(p):
    return json.loads(p.read_bytes())


def wrap(v):
    return (v+180)%360-180


def run(fit, base, extra):
    assert jax.default_backend()=="cpu"
    selection=read(fit/"sealed.json")
    steps=selection["steps"]
    ctl=np.load(base/"val-control.npz")
    y,g,q=(ctl[k] for k in ("yaw_deg","xy_scale","gripper_sign"))
    poses=selection["validation_poses"]
    statistics=defaultdict(list)
    residuals=defaultdict(list)
    stock=jax.jit(lambda p,b:apply_reader_v2(p,CFG,b))
    for arm in ("A","B","shipped"):
        p={}
        for s in SEEDS:
            if arm=="shipped":
                p[s],cfg,metadata=load_v2_checkpoint(HIST/f"perceiver_cont_warm-passive_attempt1-seed{s}.msgpack")
                assert cfg==CFG and metadata["seed"]==s
            else:
                path=fit/f"{arm}-seed{s}-step{steps}.msgpack"
                assert hashlib.sha256(path.read_bytes()).hexdigest()==selection["checkpoint_sha256"][path.name]
                p[s]=restore(path,s)
        j=0
        for bank_path in (base,extra):
            bank_arm="B" if arm=="shipped" else arm
            keys=("img_before","img_after","dis_flow") if arm=="shipped" else VIS
            bank={k:np.load(bank_path/f"val-{bank_arm}-{k}.npy",mmap_mode="r") for k in keys}
            for jlocal in range(len(bank["dis_flow"])):
                if arm=="shipped" and j>0:
                    break
                outputs={s:[] for s in SEEDS}
                for start in range(0,len(y),32):
                    dx=np.minimum(np.arange(start,start+32),len(y)-1)
                    b={k:ctl[k][dx] for k in SENSOR}
                    b.update({k:v[jlocal,dx] for k,v in bank.items()})
                    for s in SEEDS:
                        factors=decode((stock if arm=="shipped" else infer)(p[s],b))
                        outputs[s].append(np.column_stack(factors)[:min(len(y)-start,32)])
                for s in SEEDS:
                    yh,gh,qh=np.concatenate(outputs[s]).T
                    err=wrap(yh-y)
                    a=np.abs(err)
                    if arm!="shipped":
                        original=np.load(fit/f"val-{arm}-{s}-{steps}.npy",mmap_mode="r")[j]
                        residuals[arm].append(float(np.abs(a-original).max()))
                    statistics[arm].append(dict(seed=s,pose=poses[j],yaw_mae=float(a.mean()),
                                                 mean_signed_yaw=float(err.mean()),yaw_median=float(np.median(a)),
                                                 abs_log_gain=float(np.abs(np.log(gh/g)).mean()),
                                                 grip_accuracy=float(np.mean(qh==q))))
                j+=1
    return dict(steps=steps,n=len(y),results=statistics,max_abs_cpu_gpu_yaw_error_difference=
                {k:max(v) for k,v in residuals.items()})


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    for field in ("fit","base","extra"):
        cli.add_argument(field,type=Path)
    args=cli.parse_args()
    print(json.dumps(run(args.fit,args.base,args.extra),indent=2))
