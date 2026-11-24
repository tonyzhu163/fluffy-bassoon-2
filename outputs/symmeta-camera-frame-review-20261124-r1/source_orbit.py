"""Evaluate the sealed source readers on already recorded, untrained physical views of source arms."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import cv2
import jax
import numpy as np
from camera import FULL, view_inputs
from data import SENSOR
from model import decode, infer, restore


def read(path):
    return json.loads(path.read_bytes())


def run(fit,bank):
    assert jax.default_backend()=="cpu"
    seal=read(fit/"sealed.json")
    src=[r for r in read(bank)["rows"] if r["role"]=="source" and r["arm"] in ("Panda","Sawyer")]
    assert len(src)==8 and len({(r["arm"],r["id"]) for r in src})==8
    result=[]
    models={}
    for arm in ("A","B"):
        for seed in (41,42,43):
            ck=fit/f"{arm}-seed{seed}-step{seal['steps']}.msgpack"
            assert hashlib.sha256(ck.read_bytes()).hexdigest()==seal["checkpoint_sha256"][ck.name]
            models[arm,seed]=restore(ck,seed)
    for arm in ("A","B"):
        physical=[]
        for row in src:
            for view,v in row["views"].items():
                with np.load(v["passive_npz"],allow_pickle=False) as z:
                    np.testing.assert_array_equal(z["img_before"][0],z["frame0_img"])
                    np.testing.assert_array_equal(z["img_after"][:-1],z["img_before"][1:])
                    frames=np.concatenate((z["frame0_img"][None],z["img_after"]))
                    b={k:z[k] for k in SENSOR}
                b.update(view_inputs(frames,FULL,np.array(v["H_ref_to_view"]),arm))
                physical.append((row,view,b))
        b={k:np.stack([rec[2][k] for rec in physical]) for k in physical[0][2]}
        for s in (41,42,43):
            y,g,sign=decode(infer(models[arm,s],b))
            for (row,v,_), yy,gg,ss in zip(physical,y,g,sign,strict=True):
                result.append(dict(model=arm,seed=s,body=row["arm"],id=row["id"],view=v,yaw=float(yy),
                                   target=float(row["control_yaw_deg"]),yaw_error=float(abs((float(yy)-row["control_yaw_deg"]+180)%360-180)),
                                   abs_log_gain=float(abs(np.log(gg/row["xy_scale"]))),sign_correct=bool(ss==row["gripper_sign"])))
    by=defaultdict(list)
    for r in result:
        by[r["model"],r["view"]].append(r["yaw_error"])
    return dict(bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
                note="One familiar task and layout, four command maps per source body. No selection on these data.",
                means={f"{a}/{v}":float(np.mean(vals)) for (a,v),vals in by.items()}, rows=result)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    print(json.dumps(run(*map(Path,sys.argv[1:])),indent=2))
