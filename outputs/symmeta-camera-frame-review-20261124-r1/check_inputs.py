"""Read immutable source banks and independently reconstruct one authentic trajectory in every view."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.wider_family import relabel

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
SHARD = ROOT / "xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards/shard-0/diverse.npz"
C = 31.5
P = np.array([[C,0,C],[0,C,C],[0,0,1.]])
FULL = np.ones((64,64), dtype="f4")


def rot(yaw):
    yaw = np.deg2rad(yaw)
    return np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])


def hmat(pose):
    roll, angle = pose[:2]
    dx,dy = pose[2:] or (0,0)
    k = np.array([[32/np.tan(np.pi/8),0,C],[0,32/np.tan(np.pi/8),C],[0,0,1.]])
    def proj(a):
        a = np.deg2rad(a)
        center = np.array([np.cos(a),np.sin(a),1.])/np.sqrt(2)
        front = -center
        side = np.cross(front,[0,0,1])
        side /= np.linalg.norm(side)
        r = np.stack((side,np.cross(front,side),front))
        return k @ np.column_stack((r[:,0],r[:,1],-r@center))
    h = np.vstack((cv2.getRotationMatrix2D((C,C),roll,1.),[0,0,1.]))
    if angle:
        h = h @ proj(angle) @ np.linalg.inv(proj(0))
    h[:2] += np.array([[dx],[dy]])*h[2:3]
    return h/h[2,2]


def warp(a, h, fill):
    return cv2.warpPerspective(a,h,(64,64),flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=(fill,)*3)


def source_geometry(folder, frames, idx):
    labels = json.loads((folder/"ready.json").read_bytes())["train_poses"]
    def arr(arm,k):
        return np.load(folder/f"train-{arm}-{k}.npy", mmap_mode="r", allow_pickle=False)
    features = {arm:{k:arr(arm,k) for k in ("calibration","grid_h","pixel_valid","img_before","img_after","dis_flow")} for arm in ("A","B")}
    out = []
    matrices=[]
    for j,pose in enumerate(labels):
        h=hmat(pose)
        hn=np.linalg.solve(P,h@P)
        hn/=hn[2,2]
        matrices.append(hn)
        mask = warp(FULL,h,0)>=1.
        obs = np.stack([warp(v,h,127) for v in frames])
        obs[:,~mask]=127
        coverage={}
        mask_hash={}
        for arm in ("A","B"):
            bank=features[arm]
            all_h=bank["calibration"][j].reshape(-1,3,3)
            np.testing.assert_allclose(all_h,np.broadcast_to(hn,all_h.shape),rtol=1e-6,atol=1e-6)
            if arm=="A":
                inv=np.linalg.inv(h)
                pixels=np.stack([warp(v,inv,127) for v in obs])
                valid=warp(mask.astype("f4"),inv,0)>=1.
                grid=hn
            else:
                pixels,valid=obs.copy(),mask
                grid=np.linalg.inv(hn)
            all_g=bank["grid_h"][j].reshape(-1,3,3)
            np.testing.assert_allclose(all_g, np.broadcast_to(grid/grid[2,2],all_g.shape),rtol=1e-6,atol=1e-6)
            pixels[:,~valid]=127
            safe=cv2.erode(valid.astype("u1"),np.ones((3,3),"u1"),borderType=cv2.BORDER_REPLICATE)
            flow=compute_dis_flow_maps(pixels[:-1],pixels[1:],np.ones(3,bool))
            flow[:,safe==0]=0.
            for k,v in (("img_before",pixels[:-1]),("img_after",pixels[1:]),
                        ("dis_flow",flow),("pixel_valid",np.stack([safe]*3))):
                np.testing.assert_array_equal(bank[k][j,idx],v,err_msg=f"{arm} {pose} {k}")
            coverage[arm]=float(safe.mean())
            mask_hash[arm]=hashlib.sha256(safe.tobytes()).hexdigest()
        out.append(dict(pose=pose, valid_fraction=coverage, valid_mask_sha256=mask_hash, center_shift_px=(C*hn[:2,2]).tolist()))
    return out,matrices


def check(base, extra):
    i=0
    with np.load(SHARD) as z, np.load(base/"train-control.npz") as bank:
        orig={k:z[k][i:i+1,:3] for k in ("issued_mean","delta_eef","delta_jaw","valid")}
        orig.update({k:z[k][i:i+1] for k in ("yaw_deg","xy_scale","gripper_sign")})
        for k,v in orig.items():
            np.testing.assert_array_equal(v[0],bank[k][i])
        pre,post=z["img_before"][i,:3],z["img_after"][i,:3]
        np.testing.assert_array_equal(pre[1:],post[:-1])
        frames=np.concatenate((pre[:1],post))
    newyaw, gain, sign = -79., 1.29, -1
    target=dict(yaw_deg=np.array([newyaw]),shape=gain*np.eye(2)[None],
                gripper_sign=np.array([sign]),z_gain=np.array([gain]))
    changed=relabel(orig,target)
    def dif(a,b):
        return float(np.max(np.abs(a-b)))
    aold,anew=orig["issued_mean"][0],changed["issued_mean"][0]
    delt, dup=orig["delta_eef"][0],changed["delta_eef"][0]
    oldyaw, oldgain, oldsign=(orig[k][0] for k in ("yaw_deg","xy_scale","gripper_sign"))
    max_diff=dict(world_xy=dif(oldgain*aold[:,:2]@rot(oldyaw).T, gain*anew[:,:2]@rot(newyaw).T),
                  world_local_displacement=dif(delt[:,:2]@rot(oldyaw).T, dup[:,:2]@rot(newyaw).T),
                  world_z=dif(oldgain*aold[:,2], gain*anew[:,2]),
                  gripper=dif(oldsign*aold[:,6], sign*anew[:,6]))
    assert np.allclose(changed["yaw_deg"],newyaw) and np.allclose(changed["xy_scale"],gain) and np.array_equal(changed["gripper_sign"],[sign])
    allviews,allh=[],[]
    for folder in (base,extra):
        views, hs=source_geometry(folder,frames,i)
        allviews.extend(views)
        allh.extend(hs)
    sv=np.linalg.svd((np.stack(allh)-np.eye(3)).reshape(len(allh),-1),compute_uv=False)
    pr=np.full(len(allviews),.8/(len(allviews)-1))
    pr[0]=.2
    classes={}
    for probability,v in zip(pr,allviews,strict=True):
        classes.setdefault(v["valid_mask_sha256"]["B"],[]).append((v["pose"],probability))
    shortcut=dict(distinct_masks=len(classes), maximum_source_pose_class_accuracy_from_mask=float(sum(max(p for _,p in items) for items in classes.values())),
                  ambiguous_groups=[[p for p,_ in items] for items in classes.values() if len(items)>1],
                  physical_raw="All valid for all views; the mask alone is constant.")
    return dict(source_mask_diagnostic=shortcut,source_tape="shard-0, index 0",nonvisual_relabel_residual=max_diff,source_views=allviews,
                centered_H_singular_values=sv.tolist(),tested_fields=["all trace H and other-grid H in each view",
                "one trace per view: RGB before, RGB after, guarded valid mask, DIS"],n_registered_views=len(allviews))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("base",type=Path)
    parser.add_argument("extra",type=Path)
    a=parser.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(check(a.base,a.extra),indent=2))
