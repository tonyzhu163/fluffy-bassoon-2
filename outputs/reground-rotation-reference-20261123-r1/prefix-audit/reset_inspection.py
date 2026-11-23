"""CPU-only reset inspection and replay of ten archived commands. No model or scoring."""
import json
import os
from pathlib import Path
from hashlib import sha256

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["GALLIUM_DRIVER"] == "llvmpipe"
from reground.envs.libero_env import TaskEnv
from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR


def digest(array):
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pixels(obs):
    return [obs["pixels"][k].copy() for k in ("image","image2")]


def camera_delta(pair_a, pair_b):
    out = []
    for a,b in zip(pair_a,pair_b):
        x = abs(a.astype(np.int16)-b.astype(np.int16))
        out.append(dict(changed_pixels=int(np.any(x != 0,axis=-1).sum()), max_channel=int(x.max()),
                        mean_abs_channel=float(x.mean())))
    return out


def read_trace(task, start, shard):
    batch = ROOT/f"reference-native/shards/seed0/rot_z/90/base/shard{shard:03}/traces"
    key = f"{task}__{start}__0__"
    for path in sorted(batch.glob("*.npz")):
        with np.load(path) as z:
            if key+"A" in z:
                return {f:z[key+f][:n] for f,n in (("X",11),("A",10),("speed",10))}
    raise ValueError(key)


def inspect(task, ids, keep, canonical=None, midpoint=False):
    env = TaskEnv(task,"rot_z/90")
    facts, vals = [], {}
    try:
        for k in ids:
            reset = env.reset(k,0)
            vendor,renderer = [glGetString(v).decode() for v in (GL_VENDOR,GL_RENDERER)]
            assert "llvmpipe" in renderer.lower() and "mesa" in vendor.lower()
            rs = env.rs
            positions = {name:(rs.sim.model.body_pos[i].copy(),rs.sim.model.body_quat[i].copy())
                         for name,obj in rs.fixtures_dict.items()
                         for i in [rs.sim.model.body_name2id(obj.root_body)]}
            raw = pixels(reset["obs"])
            s = rs.sim.get_state().flatten().copy()
            row = dict(id=k,vendor=vendor,renderer=renderer,init_sha256=digest(env.init_states[k]),
                       init_shape=list(np.shape(env.init_states[k])),x=digest(reset["x"]),sim=digest(s),
                       camera=list(map(digest,raw)),fixture_positions={n:p.tolist() for n,(p,_) in positions.items()},
                       fixture_quats={n:q.tolist() for n,(_,q) in positions.items()},
                       n_property_initializers=len(rs.object_property_initializers))
            if k in keep:
                vals[k]=(raw,positions)
                if canonical:
                    cold_img, cold_pos = canonical[k]
                    row["vs_cold"] = camera_delta(raw,cold_img)
                    # Diagnostic intervention after the archived-style reset, only on this disposable CPU env.
                    previous = {n:(p.copy(),q.copy()) for n,(p,q) in positions.items()}
                    for n,(p,q) in cold_pos.items():
                        idx = rs.sim.model.body_name2id(rs.fixtures_dict[n].root_body)
                        rs.sim.model.body_pos[idx], rs.sim.model.body_quat[idx] = p,q
                    rs.sim.forward()
                    row["vs_cold_with_cold_fixture_poses"] = camera_delta(pixels(env.render_obs()),cold_img)
                    for n,(p,q) in previous.items():
                        idx=rs.sim.model.body_name2id(rs.fixtures_dict[n].root_body)
                        rs.sim.model.body_pos[idx],rs.sim.model.body_quat[idx]=p,q
                    rs.sim.forward()
                if k == ids[-1] and len(ids)==1:
                    tr = read_trace(task,k,4 if task=="10_3" else 15)
                    xs, speeds, sim, end = [reset["x"]], [], [], None
                    for i,a in enumerate(tr["A"],1):
                        step = env.step(a, render=(i==10 or (midpoint and i==5)))
                        xs.append(step["x"])
                        speeds.append(env.eef_speed())
                        sim.append(digest(rs.sim.get_state().flatten()))
                        if i==10: end = list(map(digest,pixels(step["obs"])))
                    xs,speeds=np.stack(xs),np.asarray(speeds,np.float32)
                    row["replay"] = dict(midpoint=midpoint,X=digest(xs),speed=digest(speeds),sim=sim,
                                         end_camera=end,archived_X_exact=digest(xs)==digest(tr["X"]),
                                         archived_speed_exact=digest(speeds)==digest(tr["speed"]),actions=digest(tr["A"]))
            facts.append(row)
    finally:
        env.close()
    return facts,vals


def main():
    receipt={}
    for label,sequence in (("cold",[44,48]),("warm",[28,32,36,40,44,48])):
        facts,vals=inspect("10_3",sequence,{44,48},None if label=="cold" else examples)
        receipt[label]=facts
        if label=="cold": examples=vals
    for task,k in (("10_3",44),("spatial_0",48)):
        for mid in (False,True):
            receipt[f"{task}_render5_{mid}"]=inspect(task,[k],{k},midpoint=mid)[0]
    (HERE/"cpu-reset-inspection.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    main()
