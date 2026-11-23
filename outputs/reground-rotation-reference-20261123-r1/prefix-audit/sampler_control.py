"""Disposable CPU mechanistic control for all six anomalous drawer starts."""
from reset_inspection import TaskEnv, ROOT, HERE, digest, pixels, camera_delta
import json
import numpy as np

def inspect(ids,production_targets,drop_four=False):
    env=TaskEnv("10_3","rot_z/90")
    out={}
    try:
        for init in ids:
            if drop_four and init==production_targets[0]:
                history=env.rs.object_property_initializers
                assert len(history)>=4
                extra=history[-4:]
                assert len({(type(x).__name__,x.name,str(x.joint_ranges)) for x in extra})==1
                del history[-4:]  # In-memory intervention on extra draws in this single diagnostic env.
            reset=env.reset(init,0)
            if init in production_targets:
                rs=env.rs
                p={n:rs.sim.model.body_pos[rs.sim.model.body_name2id(o.root_body)].copy()
                   for n,o in rs.fixtures_dict.items()}
                out[init]=dict(init=digest(env.init_states[init]),x=digest(reset["x"]),
                               sim=digest(rs.sim.get_state().flatten()),body_pos=digest(rs.sim.model.body_pos),
                               body_quat=digest(rs.sim.model.body_quat),properties=len(rs.object_property_initializers),
                               camera=pixels(reset["obs"]),positions=p)
    finally: env.close()
    return out

def main():
    rows=[]
    for slot in range(4):
        targets=[i for i in (44+slot,48+slot) if i<50]
        history=[k+slot for k in (28,32,36,40)]
        cold=inspect(targets,targets)
        warm=inspect(history+targets,targets)
        rescue=inspect(history+targets,targets,True)
        for i in targets:
            a,b,c=cold[i],warm[i],rescue[i]
            row=dict(start=i,lane=slot,prior_ids=history,property_counts=[r["properties"] for r in (a,b,c)],
                     warm_vs_cold=camera_delta(b["camera"],a["camera"]),rescue_vs_cold=camera_delta(c["camera"],a["camera"]),
                     fixture_displacement_mm={n:float(np.linalg.norm(a["positions"][n]-b["positions"][n])*1000) for n in a["positions"]})
            for k in ("init","x","sim","body_pos","body_quat"):
                row[k]=[r[k] for r in (a,b,c)]
            row["both_camera_hashes"]=[list(map(digest,r["camera"])) for r in (a,b,c)]
            rows.append(row)
    (HERE/"sampler-control.json").write_text(json.dumps(sorted(rows,key=lambda r:r["start"]),indent=2)+"\n")

if __name__=="__main__": main()
