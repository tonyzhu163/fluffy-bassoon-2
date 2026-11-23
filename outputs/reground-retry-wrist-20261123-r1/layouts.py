"""Inspect only released episode banks and object/fixture geometry before scoring."""
import numpy as np

def run(out):
    import os
    from reground.envs.libero_env import TaskEnv
    from symmeta_adapter import digest
    from retry import TASKS, support_key
    import reference as ref
    assert os.environ["CUDA_VISIBLE_DEVICES"]==""
    facts = {}
    for task in TASKS:
        env = TaskEnv(task,"rot_z/90",cameras=False)
        try:
            bank = np.asarray(env.init_states)
            assert len(bank)==50 and env.le.control_freq==20
            sim=env.rs.sim
            objects = {**env.rs.objects_dict,**env.rs.fixtures_dict}
            body = [sim.model.body_name2id(v.root_body) for v in objects.values()]
            poses = []
            for state in bank:
                sim.set_state_from_flattened(state)
                sim.forward()
                poses.append(np.concatenate((sim.data.body_xpos[body],sim.data.body_xquat[body]),axis=1))
            pair_rows = []
            for i in range(15,50):
                _,s,_ = support_key((task,i,0))
                distances = np.linalg.norm(poses[i][:,:3]-poses[s][:,:3],axis=1)
                pair_rows.append(dict(query=i,support=s, max_object_root_displacement_m=float(distances.max()),
                                      support_pose_sha256=digest(poses[s]),query_pose_sha256=digest(poses[i])))
            facts[task]=dict(bank_sha256=digest(bank),bodies=list(objects),pairs=pair_rows)
        finally:
            env.close()
    mins = {t:min(r["max_object_root_displacement_m"] for r in v["pairs"]) for t,v in facts.items()}
    ref.write(out/"layout-banks.json",facts)
    ref.require(all(x>0 for x in mins.values()),f"Repeated layout in assigned supports: {mins}")
    ref.write(out/"passed.json",dict(code=os.environ["RETRY_PIN"], pairs=sum(len(v["pairs"]) for v in facts.values()),
              compared="World position and orientation of released object and fixture root bodies at bank states",
              min_of_max_object_root_displacement_m=mins, control_hz=20, gpu_hours=0))
