"""Execute two native Object goals in the same soup-task world."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback
import numpy as np

from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest


def serve(pipe):
    root=Path('/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69')
    manifest=TaskManifest.load(str(root/'assets/task_manifest.json'))
    paths=[manifest.resolve_bddl('libero_object',g,'BASE') for g in (0,7)]
    problems=[robosuite_parse_problem(str(p)) for p in paths]
    goals=[p['goal_state'] for p in problems]
    prompts=[' '.join(v) if isinstance(v,list) else v for v in [p['language_instruction'] for p in problems]]
    names=('alphabet_soup_1','milk_1')
    env=LiberoEnvironmentAdapter()
    try:
        while True:
            op,value=pipe.recv()
            if op=='reset':
                pose,interrupted=value
                env.close()
                r=env.create(dict(suite='libero_object',task_id=0,variant='BASE',init_state_id=pose,
                                  seed=19201000+pose,_bddl_path=str(paths[0]),camera_names=['agentview','robot0_eye_in_hand'],
                                  camera_height=256,camera_width=256,control_mode='relative',max_steps=300))
                w=env._env.env
                objects=[w.objects_dict[n] for n in names]
                bodies=[w.obj_body_id[n] for n in names]
                z0=w.sim.data.body_xpos[bodies,2].copy()
                assert w.parsed_problem['goal_state']==goals[0]
                assert not any(all(w._eval_predicate(t) for t in goal) for goal in goals)
                step,start,which=0,None,None
                result=r['observation'],r['environment_reproducibility']['initial_state_fingerprint'],prompts
            elif op=='chunk':
                states,hits,contacts,heights=[],[],[],[]
                for intended in value:
                    a=intended.copy()
                    if interrupted and start is not None and start<=step<start+10:a[-1]=-1.
                    rec=env.step(a)
                    raw=rec['observation']
                    states.append(np.concatenate([raw[k] for k in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')]))
                    hits.append([all(w._eval_predicate(t) for t in goal) for goal in goals])
                    assert hits[-1][0]==rec['success']
                    grasp=[bool(w._check_grasp(w.robots[0].gripper,obj)) for obj in objects]
                    dz=w.sim.data.body_xpos[bodies,2]-z0
                    contacts.append(grasp);heights.append(dz.copy())
                    step+=1
                    lifted=np.flatnonzero(np.array(grasp)&(dz>=.03))
                    if start is None and len(lifted) and step<300:start,which=step,names[lifted[0]]
                result=raw,states,hits,contacts,heights,start
            elif op=='event':result=which
            else:raise ValueError(op)
            pipe.send((True,result))
    except EOFError:pass
    except Exception:pipe.send((False,traceback.format_exc()))
    finally:env.close()


if __name__=='__main__':
    with Connection(int(sys.argv[1])) as pipe:serve(pipe)
