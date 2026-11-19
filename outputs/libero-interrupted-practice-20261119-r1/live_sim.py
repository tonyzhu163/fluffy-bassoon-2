"""Native bottle simulator with a practice-only, evaluator-controlled actuator interruption."""
from multiprocessing.connection import Connection
import sys
import traceback

import numpy as np
from replay import ROOT, GOALS
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest
from libero.libero.envs.bddl_utils import robosuite_parse_problem


def serve(pipe):
    m=TaskManifest.load(str(ROOT/'sources/libero-pro2-e5500c69/assets/task_manifest.json'))
    paths=[m.resolve_bddl('libero_goal',g,'BASE') for g in GOALS['bottle']]
    problems=[robosuite_parse_problem(str(p)) for p in paths]
    goals=[p['goal_state'] for p in problems]
    prompts=[' '.join(x) if isinstance(x,list) else x for x in [p['language_instruction'] for p in problems]]
    env=LiberoEnvironmentAdapter()
    try:
        while True:
            op,value=pipe.recv()
            if op=='reset':
                pose,interrupted=value
                env.close()
                reset=env.create(dict(suite='libero_goal',task_id=2,variant='BASE',init_state_id=pose,seed=18180100+pose,
                                _bddl_path=str(paths[0]),camera_names=['agentview','robot0_eye_in_hand'],camera_height=256,
                                camera_width=256,control_mode='relative',max_steps=300))
                w=env._env.env
                obj=w.objects_dict['wine_bottle_1']
                body=w.obj_body_id['wine_bottle_1']
                z0=w.sim.data.body_xpos[body,2]
                step,start=0,None
                assert not any(all(w._eval_predicate(t) for t in goal) for goal in goals)
                result=reset['observation'],reset['environment_reproducibility']['initial_state_fingerprint'],prompts
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
                    contact=bool(w._check_grasp(w.robots[0].gripper,obj))
                    height=float(w.sim.data.body_xpos[body,2]-z0)
                    contacts.append(contact);heights.append(height)
                    step+=1
                    if start is None and contact and height>=.03 and step<300:start=step
                result=raw,states,hits,contacts,heights,start
            else:raise ValueError(op)
            pipe.send((True,result))
    except EOFError:pass
    except Exception:pipe.send((False,traceback.format_exc()))
    finally:env.close()


if __name__=='__main__':
    with Connection(int(sys.argv[1])) as pipe:serve(pipe)
