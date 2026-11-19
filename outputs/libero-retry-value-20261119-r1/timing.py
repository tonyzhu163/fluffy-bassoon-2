"""Vary the timing of one gripper intervention on identical recorded motor commands."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import product
from pathlib import Path
import json
import sys

import numpy as np
from libero.libero import benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest

from experiment import histories, methods, prior, read_tape, summarize_panel

ROOT = prior.ROOT.parent
PILOT = prior.ROOT/'libero-bowl-decisions-20261118-r1'
PRACTICE = range(10,15)
QUERY = range(15,20)
ORDERS = [(p, 10+(p-9)%5, p+5) for p in PRACTICE]


@lru_cache(maxsize=1)
def setup():
    manifest=TaskManifest.load(str(ROOT/'sources/libero-pro2-e5500c69/assets/task_manifest.json'))
    paths=[manifest.resolve_bddl('libero_goal',g,'BASE') for g in (4,8)]
    goals=[robosuite_parse_problem(str(path))['goal_state'] for path in paths]
    bank=benchmark.get_benchmark_dict()['libero_goal']().get_task_init_states(4)
    return str(paths[0]),goals,bank


def execute(job):
    out,pose,goal,mode=job
    worker='nominal02' if pose<12 else 'paired01' if pose<16 else 'pacing01'
    path=PILOT/worker/'episodes'/f's{pose}-BASE-g{goal}-gain1'
    rec=json.loads((path/'result.json').read_text())
    assert rec['scene']==pose and rec['goal']==goal and rec['variant']=='BASE' and rec['gain']==1
    with np.load(path/'public.npz') as tape:
        actions, reference=tape['issued'],tape['state_after']
    native=np.load(path/'evaluator-predicates.npy')
    assert actions.shape==(300,7) and native.shape==(300,2)
    world_path,goals,bank=setup()
    env=LiberoEnvironmentAdapter()
    env._env=ControlEnv(world_path,use_camera_obs=False,has_offscreen_renderer=False,camera_names=[])
    env._init_states=bank
    env._max_steps=300
    try:
        reset=env.reset(pose,18180100+pose)
        assert reset['environment_reproducibility']['initial_state_fingerprint']==rec['initial_physics']
        world=env._env.env
        assert world.parsed_problem['goal_state']==goals[0]
        obj=world.objects_dict['akita_black_bowl_1']
        body=world.obj_body_id['akita_black_bowl_1']
        z0=world.sim.data.body_xpos[body,2]
        early=late=start=None
        peak=0.
        states,hits=[],[]
        for i,original in enumerate(actions):
            a=original.copy()
            if start is not None and start<=i<start+10:
                a[-1]=-1.
            record=env.step(a)
            raw=record['observation']
            state=np.concatenate([raw[k] for k in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
            h=[all(world._eval_predicate(t) for t in goal_terms) for goal_terms in goals]
            assert h[0]==record['success']
            if start is None or i<start:
                np.testing.assert_array_equal(state,reference[i])
                np.testing.assert_array_equal(h,native[i])
            states.append(state);hits.append(h)
            z=float(world.sim.data.body_xpos[body,2]-z0)
            grasp=bool(world._check_grasp(world.robots[0].gripper,obj))
            if i<299 and early is None and grasp and z>=.03:
                early=i+1
            if early is not None:
                peak=max(peak,z)
                if i<299 and late is None and grasp and z<=peak-.03:
                    late=i+1
            start={'early':early,'late':late}.get(mode)
        target=out/f'p{pose}-g{goal}-{mode}.npz'
        np.savez_compressed(target,issued=actions,states=states,hits=hits)
        return dict(pose=pose,goal=goal,mode=mode,start=start,early=early,late=late,
                    ever=np.any(hits,axis=0).tolist(),final=hits[-1])
    finally:
        env.close()


def main(out):
    data_dir=out/'episodes'
    data_dir.mkdir()
    with np.load(out.parent/'value01/model.npz') as saved:
        bank=dict(saved)
    with ProcessPoolExecutor(max_workers=4) as pool:
        practice=list(pool.map(execute,[(data_dir,p,g,m) for p,g,m in product(PRACTICE,(0,1),('normal','early','late'))]))
        saved={}
        for m in ('late','early'):
            tapes={(p,g,0,b):read_tape(data_dir/f'p{p}-g{g}-{m if b else "normal"}.npz')
                   for p,g,b in product(PRACTICE,(0,1),(False,True))}
            placeholder={q:(np.zeros((2,2)),np.zeros((2,2))) for q in QUERY}
            d=histories(tapes,placeholder,1,orders=ORDERS)
            ps=methods(bank,d,tapes)
            saved[m]=dict(tapes=tapes,ps=ps)
        path=out/'choices.json'
        with path.open('x') as handle:
            json.dump({m:{k:p.tolist() for k,p in v['ps'].items()} for m,v in saved.items()},handle)
        print('Frozen both 80-assignment panels',flush=True)
        observed=list(pool.map(execute,[(data_dir,p,g,'query') for p,g in product(QUERY,(0,1))]))
    queries={}
    for q in QUERY:
        pair=[read_tape(data_dir/f'p{q}-g{g}-query.npz')['hits'] for g in (0,1)]
        queries[q]=(np.stack([v.any(axis=0) for v in pair]),np.stack([v[-1] for v in pair]))
    frozen=json.loads(path.read_text())
    report=dict(practices=practice,queries=observed)
    for m,v in saved.items():
        d=histories(v['tapes'],queries,1,orders=ORDERS)
        ps=methods(bank,d,v['tapes'])
        assert all(np.array_equal(p,frozen[m][k]) for k,p in ps.items())
        report[m]=summarize_panel(d,ps,queries)
        print(m,report[m]['all']['native'],flush=True)
    (out/'timing.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main(Path(sys.argv[1]))
