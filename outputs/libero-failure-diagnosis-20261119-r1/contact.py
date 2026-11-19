"""Replay the bottle practice tapes without rendering to measure physical grasp contact."""
import argparse
import json
from pathlib import Path
import numpy as np

from libero.libero import benchmark
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest

from panel import ROOT, tape_paths

PRO = ROOT.parent/'sources/libero-pro2-e5500c69'


def audit(path, states, manifest):
    record = json.loads((path/'result.json').read_text())
    with np.load(path/'public.npz') as d:
        issued, original = d['issued'], d['state_after']
    predicates = np.load(path/'evaluator-predicates.npy')
    names = [manifest.resolve_bddl('libero_goal', g, record['variant']) for g in (2,9)]
    goals = [robosuite_parse_problem(str(f))['goal_state'] for f in names]
    adapter = LiberoEnvironmentAdapter()
    adapter._init_states = states
    adapter._control_mode = 'relative'
    adapter._max_steps = 300
    adapter._env = ControlEnv(str(names[0]), use_camera_obs=False, has_offscreen_renderer=False, camera_names=[])
    try:
        idx = record['scene']
        reset = adapter.reset(idx, 18180100+idx)
        assert reset['environment_reproducibility']['initial_state_fingerprint'] == record['initial_physics']
        world = adapter._env.env
        obj = world.objects_dict['wine_bottle_1']
        body = world.obj_body_id['wine_bottle_1']
        z0 = world.sim.data.body_xpos[body, 2]
        contact, dz = [], []
        for i, act in enumerate(issued):
            raw = adapter.step(act)['observation']
            state = np.concatenate([raw[k] for k in ('robot0_eef_pos','robot0_eef_quat_xyzw','robot0_gripper_qpos')])
            np.testing.assert_array_equal(state, original[i], err_msg=f'{path} at control {i+1}')
            hits = [all(world._eval_predicate(t) for t in g) for g in goals]
            np.testing.assert_array_equal(hits, predicates[i])
            contact.append(bool(world._check_grasp(world.robots[0].gripper, obj)))
            dz.append(float(world.sim.data.body_xpos[body, 2] - z0))
        return dict(path=str(path.relative_to(ROOT)), contacts=sum(contact), first_contact=next((i+1 for i,b in enumerate(contact) if b),None),
                    peak_rise_m=max(dz), max_rise_in_grasp_m=max((z for b,z in zip(contact,dz) if b), default=None),
                    reached=record['first'][record['goal']] is not None, pose=idx, timing=record['variant'], target=record['goal'])
    finally:
        adapter.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    states = benchmark.get_benchmark_dict()['libero_goal']().get_task_init_states(2)
    manifest = TaskManifest.load(str(PRO/'assets/task_manifest.json'))
    with args.output.open('x',buffering=1) as out:
        for idx in range(6):
            for v in ('BASE','S20'):
                for path in tape_paths('bottle',idx,v)[0]:
                    print(json.dumps(audit(path, states, manifest)),file=out)
