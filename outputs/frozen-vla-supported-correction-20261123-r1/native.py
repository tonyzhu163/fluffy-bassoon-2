"""Single paired native Kinova3 screening panel for estimating rescues and harm."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adapter import issue
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from native_rollout import check_native
from panel import cases, key, save
from ur5e_motor import load_frozen_pi05, run_episode
from ur5e_sim_worker import UR5eSimulator

METHODS = ('identity','rank1','rank1_free','smooth_df')


class Mapped:
    def __init__(self, sim, action, renderer):
        self.sim, self.action, self.renderer = sim, action, renderer
        self.packets = []

    def call(self, op, value):
        if op == 'create':
            spec, first = self.sim.call(op, value)
            model = first['model_identity']
            assert spec['embodiment'] == 'nom' and spec['arm'] == model['arm'] == 'Kinova3'
            assert model['xml_sha256'] == '2884d72a0427223d333dc00fa2c199c3f86406184c234f436436c7d149002cd7'
            assert model['controller']=='OSC_POSE' and model['control_freq']==20 and model['gripper']=='PandaGripper'
            assert model['joint_count'] == len(model['joint_names']) == len(first['observation']['robot0_joint_pos'])
            assert first['worker_step_index']==0 and first['action_schema']['shape']==[7]
            if self.renderer == 'gpu':
                assert first['renderer_vendor']=='NVIDIA Corporation'
            self.first = first
            return spec, first
        assert op == 'chunk'
        issued = issue(value, **self.action)
        np.testing.assert_array_equal(issued[:,3:],value[:,3:])
        direction = self.action['direction']
        if direction is not None:
            delta = issued[:,:3].astype(float) - value[:,:3]
            v = np.array(direction)
            np.testing.assert_allclose(delta - (delta @ v)[:,None]*v, 0, atol=np.finfo(value.dtype).eps)
        step = self.sim.call(op, issued)
        n = step['steps']
        self.packets.append(dict(frozen=value[:n].tolist(), issued=issued[:n].tolist()))
        return step


class OpeningFixture:
    """Recorded pi05 opening, padded to model width after its first executed burst."""
    def __init__(self, tape):
        self.opening = np.concatenate([tape, tape]).astype(np.float32)

    def infer(self, item, *, noise):
        assert item['observation/image'].shape==(224,224,3) and noise.shape==(10,32)
        return {'actions': self.opening}


def score(cells):
    pairs = {}
    for candidate,reference in (('rank1','identity'),('rank1_free','identity'),('smooth_df','identity'),
                                ('rank1','rank1_free'),('rank1','smooth_df')):
        transitions = Counter((r['success'][reference], r['success'][candidate]) for r in cells)
        rescued = transitions[False, True]
        harmed = transitions[True, False]
        positive = harmed + transitions[True,True]
        negative = rescued + transitions[False,False]
        pairs[candidate+'_minus_'+reference] = dict(pairs=len(cells), rescued=rescued, harmed=harmed,
                 both_success=transitions[True,True], both_failure=transitions[False,False],
                 reference_successes=positive, reference_failures=negative,
                 harm_per_reference_success=harmed/positive if positive else None,
                 rescue_per_reference_failure=rescued/negative if negative else None,
                 difference=(rescued-harmed)/len(cells))
    return dict(cases=cells,contrasts=pairs)


def main(args):
    payload = args.matrices.read_bytes()
    matrix_sha = sha256(payload).hexdigest()
    assert matrix_sha == 'ad36434d3e748583ed3ca338d26b782fe93aa437652e24abf9013be166edc1a5'
    adapters = json.loads(payload)
    panel = list(cases('Kinova3'))
    if args.fixture:
        panel = panel[:1]
        raw = json.loads(args.fixture.read_bytes())
        assert raw['case'] == panel[0]
        policy = OpeningFixture(raw['chunks'][0]['cmd'])
    else:
        policy = load_frozen_pi05()
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES':''}):
        simulator = UR5eSimulator(allow_mesa=bool(args.fixture))
    cells = []
    with simulator as sim:
        for case in panel:
            row = adapters['Kinova3/'+case['suite']+'/'+str(case['task_id'])]
            support = row['support']
            assert support['arm']=='Kinova3' and support['task']==f"{case['suite']}/{case['task_id']}"
            assert support['controls']==15 and support['start']==11 and case['init_state_id'] in (14,15)
            cell = dict(case=case,success={},controls={})
            for method in METHODS:
                runner = Mapped(sim,row['adapters'][method], 'cpu' if args.fixture else 'gpu')
                record = run_episode(runner,policy,case,get_embodiment_spec('nom'),
                                     canonicalize_quat=False,max_controls=5 if args.fixture else None)
                first = runner.first
                assert sum(len(p['issued']) for p in runner.packets)==record['controls']
                assert first['environment_reproducibility']['initial_state_fingerprint']==record['reset']['physics']!=support['physics']
                opening = np.array(runner.packets[0]['frozen'],dtype=np.float32)
                if method=='identity':
                    reset, full, model = record['reset'], first['environment_reproducibility'], first['model_identity']
                    opening_sha, nominal = record['opening_proposal_sha256'], opening
                else:
                    assert reset==record['reset'] and model==first['model_identity'] and full==first['environment_reproducibility']
                    assert opening_sha==record['opening_proposal_sha256']
                    n=min(len(nominal),len(opening))
                    np.testing.assert_array_equal(nominal[:n],opening[:n])
                if args.fixture:
                    assert record['controls']==len(record['native_rewards'])==5
                else:
                    check_native(record)
                    assert record['native_return']==sum(record['native_rewards'])
                    assert not record['truncated'] or record['controls']==reset['horizon']
                cell['success'][method]=record['success']
                cell['controls'][method]=record['controls']
                save(args.out/(key(case)+'-'+method+'.json'),dict(case=case,method=method,record=record,
                     model=model,reset_physics=full,support=support,matrices_sha256=matrix_sha,
                     action_packets=runner.packets,fixture=bool(args.fixture)))
            cells.append(cell)
            print(key(case),cell['controls'] if args.fixture else cell['success'],flush=True)
            if args.fixture:
                save(args.out/'fixture.json',dict(status='qualified',paired_opening=opening_sha,physics=reset['physics'],
                    methods=list(METHODS),controls_each=5,adapter_sha256=matrix_sha))
            else:
                save(args.out/'paired.json',dict(complete=len(cells)==len(panel),**score(cells)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('out',type=Path)
    p.add_argument('matrices',type=Path)
    p.add_argument('--fixture',type=Path)
    main(p.parse_args())
