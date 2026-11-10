"""Repeat two real frozen-Sidney command tapes in new seeded RoboTwin scenes."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'robotwin-motus-release-baseline-20261109-r1'))
from evaluate import environment_args, save
from envs.camera.camera import Camera
from envs.place_phone_stand import place_phone_stand
from fluffy_bassoon.robotwin.execution import install_wrist_near

CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def snapshot(task, obs):
    def pose(actor):
        p = actor.get_pose()
        return [*p.p, *p.q]
    return dict(public=np.array(obs['joint_action']['vector']),
                poses=np.array([pose(task.phone), pose(task.stand)]),
                joints=np.concatenate([v for arm in (task.robot.left_entity, task.robot.right_entity)
                                       for v in (arm.get_qpos(), arm.get_qvel())]),
                rgb=np.stack([obs['observation'][cam]['rgb'] for cam in CAMERAS]))


def replay(path, out, number, baseline=None):
    meta = json.loads((path/'episode.json').read_text())
    actions = []
    for i, chunk in enumerate(meta['chunks']):
        a = np.load(path/f'actions-{i:02}.npy', allow_pickle=False)
        assert a.shape == (50,14) and np.isfinite(a).all() and hashlib.sha256(a.tobytes()).hexdigest() == chunk['actions_sha256']
        actions.extend(a[:chunk['delivered']])
    assert len(actions) == meta['targets'] and len(actions)>50
    cfg = environment_args(Path(os.environ['NATIVE_SOURCE']), 'demo_clean')
    cfg.update(policy_name='sidney_lerobot', ckpt_setting='sidney_release')
    task = place_phone_stand(); task.suc = task.test_num = 0
    task.setup_demo(now_ep_num=meta['index'], seed=meta['seed'], is_test=True, **cfg)
    folder = out/f"{meta['seed']}-{number}"; folder.mkdir()
    try:
        task.set_instruction(instruction=meta['instruction'])
        near = [float(c.near) for c in (task.cameras.left_camera, task.cameras.right_camera)]
        assert np.allclose(near, [.01, .01], atol=1e-9, rtol=0)
        maxima = {}; first = {}; query = {}; success_sequence=[]
        for t in range(len(actions)+1):
            if t:
                task.take_action(actions[t-1])
            values = snapshot(task, task.get_obs())
            success_sequence.append(bool(task.eval_success))
            if baseline is None:
                np.savez(folder/f'{t:04}.npz', **values)
            elif (baseline/f'{t:04}.npz').exists():
                with np.load(baseline/f'{t:04}.npz', allow_pickle=False) as original:
                    for key, value in values.items():
                        delta = float(np.abs(original[key].astype(np.float64)-value).max())
                        maxima[key] = max(delta, maxima.get(key,0.))
                        if delta and key not in first: first[key]=t
                        if t==50: query[key]=delta
            if task.eval_success: break
        return dict(seed=meta['seed'], pass_number=number, delivered=task.take_action_cnt,
                    success=bool(task.eval_success), wrist_near_m=near, success_sequence=success_sequence,
                    source_match=task.take_action_cnt==meta['targets'] and bool(task.eval_success)==meta['success'],
                    max_absolute_delta=maxima, first_mismatch=first, decision_absolute_delta=query), folder
    finally:
        task.close_env()


def main():
    out, inputs, scratch=map(Path, sys.argv[1:]); install_wrist_near(Camera,.01)
    report=dict(episodes=[],source_commit=os.environ['TASK_SOURCE'])
    for path in sorted(inputs.glob('demo_clean-*')):
        ref, folder = replay(path, scratch, 0)
        check, _ = replay(path, scratch, 1, folder)
        exact = (ref['source_match'] and check['source_match'] and ref['success_sequence']==check['success_sequence']
                 and set(check['decision_absolute_delta'])=={'public','poses','joints','rgb'} and not check['first_mismatch'])
        report['episodes'].append(dict(reference=ref, repeat=check, exact=exact)); save(out/'RESULT.json',report)
    assert len(report['episodes'])==2
    report['complete']=True; report['qualified']=all(r['exact'] for r in report['episodes']); save(out/'RESULT.json',report)


if __name__=='__main__': main()
