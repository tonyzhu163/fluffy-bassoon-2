"""Paired, goal-independent native rollouts of the released OpenPI motor."""
import argparse
import hashlib
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys
import time

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi_client import image_tools

HERE = Path(__file__).resolve().parent
LEGACY = Path('/mowenjie/fluffy-bassoon')
ROOT = Path('/mowenjie/fluffy-bassoon-2')
CKPT = LEGACY / 'inputs/pi05-libero-openpi-20261027-r1'
SRC = LEGACY / 'third_party/openpi-215abfb'
SETUP = ROOT / 'setups/libero-meta-learning-20261117-r1'
PRO = ROOT / 'sources/libero-pro2-e5500c69'


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def observation(raw, prompt):
    images = [image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(raw[k][::-1, ::-1]), 224, 224)) for k in
        ('agentview_image', 'robot0_eye_in_hand_image')]
    quat = raw['robot0_eef_quat_xyzw'].copy()
    w = np.clip(quat[3], -1, 1)
    den = np.sqrt(1-w*w)
    rot = quat[:3] * 2*np.arccos(w)/den if den else np.zeros(3)
    return {'observation/image':images[0], 'observation/wrist_image':images[1],
            'observation/state':np.concatenate((raw['robot0_eef_pos'], rot, raw['robot0_gripper_qpos'])),
            'prompt':prompt}


def call(pipe, op, arg):
    pipe.send((op, arg))
    ok, result = pipe.recv()
    if not ok:
        raise RuntimeError(result)
    return result


def rollout(policy, pipe, root, scene, variant, goal, gain, *, paced=False):
    suffix = 'paced' if paced else f'gain{gain:g}'
    output = root / f's{scene}-{variant}-g{goal}-{suffix}'
    output.mkdir()
    raw, physics, prompts = call(pipe, 'reset', (scene, variant))
    initial = {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in raw.items()}
    rng = np.random.default_rng(18183300 + scene)
    states, frames, sent, hits = [], [], [], []
    decision_steps = []
    while len(sent) < 300:
        item = observation(raw, prompts[goal])
        decision_steps.append(len(sent))
        length = 10 if paced else 5
        chunk = policy.infer(item, noise=rng.standard_normal((10,32)).astype(np.float32))['actions'][:length].copy()
        assert chunk.shape == (length,7) and np.isfinite(chunk).all()
        chunk[:, :3] *= gain
        if paced:
            wait = np.zeros((14, 7), dtype=chunk.dtype)
            wait[:, -1] = chunk[-1, -1]
            chunk = np.concatenate((chunk, wait))[:300-len(sent)]
        frames.append(np.stack([item['observation/image'],item['observation/wrist_image']]))
        raw, state, success = call(pipe, 'chunk', chunk)
        sent.extend(chunk)
        states.extend(state)
        hits.extend(success)
    hits = np.asarray(hits, dtype=bool)
    first = [int(np.flatnonzero(col)[0])+1 if col.any() else None for col in hits.T]
    np.savez_compressed(output/'public.npz', frames=frames, decision_steps=decision_steps, state_after=states, issued=sent,
                        final_images=np.stack([observation(raw,'')[k] for k in ('observation/image','observation/wrist_image')]))
    np.save(output/'evaluator-predicates.npy',hits)
    record = dict(scene=scene,variant=variant,goal=goal,prompt=prompts[goal],gain=gain,paced=paced,decisions=len(frames),
                  initial_physics=physics,initial_public=initial,first=first,
                  intended=first[goal] is not None, simultaneous=bool(hits.all(axis=1).any()),controls=len(hits))
    write(output/'result.json',record)
    print(output.name, 'first hits', first, flush=True)
    return record


def live_trials(policy, pipe, out):
    from readout import estimate_lag

    supports = json.loads((HERE/'nominal02-evidence.json').read_text())['branches']
    anchors = [float(np.median([s['public_lag'] for s in supports if s['variant']==v]))
               for v in ('BASE','S20')]
    options = np.random.default_rng(18182401).permutation([0,0,1,1]).tolist()
    rows = []
    for idx, option in enumerate(options):
        for variant in ('BASE','S20'):
            support_scene, query_scene = 20+idx, 24+idx
            block = out / f'b{idx}-{variant}'
            block.mkdir()
            support = rollout(policy, pipe, block, support_scene, variant, option, 1.)
            with np.load(block/f's{support_scene}-{variant}-g{option}-gain1/public.npz') as data:
                lag = estimate_lag(data)
            paced = abs(lag-anchors[1]) < abs(lag-anchors[0])
            for assigned in (0,1):
                reward = support['first'][assigned] is not None
                chosen = option if reward else 1-option
                trial = block / f'goal{assigned}'
                trial.mkdir()
                write(trial/'choice.json',dict(option=option,reward=reward,lag=lag,anchors=anchors,
                                                chosen=chosen,paced=paced))
                outcomes = []
                for is_paced in (paced, not paced):
                    outcomes.append(rollout(policy, pipe, trial, query_scene, variant, chosen, 1., paced=is_paced))
                a,b = outcomes
                assert a['initial_physics']==b['initial_physics'] and a['initial_public']==b['initial_public']
                fixed = {x['paced']: x['first'][assigned] is not None for x in outcomes}
                rows.append(dict(support_scene=support_scene,query_scene=query_scene,variant=variant,assigned=assigned,
                                 option=option,reward=reward,lag=lag,paced=paced,chosen=chosen,
                                 matched=fixed[paced],default=fixed[False],always_paced=fixed[True]))
                write(out/'trials.json',rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--phase', choices=('nominal','paired','pacing','live'), required=True)
    args = parser.parse_args()
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    assert subprocess.check_output(['git','-C',SRC,'rev-parse','HEAD'],text=True).strip() == '215abfb217dbac7d5f1273282331b9b1866c0479'
    assert subprocess.check_output(['git','-C',PRO,'rev-parse','HEAD'],text=True).strip() == 'e5500c690c1498eca1b7050350654964190cdf70'
    out = args.output
    out.mkdir()
    start = time.time()
    write(out/'motor.json',dict(checkpoint=str(CKPT),ready=json.loads((CKPT/'READY.json').read_text())['checkpoint'],
         norm_sha256=hashlib.sha256((CKPT/'assets/physical-intelligence/libero/norm_stats.json').read_bytes()).hexdigest(),
         openpi='215abfb217dbac7d5f1273282331b9b1866c0479',nominal_replan_steps=5,device=str(jax.devices()[0])))
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'),str(CKPT))
    pipe, peer = Pipe()
    # The pinned LIBERO release stores its NumPy reset bank using torch's legacy format.
    sim_env = dict(os.environ,TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',PYTHONPATH=f'{PRO}:{SETUP}/simulator-source',
                   LIBERO_CONFIG_PATH=str(SETUP/'libero-config'), LIBERO_PRO_ROOT=str(SETUP/'simulator-source'))
    proc = subprocess.Popen([SETUP/'env/bin/python','-Bu',HERE/'sim.py',str(peer.fileno())],
                             pass_fds=(peer.fileno(),),env=sim_env)
    peer.close()
    records = []
    try:
        if args.phase == 'live':
            trials = live_trials(policy, pipe, out)
            write(out/'complete.json',dict(trials=trials,episodes=trials*2+trials//2,elapsed_seconds=time.time()-start))
            return
        scenes = {'nominal': (10,11), 'paired': (12,13,14,15), 'pacing': (16,17,18,19)}[args.phase]
        options = {'nominal': [(1.,False)], 'paired': [(1.,False),(.5,False)],
                   'pacing': [(1.,False),(1.,True)]}[args.phase]
        for scene in scenes:
            for variant in ('BASE','S20'):
                for gain, paced in options:
                    for goal in range(2):
                        r = rollout(policy, pipe, out, scene, variant, goal, gain, paced=paced)
                        for prev in records:
                            if (prev['scene'],prev['variant'])==(scene,variant):
                                assert prev['initial_physics']==r['initial_physics'] and prev['initial_public']==r['initial_public'], 'branch reset mismatch'
                        records.append(r)
                        write(out/'progress.json',records)
        write(out/'complete.json',dict(episodes=len(records),elapsed_seconds=time.time()-start))
    finally:
        pipe.close()
        proc.wait(timeout=30)


if __name__=='__main__':
    main()
