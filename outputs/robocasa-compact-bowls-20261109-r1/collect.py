"""Frozen motor qualification; admit fresh feedback trials only after both skills work."""
import argparse
import collections
import itertools
import json
import time
from pathlib import Path

import numpy as np
from layout import FRUIT, reset_scene
from pilot import STATES, convert_action, digest, inputs, ou, stable_counter_regions, with_checkpoint_stats, write
from score_goals import measure

ARMS = ('left', 'right')


def rollout(policy, out, seed, arm):
    w, obs, layout = reset_scene(seed)
    try:
        assert all(layout['checks'].values()), layout
        env = w.unwrapped.env
        init = inputs(obs, '')
        physical = digest(env.sim.get_state().flatten(), env.sim.data.cam_xpos, env.sim.data.cam_xmat,
                          env.sim.model.body_pos, env.sim.model.geom_pos)
        view = digest(*(init[k] for k in init if k != 'prompt'))
        prompt = f'Put one fruit from the plate into the {arm} bowl. Release it and move your hand away.'
        rng = np.random.default_rng(seed + 501003)
        bowls = layout['bowls']
        queue = collections.deque()
        grasp, inside, cleared = None, [None, None], [None, None]
        native = False
        trace = []
        min_dist = [None, None]
        for t in range(1, 501):
            if not queue:
                result = policy.infer(inputs(obs, prompt), noise=rng.standard_normal((50,32), dtype=np.float32))
                a = np.asarray(result['actions'])
                assert a.shape == (50, 12) and np.isfinite(a).all()
                queue.extend(a[:5])
            a = queue.popleft()
            obs, _, done, trunc, info = w.step(convert_action(a))
            native |= bool(info['success'])
            trace.append(np.concatenate([a] + [obs[k] for k in STATES]))
            if grasp is None and any(env._check_grasp(env.robots[0].gripper['right'], env.objects[f]) for f in FRUIT):
                grasp = t
            hand = env.sim.data.site_xpos[env.robots[0].eef_site_id['right']]
            for i, b in enumerate(bowls):
                for f in FRUIT:
                    if ou.check_obj_in_receptacle(env, f, b):
                        if inside[i] is None:
                            inside[i] = t
                        d = float(np.linalg.norm(hand - env.sim.data.body_xpos[env.obj_body_id[f]]))
                        min_dist[i] = max(d, min_dist[i] or 0.)
                        if ou.gripper_obj_far(env, f) and cleared[i] is None:
                            cleared[i] = t
            if any(cleared) or done or trunc:
                break
        row = dict(seed=seed, arm=arm, controls=t, prompt=prompt, physical=physical, initial_public=view,
                   first_grasp=grasp, first_inside=inside, first_released=cleared,
                   max_hand_distance_while_inside_m=min_dist, native_success=native,
                   placed=[v is not None for v in cleared], layout=layout)
        np.savez_compressed(out / f'{seed}-{arm}.npz', action_post=np.array(trace))
        write(out / f'{seed}-{arm}.json', row)
        print('EPISODE', seed, arm, t, grasp, inside, cleared, flush=True)
        return row
    finally:
        w.close()


def panel(policy, out, seeds):
    rows = []
    for seed in seeds:
        scene = {a: rollout(policy, out, seed, a) for a in ARMS}
        assert len({(r['physical'],r['initial_public']) for r in scene.values()}) == 1, seed
        rows.append(scene)
        write(out / 'progress.json', dict(last=seed, unix=time.time()))
    return rows


def fit(dev):
    choices = list(itertools.product(ARMS, repeat=3))
    def optimized(options):
        return max(options, key=lambda p: (measure(dev, p)['total'], measure(dev, p)['support']))
    return optimized(choices), optimized([p for p in choices if p[1] == p[2]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--dev', nargs=4, required=True, type=int)
    p.add_argument('--fresh', nargs=4, required=True, type=int)
    args = p.parse_args()
    assert len(set(args.dev + args.fresh)) == 8
    import jax
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.shared import normalize
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(args.checkpoint / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    motor = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats,
                                                sample_kwargs={'num_steps': 10})
    dev = panel(motor, args.out, args.dev)
    counts = [[sum(s[a]['placed'][j] for s in dev) for j in range(2)] for a in ARMS]
    paired = sum(s['left']['placed'][0] and s['right']['placed'][1] for s in dev)
    passed = bool(paired and counts[0][0]+counts[1][1] > counts[0][1]+counts[1][0])
    report = dict(counts_request_by_outcome=counts, paired_both=paired, passed=passed, dev=args.dev)
    write(args.out / 'gate.json', report)
    if passed:
        dev_pairs = list(zip(dev[::2], dev[1::2]))
        active, static = fit(dev_pairs)
        write(args.out / 'frozen.json', dict(active=active, static=static))
        fresh = panel(motor, args.out, args.fresh)
        pairs = list(zip(fresh[::2], fresh[1::2]))
        report['fresh'] = dict(active=measure(pairs, active), memoryless=measure(pairs, static),
            reverse=measure(pairs, active, reverse=True), active_policy=active, static_policy=static)
    write(args.out / 'complete.json', report)


if __name__ == '__main__':
    main()
