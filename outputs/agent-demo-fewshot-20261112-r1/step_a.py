"""Step A, compact-bowl carry: can steering change what frozen π0.5 does after a grasp?

Per scene, one frozen π0.5 pickup under a neutral prompt is recorded up to the
first grasp. Every arm then replays that exact prefix and continues differently:

* scripted  the privileged carry of the compact-bowl run; its executed commands
            become the same-scene target tape for the requested side
* native    frozen π0.5 with the side named in the prompt, no steering
* guide w   the same, steered toward the tape on translation and gripper
* select n  the same, keeping the nearest of n unguided samples at each replan

The tape is indexed by controls since the grasp (open loop). Policy arms share
one noise stream, so native and guide differ only by the steering.
"""
import argparse
import collections
import time
from pathlib import Path

import numpy as np
from layout import FRUIT, reset_scene
from pilot import STATES, convert_action, digest, inputs, ou, stable_counter_regions, with_checkpoint_stats, write

from fluffy_bassoon.robocasa.guided_sampling import Motor

SIDES = ('left', 'right')
HORIZON, EXECUTE = 500, 5
NEUTRAL = 'Put one fruit from the plate into a bowl.'


def side_prompt(side):
    return f'Put one fruit from the plate into the {side} bowl. Release it and move your hand away.'


def physical(env):
    return digest(env.sim.get_state().flatten(), env.sim.data.cam_xpos, env.sim.data.cam_xmat,
                  env.sim.model.body_pos, env.sim.model.geom_pos)


def held_fruit(env):
    gripper = env.robots[0].gripper['right']
    return [f for f in FRUIT if env._check_grasp(gripper, env.objects[f])]


def placed(env, bowls):
    return [bool(any(ou.check_obj_in_receptacle(env, f, b) and ou.gripper_obj_far(env, f) for f in FRUIT))
            for b in bowls]


def state_of(obs):
    return np.concatenate([obs[k] for k in STATES])


def pickup(motor, seed):
    """Frozen π0.5 under the neutral prompt until the first grasp."""
    w, obs, layout = reset_scene(seed)
    try:
        assert all(layout['checks'].values()), layout
        env = w.unwrapped.env
        start = physical(env)
        rng = np.random.default_rng(seed + 501003)
        zero = np.zeros((motor.horizon, motor.width), np.float32)
        queue, tape = collections.deque(), []
        for n in range(1, HORIZON + 1):
            if not queue:
                chunk, _ = motor.guide(inputs(obs, NEUTRAL), rng.standard_normal((50, 32), dtype=np.float32),
                                       zero, zero, 0.)
                queue.extend(chunk[:EXECUTE])
            a = queue.popleft()
            obs, _, done, trunc, _ = w.step(convert_action(a))
            tape.append(np.concatenate([a, state_of(obs)]))
            if held_fruit(env):
                return dict(seed=seed, physical=start, first_grasp=n, layout=layout), np.array(tape)
            if done or trunc:
                break
        return dict(seed=seed, physical=start, first_grasp=None, layout=layout), np.array(tape)
    finally:
        w.close()


def replay(seed, prefix, tape):
    """Reset the scene and replay the pickup, checking every public state."""
    w, obs, layout = reset_scene(seed)
    try:
        env = w.unwrapped.env
        assert physical(env) == prefix['physical'] and layout['bowls'] == prefix['layout']['bowls']
        for row in tape:
            obs, *_ = w.step(convert_action(row[:12]))
            np.testing.assert_array_equal(state_of(obs), row[12:])
        assert held_fruit(env)
        return w, obs, env
    except BaseException:
        w.close()
        raise


def scripted(seed, prefix, tape, side):
    """The compact-bowl run's privileged carry (fork_grasp.py), recording executed commands."""
    w, _, env = replay(seed, prefix, tape)
    try:
        robot = env.robots[0]
        c = robot.part_controllers['right']
        assert c.input_ref_frame == 'base' and c.input_type == 'delta'
        hand = lambda: env.sim.data.site_xpos[robot.eef_site_id['right']].copy()
        pos = lambda k: env.sim.data.body_xpos[env.obj_body_id[k]].copy()
        bowls, fruit, n = prefix['layout']['bowls'], held_fruit(env)[0], len(tape)
        start, b = hand(), pos(bowls[SIDES.index(side)])
        offset = start - pos(fruit)
        high = max(start[2] + .16, b[2] + .18 + offset[2])
        over = np.array([*(b[:2] + offset[:2]), high])
        low = over.copy(); low[2] = b[2] + .075 + offset[2]
        away = over.copy(); away[:2] -= .12 * c.origin_ori[:2, 0]; away[2] += .12
        lift = start.copy(); lift[2] = high
        phases = [('secure', start, 1., 8), ('raise', lift, 1., 45), ('carry', over, 1., 65),
                  ('lower', low, 1., 40), ('open', low, 0., 12), ('retreat', away, 0., 65)]
        commands, result, done = [], [False, False], False
        for phase, target, close, length in phases:
            for _ in range(min(length, HORIZON - n)):
                a = np.zeros(12)
                a[:3] = np.clip(c.origin_ori.T @ (target - hand()) / .05, -.55, .55)
                a[6] = close
                _, _, done, trunc, _ = w.step(convert_action(a)); n += 1
                commands.append(a)
                result = placed(env, bowls)
                done = any(result) or done or trunc
                if done or (phase in ('raise', 'carry', 'lower', 'retreat')
                            and np.linalg.norm(target - hand()) < .015):
                    break
            if done or n >= HORIZON:
                break
        return dict(arm='scripted', placed=result, controls=n), np.array(commands, np.float32)
    finally:
        w.close()


def window(commands, k):
    """The next 50 tape commands after ``k`` controls since the grasp, with validity."""
    out, valid = np.zeros((50, 12), np.float32), np.zeros(50, bool)
    part = commands[k:k + 50]
    out[:len(part)], valid[:len(part)] = part, True
    return out, valid


def steered(motor, seed, prefix, tape, side, commands, *, weight=0., samples=0):
    w, obs, env = replay(seed, prefix, tape)
    try:
        bowls, n, k = prefix['layout']['bowls'], len(tape), 0
        rng = np.random.default_rng(seed + 777)
        extra = np.random.default_rng(seed + 778)
        queue, executed, gaps, spreads = collections.deque(), [], [], []
        result = [False, False]
        while n < HORIZON:
            if not queue:
                target, mask = motor.encode(*window(commands, k))
                noise = rng.standard_normal((50, 32), dtype=np.float32)
                if samples:
                    more = extra.standard_normal((samples - 1, 50, 32), dtype=np.float32)
                    chunk, gap, all_gaps = motor.select(inputs(obs, side_prompt(side)),
                                                        np.concatenate([noise[None], more]), target, mask)
                    spreads.append(all_gaps)
                else:
                    chunk, gap = motor.guide(inputs(obs, side_prompt(side)), noise, target, mask, weight)
                gaps.append(gap)
                queue.extend(chunk[:EXECUTE])
            a = queue.popleft()
            obs, _, done, trunc, _ = w.step(convert_action(a)); n += 1; k += 1
            executed.append(a)
            result = placed(env, bowls)
            if any(result) or done or trunc:
                break
        row = dict(placed=result, controls=n, still_holding=bool(held_fruit(env)),
                   target_gap_per_replan=gaps)
        if samples:
            row['sample_gaps_per_replan'] = spreads
        return row, np.array(executed, np.float32)
    finally:
        w.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--seeds', nargs='+', required=True, type=int)
    p.add_argument('--weights', nargs='+', type=float, default=[.25, .5, 1.])
    p.add_argument('--samples', type=int, default=8)
    args = p.parse_args()
    import jax
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(args.checkpoint / 'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats,
                                                 sample_kwargs={'num_steps': 10})
    motor = Motor(policy, stats)

    w, obs, _ = reset_scene(args.seeds[0])
    try:
        gap = motor.parity(inputs(obs, NEUTRAL),
                           np.random.default_rng(0).standard_normal((50, 32), dtype=np.float32))
    finally:
        w.close()
    write(args.out / 'parity.json', dict(max_abs_decoded_action_gap=gap, seed=args.seeds[0]))
    # Same computation as the pinned sampler; anything above float noise means the copied loop differs.
    assert gap < 1e-4, gap

    rows = []
    for seed in args.seeds:
        prefix, tape = pickup(motor, seed)
        np.savez_compressed(args.out / f'{seed}-pickup.npz', action_post=tape)
        write(args.out / f'{seed}-pickup.json', prefix)
        print('PICKUP', seed, prefix['first_grasp'], flush=True)
        if prefix['first_grasp'] is None:
            rows.append(dict(seed=seed, first_grasp=None))
            continue
        for side in SIDES:
            oracle, commands = scripted(seed, prefix, tape, side)
            want = SIDES.index(side)
            arms = [oracle]
            # Without a successful same-scene carry there is no known-good target to steer toward.
            if oracle['placed'][want] and not oracle['placed'][1 - want]:
                plan = [('native', dict())] + [(f'guide-{x:g}', dict(weight=x)) for x in args.weights]
                plan.append((f'select-{args.samples}', dict(samples=args.samples)))
                for name, kwargs in plan:
                    row, executed = steered(motor, seed, prefix, tape, side, commands, **kwargs)
                    np.savez_compressed(args.out / f'{seed}-{side}-{name}.npz', actions=executed)
                    arms.append(dict(arm=name, **row))
                    print('ARM', seed, side, name, row['placed'], row['controls'], flush=True)
            np.savez_compressed(args.out / f'{seed}-{side}-scripted.npz', actions=commands)
            rows.append(dict(seed=seed, side=side, first_grasp=prefix['first_grasp'], arms=arms))
            write(args.out / 'rows.json', rows)
            write(args.out / 'progress.json', dict(seed=seed, side=side, unix=time.time()))
    write(args.out / 'complete.json', dict(seeds=args.seeds, weights=args.weights, samples=args.samples,
                                           parity=gap, rows=rows))


if __name__ == '__main__':
    main()
