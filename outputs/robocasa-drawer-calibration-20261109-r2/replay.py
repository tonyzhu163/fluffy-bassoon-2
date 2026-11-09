"""One native policy command tape per scene; exact physics-prefix check before the pulse."""
from collections import deque
import functools
import json
import os
from pathlib import Path
import random
import sys
import time

import gymnasium as gym
import imageio.v2 as imageio
import jax
import numpy as np
import robocasa
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'robocasa-drawer-calibration-20261109-r1'))
from pilot import (CAM, INPUT, STATE, HORIZON, MAGNITUDE, NPAIRS, config, convert_action, digest,
                   handle, inputs, normalize, policy_config, stable_counter_regions, with_checkpoint_stats, write)

GATE_M = .10


def reset(scene):
    random.seed(scene)
    np.random.seed(scene)
    env = gym.make('robocasa/OpenDrawer', split='pretrain', seed=scene)
    obs, _ = env.reset(seed=scene)
    return env, obs


def physics(raw):
    return digest(raw.sim.get_state().flatten(), raw.sim.data.cam_xpos)


def initial(raw, x):
    return digest(physics(raw), raw.sim.model.geom_pos, raw.sim.model.body_pos,
                  x['observation/state'], x['prompt'], *[x[k] for k in INPUT])


def geometry(raw, robot, centers):
    return (np.asarray(raw.sim.data.site_xpos[robot.eef_site_id['right']]) -
            np.asarray(raw.sim.data.geom_xpos)[centers].mean(axis=0))


def record_tape(policy, scene, noise_seed, path, log):
    env, obs = reset(scene)
    try:
        raw = env.unwrapped.env
        robot = raw.robots[0]
        _, centers = handle(raw)
        rng = np.random.default_rng(noise_seed)
        plan, actions, hashes = deque(), [], []
        anchor = initial(raw, inputs(obs))
        trigger = None
        for t in range(HORIZON):
            hashes.append(physics(raw))
            if trigger is None and np.linalg.norm(geometry(raw, robot, centers)) <= GATE_M:
                trigger = t
            if not plan:
                noise = rng.standard_normal((50,32), dtype=np.float32)
                chunk = np.asarray(policy.infer(inputs(obs), noise=noise)['actions'])
                assert chunk.shape == (50,12) and np.isfinite(chunk).all()
                plan.extend(chunk[:5])
            a = plan.popleft()
            actions.append(a.copy())
            obs, _, done, trunc, info = env.step(convert_action(a))
            if t % 100 == 0:
                log('tape_control', scene=scene, step=t, trigger=trigger)
            if info['success'] or done or trunc:
                break
        np.savez_compressed(path, scene=scene, noise_seed=noise_seed, actions=actions, physical_prefix=hashes,
                            initial=anchor, trigger=-1 if trigger is None else trigger, native_success=info['success'])
        log('tape_complete', scene=scene, controls=len(actions), trigger=trigger, native_success=bool(info['success']))
        return anchor, actions, hashes, trigger, bool(info['success'])
    finally:
        env.close()


def replay(out, ordinal, pair, role, scene, sign, comp, tape, log):
    anchor, actions, hashes, trigger, native_success = tape
    folder = out/f'trial-{ordinal:03d}'
    pub = folder/'public'
    pub.mkdir(parents=True)
    env, obs = reset(scene)
    try:
        raw = env.unwrapped.env
        robot = raw.robots[0]
        geoms, centers = handle(raw)
        controller = robot.part_controllers['right']
        x0 = inputs(obs)
        assert initial(raw, x0) == anchor, ('initial state or pixels differ', scene)
        event = {}
        active = False
        shift = MAGNITUDE * sign * (1 + comp)
        original = controller.compute_goal_pos

        @functools.wraps(original)
        def offset(*args, **kwargs):
            goal = original(*args, **kwargs)
            if not active:
                return goal
            assert not event
            base_y = np.array([0., 1., 0.])
            world_y = controller.origin_ori @ base_y
            assert controller.input_ref_frame in ('base', 'world')
            axis = base_y if controller.input_ref_frame == 'base' else world_y
            target = goal + shift * axis
            event.update(axis_world=world_y.tolist(), nominal=goal.tolist(), delivered=target.tolist(), shift_m=shift)
            return target
        controller.compute_goal_pos = offset
        states = [x0['observation/state']]
        rewards, samples = [], {}
        acquired = False
        close_t = None
        was_open = False
        prev_window = deque(maxlen=3)
        save_until = -1
        def save_view(t, x):
            for k, tag in zip(INPUT, ('left','wrist','right')):
                imageio.imwrite(pub/f'{t:04d}-{tag}.jpg', x[k], quality=90)
        trajectory_hashes = []
        for t, a in enumerate(actions):
            fp = physics(raw)
            trajectory_hashes.append(fp)
            if trigger is None or t <= trigger or shift == 0:
                assert fp == hashes[t], ('common command tape did not reproduce physics', scene, t, trigger, shift)
            active = trigger is not None and t == trigger
            close = bool(a[6] >= .5)
            rising = was_open and close
            was_open = not close
            if role == 'support':
                x = x0 if t == 0 else inputs(obs)
                prev_window.append((t, x))
                if rising:
                    save_until = t + 5
                    for i, old in prev_window:
                        save_view(i, old)
                if t % 5 == 0 or t <= save_until:
                    save_view(t, x)
            if active:
                samples['pre_trigger_distance_m'] = float(np.linalg.norm(geometry(raw,robot,centers)))
                assert samples['pre_trigger_distance_m'] <= GATE_M
            if trigger is not None and t >= trigger and close_t is None and rising:
                close_t = t
            obs, rew, done, trunc, info = env.step(convert_action(a))
            assert not active or event
            if event and (t-trigger in (0,1,4,8) or t == close_t):
                signed = float(geometry(raw,robot,centers) @ np.array(event['axis_world']))
                if t - trigger in (0,1,4,8):
                    samples[str(t-trigger+1)] = signed
                if t == close_t:
                    samples['at_closure'] = signed
            states.append(np.concatenate([obs[k] for k in STATE]))
            acquired |= bool(raw._check_grasp(robot.gripper['right'],geoms))
            rewards.append(float(rew))
            if t % 100 == 0:
                log('replay_control', ordinal=ordinal, pair=pair, role=role, sign=sign, comp=comp, step=t)
            if info['success'] or done or trunc:
                break
        success = bool(info['success'])
        if shift == 0 or trigger is None:
            assert success == native_success and len(rewards) == len(actions)
        if role == 'support':
            save_view(t+1, inputs(obs))
        np.savez_compressed(pub/'trajectory.npz', requested_action=np.array(actions[:t+1]), measured_state=states,
                            native_reward=rewards, prompt=x0['prompt'])
        row = dict(ordinal=ordinal,pair=pair,role=role,sign=sign,compensation=comp,scene_seed=scene,
                   paired_initial_sha256=anchor,tape_sha256=digest(np.array(actions)),controls=t+1,
                   approach_control=trigger, closure_control=close_t, success=success,acquired_handle=acquired,
                   signed_handle_miss_m=samples,physical_target=event,
                   trajectory_sha256=digest(*trajectory_hashes,physics(raw)))
        write(out/f'episode-{ordinal:03d}.json',row)
        log('episode_complete',ordinal=ordinal,pair=pair,role=role,controls=t+1,trigger=trigger,
            closure=close_t,success=success,acquired=acquired)
        return row
    finally:
        env.close()


def main():
    assert os.uname().nodename.startswith('job-')
    out, ckpt = map(Path,sys.argv[1:])
    def log(phase,**kwargs):
        row = dict(phase=phase,unix=time.time(),**kwargs)
        with (out/'events.jsonl').open('a') as fp:
            fp.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    assert robocasa.__version__ == '1.0.1' and len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    log('loading_frozen_motor')
    stats = normalize.load(ckpt/'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'),stats)
    motor = policy_config.create_trained_policy(cfg,ckpt,sample_kwargs={'num_steps':10},norm_stats=stats)
    rows, ntape = [], 0
    for pair in range(NPAIRS):
        initial_hashes = []
        for role in ('support','query'):
            delta = 2 * pair + int(role=='query')
            scene = 910210 + delta
            tape = record_tape(motor,scene,1700210+delta,out/f'tape-{pair}-{role}.npz',log)
            ntape += len(tape[1])
            initial_hashes.append(tape[0])
            assert len(set(initial_hashes)) == len(initial_hashes)
            for comp in ([0] if role=='support' else [-1,0,1]):
                rr = [replay(out,len(rows)+j,pair,role,scene,s,comp,tape,log) for j,s in enumerate((-1,1))]
                rows.extend(rr)
                if comp == -1:
                    assert rr[0]['trajectory_sha256'] == rr[1]['trajectory_sha256'], 'inverse parity failed'
        log('pair_complete',pair=pair,scored_episodes=len(rows))
    write(out/'RESULT.json',dict(episodes=len(rows),pairs=NPAIRS,tape_controls=ntape,
                                 replay_controls=sum(r['controls'] for r in rows),training_updates=0))
    log('complete',episodes=len(rows))


if __name__=='__main__':
    main()
