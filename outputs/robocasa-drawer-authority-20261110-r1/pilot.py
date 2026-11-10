"""Check a bounded decoded-action correction using matched, closed-loop branches."""
import json
from pathlib import Path
import sys
import time

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'robocasa-drawer-calibration-20261109-r2'))
from replay import (GATE_M, HORIZON, INPUT, config, convert_action, digest, geometry,
                    handle, inputs, normalize, physics, policy_config, reset,
                    stable_counter_regions, with_checkpoint_stats)


def public_hash(obs):
    x = inputs(obs)
    return digest(x['observation/state'], x['prompt'], *(x[k] for k in INPUT))


class RestoreMismatch(Exception):
    pass


def rollout(policy, out, scene, arm, *, tape=None, bias=0., residual=0., repeat=False):
    env, obs = reset(scene)
    try:
        raw = env.unwrapped.env
        robot = raw.robots[0]
        controller = robot.part_controllers['right']
        assert controller.input_ref_frame == 'base' and controller.input_type == 'delta'
        gain = ((controller.output_max - controller.output_min) /
                (controller.input_max - controller.input_min))[1]
        assert 0 < gain < 1
        names, centers = handle(raw)
        nominal_goal = controller.compute_goal_pos
        decision = None if tape is None else tape['decision']
        goal_calls = 0
        active = False
        def shifted(*args, **kwargs):
            nonlocal goal_calls
            goal = nominal_goal(*args, **kwargs)
            if active:
                goal_calls += 1
                if bias:
                    goal = goal.copy()
                    goal[1] += bias
            return goal
        controller.compute_goal_pos = shifted
        actions, hashes, observations, misses, realized = [], [], [], [], []
        noise = np.random.default_rng(scene + 700000).standard_normal((HORIZON // 5, 50, 32), dtype=np.float32)
        first_grasp = None
        decision_obs = None
        chunk = None
        for t in range(HORIZON):
            fp = physics(raw)
            hashes.append(fp)
            dist = geometry(raw, robot, centers)
            if tape is None and decision is None and first_grasp is None and t % 5 == 0 and np.linalg.norm(dist) <= GATE_M:
                decision = t
            if tape is not None and (repeat or t <= decision) and fp != tape['hashes'][t]:
                raise RestoreMismatch(f'physics at control {t}')
            if t == decision:
                decision_obs = public_hash(obs)
                if tape is not None and decision_obs != tape['decision_obs']:
                    raise RestoreMismatch(f'policy inputs at decision {t}')
            if tape is not None and (repeat or t < decision):
                a = tape['actions'][t].copy()
            else:
                if t % 5 == 0:
                    chunk = np.asarray(policy.infer(inputs(obs), noise=noise[t // 5])['actions'])
                    assert chunk.shape == (50, 12) and np.isfinite(chunk).all()
                a = chunk[t % 5].copy()
            active = decision is not None and decision <= t < decision + 5
            if active:
                old = a[1].item()
                if residual:
                    a[1] = np.clip(old + residual / gain, controller.input_min[1], controller.input_max[1])
                realized.append(float(gain * (np.clip(a[1], controller.input_min[1], controller.input_max[1]) - np.clip(old, controller.input_min[1], controller.input_max[1]))))
            actions.append(a)
            obs, reward, done, truncated, info = env.step(convert_action(a))
            success = bool(info['success'])
            if first_grasp is None and raw._check_grasp(robot.gripper['right'], names):
                first_grasp = t + 1
            if decision is not None and t - decision in (0, 4, 9, 19):
                signed = geometry(raw, robot, centers) @ controller.origin_ori[:, 1]
                misses.append([t + 1, float(signed)])
            if t % 5 == 4 or success or done or truncated:
                observations.append([t + 1, float(reward), *np.asarray(obs['state.end_effector_position_relative']).tolist()])
            if success or done or truncated:
                break
        assert goal_calls == len(realized)
        if repeat and (success != tape['success'] or len(actions) != len(tape['actions']) or physics(raw) != tape['terminal']):
            raise RestoreMismatch('terminal state, length or success')
        data = dict(actions=np.array(actions), hashes=hashes, decision=decision, decision_obs=decision_obs,
                    terminal=physics(raw), success=success)
        np.savez_compressed(out/f'{scene}-{arm}.npz', actions=data['actions'], physics=hashes,
                            public_eef_reward=observations)
        row = dict(scene=scene, arm=arm, bias_m=bias, requested_residual_m=residual,
                   applied_residual_m=realized, controls=t+1, success=success, first_grasp=first_grasp,
                   decision=decision, decision_observation_sha256=decision_obs, signed_miss_m=misses,
                   matched_pre_action_states=0 if tape is None else (t+1 if repeat else decision+1),
                   terminal_sha256=data['terminal'])
        return row, data
    finally:
        env.close()


def main():
    out, ckpt = map(Path, sys.argv[1:])
    def log(kind, **kwargs):
        row = dict(kind=kind, unix=time.time(), **kwargs)
        with (out/'events.jsonl').open('a') as f:
            print(json.dumps(row), file=f, flush=True)
        print(json.dumps(row), flush=True)
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(ckpt/'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    policy = policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={'num_steps':10}, norm_stats=stats)
    completed = 0
    for scene in range(982100, 982108):
        log('screening', scene=scene)
        row, tape = rollout(policy, out, scene, 'nominal')
        log('episode', **row)
        if not row['success'] or row['decision'] is None:
            continue
        try:
            row, _ = rollout(policy, out, scene, 'repeat', tape=tape, repeat=True)
            log('episode', **row)
            for sign in (-1, 1):
                for mm in (0, -40, -20, 20, 40):
                    arm = f'b{sign:+d}-r{mm:+d}'
                    row, _ = rollout(policy, out, scene, arm, tape=tape, bias=sign*.04, residual=mm/1000)
                    log('episode', **row)
            completed += 1
        except RestoreMismatch as exc:
            log('invalid_restore', scene=scene, reason=str(exc))
        if completed == 3:
            break
    log('complete', qualified_scenes=completed)


if __name__ == '__main__':
    main()
