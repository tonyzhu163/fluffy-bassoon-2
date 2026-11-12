"""Matched native pi0.5 proposals with full-prefix replay, no learned rewards."""
import argparse
import hashlib
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np

from fluffy_bassoon.agent_sim.robocasa import CAMERAS, STATES
from fluffy_bassoon.robocasa.pairing import stable_counter_regions
from reference_eval import with_checkpoint_stats

SEEDS = (120700, 120701, 120702, 120703)
QUERIES = (100, 250)
BUDGET = 500


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def simple(value, depth=0):
    """Fingerprint actual numerical state without process-specific object reprs."""
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=value.shape,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, np.generic):
        return simple(value.item(), depth)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.random.Generator):
        return simple(value.bit_generator.state)
    if isinstance(value, np.random.RandomState):
        return simple(value.get_state())
    if depth < 4 and isinstance(value, (tuple, list)):
        return [simple(v, depth + 1) for v in value]
    if depth < 4 and isinstance(value, dict):
        return {str(k): simple(v, depth + 1) for k, v in value.items()}
    return {'class': type(value).__module__ + '.' + type(value).__name__}


def digest(value):
    return hashlib.sha256(json.dumps(simple(value), sort_keys=True, allow_nan=True).encode()).hexdigest()


def state(obs):
    return np.concatenate([obs[k] for k in STATES])


def identity(wrapped, obs):
    env = wrapped.unwrapped.env
    sim = env.sim
    controllers = {str(i): {k: simple(v.__dict__) for k, v in robot.part_controllers.items()}
                   for i, robot in enumerate(env.robots)}
    # Full replay reconstructs opaque objects; these checks audit visible numeric
    # state. Candidate-zero continuation is the behavioral replay qualification.
    return dict(physics=digest([sim.get_state().flatten(), sim.data.cam_xpos,
        sim.data.cam_xmat, sim.model.body_pos, sim.model.geom_pos]),
        public=digest([state(obs), *[obs[k] for k in CAMERAS], obs['annotation.human.task_description']]),
        controllers=digest(controllers), task=digest(env.__dict__),
        numpy_rng=digest(np.random.get_state()), python_rng=digest(random.getstate()))


def inputs(obs):
    from openpi_client import image_tools
    names = ('observation/image', 'observation/wrist_image', 'observation/right_image')
    result = {name: image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[k]), 224, 224)) for name, k in zip(names, CAMERAS)}
    return dict(result, **{'observation/state': state(obs), 'prompt': obs['annotation.human.task_description']})


def noise(seed, step, candidate=0):
    return np.random.default_rng(np.random.SeedSequence([seed, step, candidate, 61927])).standard_normal(
        (50, 32), dtype=np.float32)


def propose(policy, obs, seed, step, candidate=0):
    actions = np.asarray(policy.infer(inputs(obs), noise=noise(seed, step, candidate))['actions'])
    if actions.shape != (50, 12) or not np.isfinite(actions).all():
        raise ValueError('Invalid native Human300 proposal')
    return actions.copy()


def reset(seed):
    import gymnasium as gym
    import robocasa  # noqa: F401
    random.seed(seed)
    np.random.seed(seed)
    wrapped = gym.make('robocasa/OpenDrawer', split='pretrain', seed=seed)
    obs, _ = wrapped.reset(seed=seed)
    return wrapped, obs


def picture(path, obs, title):
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (768, 280), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), title + ' | left / wrist / right', fill='black')
    for i, key in enumerate(CAMERAS):
        sheet.paste(Image.fromarray(obs[key]), (256*i, 24))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    out.mkdir(exist_ok=False)
    started = time.time()

    def event(phase, **fields):
        row = dict(phase=phase, elapsed_seconds=time.time()-started, **fields)
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    import jax
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config
    from robocasa.utils.env_utils import convert_action
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    stable_counter_regions()
    stats = normalize.load(args.checkpoint/'assets')
    cfg = with_checkpoint_stats(config.get_config('pi05_pretrain_human300'), stats)
    event('loading', device=str(jax.devices()[0]))
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats,
                                                sample_kwargs={'num_steps': 10})
    event('loaded')
    results = []
    for seed in SEEDS:
        dest = out/str(seed)
        dest.mkdir()
        frames = dest/'public'
        frames.mkdir()
        wrapped, obs = reset(seed)
        commands, states, rewards, successes, anchors = [], [state(obs)], [], [], {}
        candidates, query_records, rngs = {}, {}, {}
        try:
            initial = identity(wrapped, obs)
            prompt = obs['annotation.human.task_description']
            for step in range(0, BUDGET, 5):
                anchors[step] = identity(wrapped, obs)
                rngs[step] = (random.getstate(), np.random.get_state())
                if any(q-50 <= step <= q for q in QUERIES) and step % 10 == 0:
                    picture(frames/f'{step:04d}.png', obs, f'Control {step}')
                ordinary = propose(policy, obs, seed, step)
                if step in QUERIES:
                    candidates[step] = np.stack([ordinary, propose(policy, obs, seed, step, 1)])
                    c = wrapped.unwrapped.env.robots[0].part_controllers['right']
                    query_records[step] = dict(seed=seed, query=step, prompt=prompt,
                        current_state={k: np.asarray(obs[k]).tolist() for k in STATES},
                        candidate_commands=candidates[step][:, :5].tolist(),
                        action_order=['dx','dy','dz','drx','dry','drz','gripper_close',
                                      'base0','base1','base2','torso','base_mode'],
                        controller=dict(frame=c.input_ref_frame, type=c.input_type,
                            output_min=np.asarray(c.output_min).tolist(), output_max=np.asarray(c.output_max).tolist()),
                        history_controls=list(range(step-50, step+1, 10)),
                        history_commands=np.asarray(commands[-50:]).tolist(),
                        history_states=np.asarray(states[-51:]).tolist(),
                        fork=anchors[step], budget=BUDGET, delivered_candidate_controls=5)
                for action in ordinary[:5]:
                    obs, reward, done, truncated, info = wrapped.step(convert_action(action))
                    commands.append(action.copy()); states.append(state(obs)); rewards.append(float(reward))
                    successes.append(bool(info['success']))
                    if done or truncated:
                        raise RuntimeError('Unexpected native boundary within pilot budget')
                if step % 50 == 0:
                    event('baseline', seed=seed, controls=step+5, success=successes[-1])
            anchors[BUDGET] = identity(wrapped, obs)
            rngs[BUDGET] = (random.getstate(), np.random.get_state())
            with (dest/'global-rng.pkl').open('wb') as stream:
                pickle.dump(rngs, stream)
            np.savez_compressed(dest/'baseline.npz', commands=commands, states=states, rewards=rewards,
                successes=successes, candidates=np.stack([candidates[q] for q in QUERIES]))
            write(dest/'baseline.json', dict(seed=seed, prompt=prompt, initial=initial,
                final=anchors[BUDGET], success_once=any(successes), final_success=successes[-1],
                native_return=sum(rewards), anchors=anchors))
            for q, value in query_records.items():
                write(dest/f'query-{q}.json', value)
            event('baseline_complete', seed=seed, success_once=any(successes), native_return=sum(rewards))
        finally:
            wrapped.close()
        for q in QUERIES:
            for candidate in (0, 1):
                wrapped, obs = reset(seed)
                branch_commands, branch_rewards, branch_success = [], [], []
                try:
                    assert identity(wrapped, obs) == initial, ('initial', seed, q, candidate)
                    for step, action in enumerate(commands[:q]):
                        obs, reward, done, truncated, info = wrapped.step(convert_action(action))
                        np.testing.assert_array_equal(state(obs), states[step+1])
                    # The action replay omits inference and image-writing calls.
                    # Restore their recorded global RNG effects as part of the
                    # complete fork, then require the same strict fingerprint.
                    random.setstate(rngs[q][0])
                    np.random.set_state(rngs[q][1])
                    fork = identity(wrapped, obs)
                    assert fork == anchors[q], ('fork', seed, q, candidate, fork, anchors[q])
                    for step in range(q, BUDGET, 5):
                        if candidate == 0:
                            assert identity(wrapped, obs) == anchors[step], ('continuation', seed, q, step)
                        action_chunk = candidates[q][candidate] if step == q else propose(policy, obs, seed, step)
                        if candidate == 0:
                            np.testing.assert_array_equal(action_chunk[:5], commands[step:step+5])
                        for action in action_chunk[:5]:
                            obs, reward, done, truncated, info = wrapped.step(convert_action(action))
                            branch_commands.append(action.copy()); branch_rewards.append(float(reward))
                            branch_success.append(bool(info['success']))
                            if done or truncated:
                                raise RuntimeError('Unexpected native branch boundary')
                        if step % 100 == 0:
                            event('branch', seed=seed, query=q, candidate=candidate, controls=step+5)
                    if candidate == 0:
                        assert identity(wrapped, obs) == anchors[BUDGET]
                        assert branch_rewards == rewards[q:] and branch_success == successes[q:]
                    np.savez_compressed(dest/f'branch-{q}-{candidate}.npz', commands=branch_commands,
                                        rewards=branch_rewards, successes=branch_success)
                    picture(dest/f'branch-{q}-{candidate}-final.png', obs, f'Final {seed} {q} {candidate}')
                    result = dict(seed=seed, query=q, candidate=candidate, fork_exact=True,
                        global_rng_restored=True,
                        prefix_state_checks=q, baseline_continuation_exact=candidate == 0,
                        success_before_query=any(successes[:q]), success_once=any(branch_success),
                        final_success=branch_success[-1], native_return=sum(branch_rewards),
                        controls=BUDGET-q)
                    write(dest/f'branch-{q}-{candidate}.json', result)
                    results.append(result)
                    write(out/'branches.json', results)
                    event('branch_complete', **result)
                finally:
                    wrapped.close()
    write(out/'complete.json', dict(scenes=len(SEEDS), queries=len(SEEDS)*len(QUERIES), branches=results,
                                   wall_seconds=time.time()-started))
    event('complete')


if __name__ == '__main__':
    main()
