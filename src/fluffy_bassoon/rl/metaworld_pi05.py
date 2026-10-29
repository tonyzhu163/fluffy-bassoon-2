"""Released RLinf pi0.5 MetaWorld SFT under fixed noise arms and discrete goals (CPU pieces).

outputs/libero-noise-icrl-20261029-r1/PLAN.md, Meta-World companion, with the released
motor `RLinf/RLinf-Pi05-MetaWorld-SFT` (few-shot SFT on lerobot/metaworld_mt50; RLinf
reports 43.8% MT50 success). Conventions follow RLinf e1a82eb's
toolkits/standalone_eval_scripts/openpi/metaworld_eval.py: camera 2 moved to
(0.75, 0.075, 0.7), 480 px image rotated 180 degrees, state = obs[:4], RLinf's task
prompt, 15 settle steps with a zero action, 5-action chunks from a 5-step horizon,
5 flow steps, 160 steps.

A condition is one of four discrete goals at least 0.1 apart (twice the 0.05 success
radius), so a success identifies the goal. With the goal marker visible this measures
headroom on a seen task. With it hidden (site alpha 0) the goal is invisible to the
motor: the state is the hand only, ML1 zeroes the goal in obs, and the prompt never
names it. Then noise can only find the goal if the motor's actions are multimodal over
goals, which `goal_coverage` checks. Each scene fixes the object; Meta-World's reset
loops forever unless it is at least 0.15 from the goal in x-y.
"""
from collections import defaultdict
import hashlib

import numpy as np


TASKS = {
    "reach-v3": dict(
        prompt="Reach a goal position",
        goals=((-.1, .8, .15), (.1, .8, .15), (-.1, .9, .25), (.1, .9, .25)),
        object_low=(-.1, .6, .02), object_high=(.1, .7, .02)),
    "push-v3": dict(
        prompt="Push the puck to a goal",
        goals=((-.1, .8, .01), (.1, .8, .01), (-.1, .9, .01), (.1, .9, .01)),
        object_low=(-.1, .6, .02), object_high=(.1, .7, .02)),
}
CAMERA_ID, CAMERA_POS, RESOLUTION = 2, (.75, .075, .7), 480
SETTLE, CHUNK, STEPS, FLOW_STEPS = 15, 5, 160, 5
NOISE_SHAPE = (5, 32)  # pi05_metaworld action_horizon x model action_dim
STATE_DIM, ACTION_DIM = 4, 4
MIN_GOAL_SEPARATION, MIN_OBJECT_GOAL_XY = .1, .15
PIXEL_TOLERANCE = 8
NATIVE = -1


def task_spec(name):
    if name not in TASKS:
        raise ValueError(f"Task is one of {sorted(TASKS)}")
    spec = TASKS[name]
    goals = np.asarray(spec["goals"], dtype=np.float64)
    distances = np.linalg.norm(goals[:, None] - goals[None], axis=-1)[np.triu_indices(len(goals), 1)]
    if distances.min() < MIN_GOAL_SEPARATION - 1e-9:
        raise ValueError("Goals must be at least twice the success radius apart")
    return spec, goals


def scenes(name, count, seed):
    """Object positions far enough from every goal for Meta-World's reset."""
    spec, goals = task_spec(name)
    rng = np.random.default_rng(seed)
    chosen = []
    for _ in range(1000*count):
        position = rng.uniform(spec["object_low"], spec["object_high"])
        if np.linalg.norm(goals[:, :2] - position[:2], axis=-1).min() >= MIN_OBJECT_GOAL_XY:
            chosen.append(position)
            if len(chosen) == count:
                return np.asarray(chosen)
    raise RuntimeError("Object range cannot keep its distance from the goals")


def arm_noise(seeds):
    """One fixed initial-noise array per arm seed, [arms, *NOISE_SHAPE] float32."""
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**63 for seed in seeds):
        raise ValueError("Require integer arm seeds in [0, 2**63)")
    return np.stack([np.random.default_rng(seed).standard_normal(NOISE_SHAPE).astype(np.float32)
                     for seed in seeds])


def make_env(name, *, goal_visible, seed=0):
    """An ML1 env rendering like RLinf's Meta-World evaluation."""
    import metaworld
    benchmark = metaworld.ML1(name, seed=seed)
    env = benchmark.train_classes[name](render_mode="rgb_array", camera_id=CAMERA_ID)
    env.set_task(benchmark.train_tasks[0])
    env._partially_observable = True
    env.model.cam_pos[CAMERA_ID] = CAMERA_POS
    if not goal_visible:
        env.model.site_rgba[env.model.site("goal").id, 3] = 0.0
    return env


def reset(env, name, goal, position):
    """Reset to one scene and goal, then settle; the vector is (object xyz, goal xyz)."""
    env._last_rand_vec = np.concatenate([position, goal]).astype(np.float64)
    env._freeze_rand_vec = True
    obs, _ = env.reset()
    target = np.asarray(env._target_pos)
    if not np.allclose(target[:2] if name == "push-v3" else target, goal[:2] if name == "push-v3" else goal):
        raise RuntimeError("Meta-World did not place the requested goal")
    for _ in range(SETTLE):
        obs, *_ = env.step(np.zeros(ACTION_DIM))
    if np.any(obs[-3:] != 0):
        raise RuntimeError("The goal leaked into the observation")
    return obs


def policy_input(env, obs, prompt):
    image = np.ascontiguousarray(env.render()[::-1, ::-1])
    return {"observation/image": image, "observation/state": np.asarray(obs[:STATE_DIM], dtype=np.float32),
            "prompt": prompt}


def goal_visibility(env, name, position, tolerance=PIXEL_TOLERANCE):
    """Pixels that differ by more than `tolerance` between goals from one scene (0 when hidden).

    Successive resets of one scene and goal differ by one intensity level in a few
    pixels (solver warm start), so exact equality would fail without a leak; the
    visible marker changes over a thousand pixels by about 200 levels.
    """
    _, goals = task_spec(name)
    images = []
    for goal in goals:
        reset(env, name, goal, position)
        images.append(env.render().astype(np.int16))
    return int(max((np.abs(image - images[0]) > tolerance).sum() for image in images))


def rollout(env, act, name, goal, position, steps=STEPS):
    """One episode under `act(policy_input) -> [>= CHUNK, >= ACTION_DIM]`.

    Returns (success_once, sha256 of the issued actions, issued [chunks, CHUNK, 4]).
    """
    spec, _ = task_spec(name)
    obs = reset(env, name, goal, position)
    issued, success, done, step = [], False, False, 0
    while not done:
        chunk = np.asarray(act(policy_input(env, obs, spec["prompt"])), dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] < CHUNK or chunk.shape[1] < ACTION_DIM:
            raise ValueError("Policy chunk is shorter than the executed chunk")
        chunk = chunk[:CHUNK, :ACTION_DIM]
        issued.append(chunk)
        for action in chunk:
            obs, _, _, truncated, info = env.step(action)
            step += 1
            success = bool(info["success"])
            done = success or truncated or step >= steps
            if done:
                break
    issued = np.stack(issued)
    return bool(success), hashlib.sha256(issued.tobytes()).hexdigest(), issued


def goal_coverage(rows, threshold=.5):
    """Mode-spread check: do fixed noise arms reach different goals?

    Per goal, the best arm's success (mean over scenes and repeats), the number of arms
    reaching it at `threshold` and native success. Passing needs every goal reached by
    some arm and arms that prefer more than one goal. With a hidden goal and a
    multimodal motor, native success should be near 1/goals.
    """
    success = defaultdict(list)
    for row in rows:
        success[row["variant"], row["arm"]].append(float(row["success_once"]))
    goals = sorted({goal for goal, _ in success})
    arms = sorted({arm for _, arm in success if arm != NATIVE})
    rate = {key: float(np.mean(values)) for key, values in success.items()}
    per_goal = [dict(goal=g, best_arm_success=max(rate.get((g, a), 0.) for a in arms),
                     arms_reaching=sum(rate.get((g, a), 0.) >= threshold for a in arms),
                     native=rate.get((g, NATIVE)))
                for g in goals]
    preferred = {max(goals, key=lambda g: rate.get((g, a), 0.)) for a in arms
                 if max(rate.get((g, a), 0.) for g in goals) >= threshold}
    return dict(per_goal=per_goal, goals_preferred_by_some_arm=len(preferred), threshold=threshold,
                passed=bool(goals and arms) and all(r["arms_reaching"] > 0 for r in per_goal)
                and len(preferred) > 1)
