"""Collect a neutral frozen pickup and physically test both continuations on its exact fork."""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from layout import FRUIT, reset_scene
from pilot import CAMERAS, STATES, convert_action, digest, inputs, ou, stable_counter_regions

PROMPT = "Put one fruit from the plate into a bowl. Release it and move your hand away."
SIDES = ("left", "right")


def write(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def state(obs):
    return np.concatenate([obs[k] for k in STATES])


def identity(env, obs):
    sim = env.sim
    physical = digest(sim.get_state().flatten(), sim.data.cam_xpos, sim.data.cam_xmat,
                      sim.model.body_pos, sim.model.geom_pos)
    return {"physical": physical, "public": digest(state(obs), *(obs[k] for k in CAMERAS))}


def pooled(obs):
    features = [state(obs)]
    for k in CAMERAS:
        image = np.asarray(obs[k])
        h, w, ch = image.shape
        assert h % 4 == w % 4 == 0 and ch == 3 and image.dtype == np.uint8
        features.append(image.reshape(4, h//4, 4, w//4, 3).mean((1, 3)).ravel() / 255.)
    return np.concatenate(features).astype(np.float32)


def held_fruit(env):
    gripper = env.robots[0].gripper["right"]
    return next((f for f in FRUIT if env._check_grasp(gripper, env.objects[f])), None)


def release(env, bowls):
    return [any(ou.check_obj_in_receptacle(env, f, b) and ou.gripper_obj_far(env, f)
                for f in FRUIT) for b in bowls]


def pickup(motor, out, seed):
    dest = out / str(seed)
    dest.mkdir()
    wrapped, obs, layout = reset_scene(seed)
    try:
        assert all(layout["checks"].values())
        env = wrapped.unwrapped.env
        initial = identity(env, obs)
        noise = np.random.default_rng(seed + 501003)
        queue = deque()
        commands, post, history, positions = [], [], [], []
        fruit = None
        for step in range(500):
            if not queue:
                positions.append(step)
                history.append(pooled(obs))
                pred = motor.infer(inputs(obs, PROMPT), noise=noise.standard_normal((50, 32), dtype=np.float32))
                actions = np.asarray(pred["actions"])
                assert actions.shape == (50, 12) and np.isfinite(actions).all()
                queue.extend(actions[:5])
            action = queue.popleft()
            obs, _, done, truncated, _ = wrapped.step(convert_action(action))
            commands.append(action.copy())
            post.append(state(obs))
            fruit = held_fruit(env)
            if fruit or done or truncated:
                break
        positions.append(len(commands))
        history.append(pooled(obs))
        np.savez_compressed(dest/"pickup.npz", commands=commands, post=post, public=history, offsets=positions)
        data = {"seed":seed, "controls":len(commands), "fruit":fruit, "initial":initial,
                "decision":identity(env, obs), "geometry":layout}
        write(dest/"pickup.json", data)
        print("PICKUP", seed, len(commands), fruit, flush=True)
    finally:
        wrapped.close()


def carry(env, fruit, bowl):
    """The six positions, gains and phase time limits used in the November 9 carry."""
    robot = env.robots[0]
    c = robot.part_controllers["right"]
    assert c.input_ref_frame == "base" and c.input_type == "delta"
    def hand():
        return env.sim.data.site_xpos[robot.eef_site_id["right"]].copy()
    def pos(k):
        return env.sim.data.body_xpos[env.obj_body_id[k]].copy()
    start, target, offset = hand(), pos(bowl), hand()-pos(fruit)
    high = max(start[2]+.16, target[2]+.18+offset[2])
    lift = start.copy()
    lift[2] = high
    over = np.array([*(target[:2]+offset[:2]), high])
    low = over.copy()
    low[2] = target[2]+.075+offset[2]
    away = over.copy()
    away[:2] -= .12*c.origin_ori[:2, 0]
    away[2] += .12
    phases = (("secure",start,1.,8), ("raise",lift,1.,45), ("carry",over,1.,65),
              ("lower",low,1.,40), ("open",low,0.,12), ("retreat",away,0.,65))
    for phase, xyz, grip, limit in phases:
        for _ in range(limit):
            action = np.zeros(12)
            action[:3] = np.clip(c.origin_ori.T @ (xyz-hand())/.05, -.55, .55)
            action[6] = grip
            yield phase, action
            if phase not in ("secure", "open") and np.linalg.norm(xyz-hand()) < .015:
                break


def branch(out, seed, side):
    dest = out / str(seed)
    prior = json.loads((dest/"pickup.json").read_text())
    wrapped, obs, layout = reset_scene(seed)
    try:
        env = wrapped.unwrapped.env
        assert all(layout["checks"].values()) and identity(env, obs) == prior["initial"], (seed, "initial")
        with np.load(dest/"pickup.npz", allow_pickle=False) as prefix:
            for action, after in zip(prefix["commands"], prefix["post"], strict=True):
                obs, _, _, _, _ = wrapped.step(convert_action(action))
                np.testing.assert_array_equal(state(obs), after)
        assert identity(env, obs) == prior["decision"] and held_fruit(env) == prior["fruit"], (seed, "fork")
        result = {"seed":seed, "side":side, "decision":prior["decision"], "prefix":prior["controls"],
                  "placed":[False,False], "native_four_fruit":False, "branch_controls":0, "phases":{}}
        features, frames, states, offsets, commands = [], [], [], [], []
        generator = carry(env, prior["fruit"], layout["bowls"][SIDES.index(side)]) if prior["fruit"] else ()
        for (phase, action), _ in zip(generator, range(500-prior["controls"])):
            if len(commands) % 5 == 0:
                offsets.append(len(commands))
                features.append(pooled(obs))
                frames.append(np.stack([obs[k] for k in CAMERAS]))
                states.append(state(obs))
            obs, _, done, truncated, info = wrapped.step(convert_action(action))
            commands.append(action)
            result["phases"][phase] = result["phases"].get(phase,0)+1
            result["native_four_fruit"] |= bool(info["success"])
            result["placed"] = release(env, layout["bowls"])
            if any(result["placed"]) or done or truncated:
                break
        offsets.append(len(commands))
        features.append(pooled(obs))
        frames.append(np.stack([obs[k] for k in CAMERAS]))
        states.append(state(obs))
        np.savez_compressed(dest/f"{side}.npz", frames=np.stack(frames), states=states, public=features,
                            offsets=offsets, commands=np.asarray(commands,dtype=np.float64).reshape(-1,12))
        result["branch_controls"] = len(commands)
        write(dest/f"{side}.json", result)
        print("BRANCH", seed, side, result["placed"], len(commands), flush=True)
    finally:
        wrapped.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("pickup","left","right"))
    ap.add_argument("out", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("seeds", nargs="+", type=int)
    args = ap.parse_args()
    stable_counter_regions()
    if args.mode == "pickup":
        import jax
        from openpi.policies import policy_config
        from openpi.shared import normalize
        from openpi.training import config
        from pilot import with_checkpoint_stats
        assert len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu"
        stats = normalize.load(args.checkpoint/"assets")
        cfg = with_checkpoint_stats(config.get_config("pi05_pretrain_human300"), stats)
        motor = policy_config.create_trained_policy(cfg, args.checkpoint, norm_stats=stats,
                                                    sample_kwargs={"num_steps":10})
        for seed in args.seeds:
            pickup(motor, args.out, seed)
    else:
        for seed in args.seeds:
            branch(args.out, seed, args.mode)


if __name__ == "__main__":
    main()
