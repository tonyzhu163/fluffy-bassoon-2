"""Reproduce one scored motor episode while recording its public views and physical path."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import eval_bowl
from collect import CAMERAS, SIDES, held_fruit, release, stable_counter_regions, state
from readout import read


def trace(args):
    torch.set_num_threads(3)
    torch.manual_seed(991)
    stable_counter_regions()
    original = read(args.record)
    plan = {k: original[k] for k in ("arm", "goal", "query", "support", "support_goal", "probe")}
    assert plan["arm"] == "oracle"
    prefix = read(args.source / str(plan["query"]) / "pickup.json")
    assert prefix["fruit"] is not None
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    native, student = eval_bowl.actor(args.code, weights)
    args.out.mkdir()
    path, images, image_steps = [], [], []
    reset = eval_bowl.reset_scene

    def observed_reset(seed):
        wrapped, obs, layout = reset(seed)
        env = wrapped.unwrapped.env
        step = wrapped.step
        count = 0
        fruit = prefix["fruit"]
        bowl = layout["bowls"][plan["goal"]]

        def observe(current):
            fruit_pos = env.sim.data.body_xpos[env.obj_body_id[fruit]].copy()
            bowl_pos = env.sim.data.body_xpos[env.obj_body_id[bowl]].copy()
            placed = release(env, layout["bowls"])
            path.append(dict(control=count, state=state(current).tolist(),
                             held=held_fruit(env) == fruit, placed=placed,
                             fruit=fruit_pos.tolist(), bowl=bowl_pos.tolist()))
            if count >= prefix["controls"] and (count == prefix["controls"] or count % 10 == 0 or any(placed) or count == 500):
                images.append(np.stack([current[k].copy() for k in CAMERAS]))
                image_steps.append(count)

        def observed_step(action):
            nonlocal count
            result = step(action)
            count += 1
            observe(result[0])
            return result

        observe(obs)
        wrapped.step = observed_step
        return wrapped, obs, layout

    eval_bowl.reset_scene = observed_reset
    actual = eval_bowl.episode(native, student, weights, args.source, plan)
    assert actual == original
    np.savez_compressed(args.out / "views.npz", frames=images, controls=image_steps)
    for index in np.unique(np.linspace(0, len(images) - 1, 5, dtype=int)):
        for view in (0, 1):
            Image.fromarray(images[index][view]).save(args.out / f"control-{image_steps[index]:03d}-view-{view}.png")
    teacher = args.source / str(plan["query"]) / f"{SIDES[plan['goal']]}.npz"
    with np.load(teacher) as tape:
        frames = tape["frames"]
        for index in np.unique(np.linspace(0, len(frames) - 1, 5, dtype=int)):
            Image.fromarray(frames[index][0]).save(args.out / f"teacher-replan-{index:02d}.png")
    report = dict(record=str(args.record), checkpoint=str(args.checkpoint),
                  reproduced_original_outcome=True, outcome=actual, physical_path=path)
    (args.out / "trace.json").write_text(json.dumps(report, indent=2))
    after = [r for r in path if r["control"] >= prefix["controls"]]
    closest = min(np.linalg.norm(np.array(r["fruit"])[:2] - np.array(r["bowl"])[:2]) for r in after)
    print(json.dumps(dict(query=plan["query"], goal=plan["goal"], placed=actual["placed"],
                          ending_held=after[-1]["held"], closest_horizontal_target_distance=closest)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("code", "checkpoint", "source", "record", "out"):
        parser.add_argument(name, type=Path)
    trace(parser.parse_args())
