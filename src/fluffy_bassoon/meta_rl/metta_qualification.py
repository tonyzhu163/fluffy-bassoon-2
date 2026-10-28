"""Allocated simulator headroom controls; the oracle has explicit private access."""

import argparse
from collections import deque
import json
from pathlib import Path
import time

import numpy as np

from fluffy_bassoon.evaluation_stages import write_json
from .metta_env import FREE_CELLS, MemoryTrial, memory_pools
from .screen import allocation, clean_source


def cell_path(start, goal):
    queue, parents = deque([start]), {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            result = []
            while cell != start:
                result.append(cell)
                cell = parents[cell]
            return result[::-1]
        r, c = cell
        for next_cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if next_cell in FREE_CELLS and next_cell not in parents:
                parents[next_cell] = cell
                queue.append(next_cell)
    raise ValueError("Goal is unreachable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence, source = allocation(), clean_source(args.source_commit)
    pools = memory_pools("pointmaze", 0)
    result = dict(source_commit=source, allocation=evidence,
                  oracle_access="true goal, maze layout and physical state; headroom only, never a policy input",
                  records={})
    started = time.monotonic()
    for split in ("train_eval", "test"):
        classes, tasks = pools[split]
        result["records"][split] = {}
        for condition in ("oracle", "white_noise", "persistent_noise"):
            records = []
            for index, task in enumerate(tasks):
                env = MemoryTrial(classes[task.env_name], [task], benchmark="pointmaze", attempts=2, horizon=500)
                rng = np.random.default_rng(1_000_000_000 + index)
                steps, episode = 0, -1
                try:
                    obs, _ = env.reset(seed=1_000_000_000 + index)
                    while not env.finished:
                        if condition == "oracle":
                            if env.episode != episode:
                                path = cell_path(tuple(env.setting["start"]), tuple(env.setting["goal"]))
                                waypoints = [np.array([c - 3., 3. - r]) for r, c in path]
                                waypoints[-1] = env.env.unwrapped.goal.copy()
                                waypoint, episode = 0, env.episode
                            state = obs["state"]
                            if waypoint < len(waypoints) - 1 and np.linalg.norm(state[:2] - waypoints[waypoint]) < .15:
                                waypoint += 1
                            # The XML actuator gear is 100: use bounded PD force,
                            # rather than saturating every step and overshooting turns.
                            action = np.clip(.5 * (waypoints[waypoint] - state[:2]) - .2 * state[2:], -1., 1.)
                        elif condition == "white_noise":
                            action = np.tanh(rng.normal(size=2))
                        else:
                            if steps % 10 == 0:
                                persistent = np.tanh(rng.normal(size=2))
                            action = persistent
                        obs, _, _, _, _ = env.step(np.asarray(action, np.float32))
                        steps += 1
                    records.append(dict(task=json.loads(task.data), outcomes=env.outcomes,
                        returns=env.native_returns, episode_metrics=env.episode_metrics,
                        environment_steps=steps))
                finally:
                    env.close()
            result["records"][split][condition] = dict(
                success=float(np.mean([row["outcomes"] for row in records])), trials=records,
                environment_steps=sum(row["environment_steps"] for row in records))
    result["elapsed_seconds"] = time.monotonic() - started
    result["oracle_pass"] = all(result["records"][split]["oracle"]["success"] >= .95
                                 for split in result["records"])
    write_json(args.output, result)
    print(json.dumps({split: {name: value["success"] for name, value in conditions.items()}
                      for split, conditions in result["records"].items()}))
    if not result["oracle_pass"]:
        raise RuntimeError("PointMaze navigation headroom is not qualified")


if __name__ == "__main__":
    main()
