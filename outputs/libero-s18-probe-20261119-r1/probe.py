"""Compare paired public wrist positions and motions under released BASE and S18."""
import argparse
from functools import partial
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from libero.libero import envs
from libero.libero.envs.env_wrapper import ControlEnv
from libero_remote.environment.libero import LiberoEnvironmentAdapter
from libero_remote.environment.task_manifest import TaskManifest


def probe(manifest, suite, task, case, seed, signs):
    spec = dict(suite=suite, task_id=task, variant=case, init_state_id=2, seed=seed,
                camera_names=[], camera_height=256, camera_width=256, control_mode="relative",
                max_steps=manifest.require(suite, task, case)["max_steps"],
                _bddl_path=str(manifest.resolve_bddl(suite, task, case)))
    env = LiberoEnvironmentAdapter()
    try:
        reset = env.create(spec)
        identity = reset["environment_reproducibility"]
        positions = [reset["observation"]["robot0_eef_pos"].tolist()]
        rewards, done = [], False
        for axis in (0, 1):
            action = np.zeros(7, dtype=np.float32)
            action[axis] = .45 * signs[axis]
            for _ in range(5):
                result = env.step(action)
                positions.append(result["observation"]["robot0_eef_pos"].tolist())
                rewards.append(float(result["reward"]))
                done = bool(result["terminated"] or result["truncated"])
                if done:
                    break
            if done:
                break
        return dict(positions=positions, rewards=rewards, done=done,
                    initial_physics=identity["initial_state_fingerprint"],
                    plan_fingerprint=identity["perturbation_plan_fingerprint"])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    run = Path(__file__).resolve().parent
    split = json.loads((run.parent / "libero-meta-learning-20261117-r1/SPLITS.json").read_text())
    tasks = [(suite, task) for suite, ids in sorted(split["development"].items()) for task in ids]
    manifest = TaskManifest.load(str(args.release / "assets/task_manifest.json"))
    headless = partial(ControlEnv, use_camera_obs=False, has_offscreen_renderer=False)
    rows = []
    with patch.object(envs, "OffScreenRenderEnv", headless):
        for i, (suite, task) in enumerate(tasks):
            seed = 19117100 + i
            direction = [1 if i % 2 == 0 else -1, 1 if i % 4 < 2 else -1]
            traces = {case: probe(manifest, suite, task, case, seed, direction) for case in ("BASE", "S18")}
            a, b = [np.asarray(traces[case]["positions"], dtype=float) for case in ("BASE", "S18")]
            count = min(len(a), len(b))
            offset = b[:count] - a[:count]
            drift = np.diff(offset, axis=0)
            rows.append(dict(suite=suite, task=task, seed=seed, reset=2, direction=direction,
                             matched_initial_physics=traces["BASE"]["initial_physics"] == traces["S18"]["initial_physics"],
                             controls_compared=count - 1, reset_offset_m=offset[0].tolist(),
                             max_displacement_difference_m=float(np.linalg.norm(drift, axis=1).max()),
                             max_offset_change_m=float(np.linalg.norm(offset - offset[:1], axis=1).max()),
                             traces=traces))
            print(suite, task, rows[-1]["max_displacement_difference_m"], flush=True)
    (args.output / "summary.json").write_text(json.dumps(dict(manifest=manifest.sha256, pairs=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
