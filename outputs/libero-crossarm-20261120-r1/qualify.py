"""Measure the three physical reset models and one real control on the allocated CPU."""

import json
from pathlib import Path
import sys

import numpy as np

from arm_env import ArmEnv


def main(path):
    path.mkdir(exist_ok=False)
    cases = []
    for arm, joints in (("Panda", 7), ("Sawyer", 7), ("UR5e", 6)):
        with ArmEnv(arm=arm, suite="libero_goal", task_id=4) as env:
            start = env.reset(init_state_id=0, seed=20261520)
            raw = start["observation"]
            identity = start["model_identity"]
            assert identity["arm"] == arm and len(raw["robot0_joint_pos"]) == joints
            assert raw["robot0_eye_in_hand_image"].shape == raw["agentview_image"].shape == (256, 256, 3)
            seen = []
            for cmd in (0.4, 0.4, 0.0):
                action = np.array([cmd, 0, 0, 0, 0, 0, -0.4], dtype=np.float32)
                step = env.step(action)
                seen.append(dict(motion=step["sensor_motion"].tolist(), reward=float(step["reward"])))
            case = dict(model=identity, physics=start["environment_reproducibility"],
                        prompt=start["language_instruction"], controls=seen)
            cases.append(case)
            (path / (arm + ".json")).write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
            print(arm, joints, raw["robot0_eef_pos"].tolist(), seen[0]["motion"], flush=True)
    assert len({case["physics"]["scene_fingerprint"] for case in cases}) == 1
    print("Verified distinct models, common scene, both cameras, and three measured controls per arm.", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
