"""Check what two open-jaw, free-space X pulses reveal before any fitting."""
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from libero_remote.environment.task_manifest import TaskManifest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero-axis-native-20261119-r1"))
from native import episode

release = Path("/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69")
manifest = TaskManifest.load(str(release / "assets/task_manifest.json"))
rows = []
for pose in range(30, 36):
    starts, observations = [], {}
    for case, wire in product(("BASE", "D17"), (1, -1)):
        spec = dict(suite="libero_goal", task_id=2, variant=case, init_state_id=pose,
                    seed=19193000 + pose)
        path = manifest.resolve_bddl("libero_goal", 2, case)
        horizon = manifest.require("libero_goal", 2, case)["max_steps"]
        with episode(path, spec, horizon) as (env, start):
            starts.append(start["physics"])
            tape = []
            for pulse in (0.45, -0.45):
                for _ in range(5):
                    command = np.zeros(7, dtype=np.float32)
                    command[0], command[-1] = pulse * wire, -1
                    step = env.step(command)
                    assert not step["terminated"] and not step["truncated"]
                    raw = step["observation"]
                    tape.append(np.r_[raw["robot0_eef_pos"], raw["robot0_gripper_qpos"]])
            observations[case, wire] = np.stack(tape)
            dx = observations[case, wire][4, 0] - start["initial_position"][0]
            rows.append(dict(pose=pose, condition=case, wire=wire, dx=float(dx),
                             decoded=int(np.sign(dx)), physics=start["physics"],
                             public_wrist_jaw=observations[case, wire].tolist()))
    assert len(set(starts)) == 1, f"Starting physics differ on pose {pose}"
    for wire in (1, -1):
        gap = np.max(np.abs(observations["BASE", wire] - observations["D17", wire]))
        print(pose, wire, "maximum BASE/D17 wrist-or-jaw difference", gap, flush=True)

with Path(sys.argv[1]).open("x") as stream:
    json.dump(rows, stream, allow_nan=False)
