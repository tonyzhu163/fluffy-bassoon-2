"""Predeclared native transfer panel for the frozen pi05_libero policy."""
import json
from pathlib import Path

TASKS = (("libero_goal", 3), ("libero_10", 8), ("libero_goal", 6))
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
LAYOUTS = (14, 15)
ARMS = ("Panda", "Kinova3")


def cases(arm):
    for suite, task_id in TASKS:
        for layout in LAYOUTS:
            seed = 41800000 + 10000 * SUITES.index(suite) + 100 * task_id + layout
            yield dict(arm=arm, suite=suite, task_id=task_id, init_state_id=layout, variant="BASE", seed=seed)


def key(case):
    return f"{case['suite']}-{case['task_id']}-{case['init_state_id']}"


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main(arm):
    from ur5e_motor import public_digest
    from ur5e_sim_worker import UR5eSimulator
    import numpy as np
    out = []
    with UR5eSimulator(allow_mesa=True) as sim:
        for case in cases(arm):
            spec, first = sim.call("create", dict(case, embodiment="nom"))
            model = first["model_identity"]
            if model["arm"] != arm or first["worker_step_index"] != 0 or spec["embodiment"] != "nom":
                raise ValueError(spec)
            step = sim.call("chunk", np.zeros((1, 7), np.float32))
            if step["steps"] != 1 or step["terminated"] or step["truncated"] or not np.isfinite(step["rewards"]).all():
                raise ValueError(f"Invalid native qualification step on {case}")
            out.append(dict(case=spec, instruction=first["language_instruction"], model=model,
                            renderer=first["renderer"], schema=first["action_schema"],
                            public=public_digest(first["observation"]), reset=first["environment_reproducibility"],
                            observed_native_reward=step["rewards"], first_control_success=bool(step["success"])))
            save(f"cpu-{arm}.json", out)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
