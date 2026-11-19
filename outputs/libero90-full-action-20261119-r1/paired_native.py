"""Verify that sampling and discarding another continuous chunk preserves the native retry."""
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
SISTER = HERE.parent / "libero90-meta-retry-20261119-r1"
sys.path.insert(0, str(SISTER))
from collect import FrozenMotor, policy_input, runtime_identity
from standard_sim import Simulator, fingerprint


def episode(sim, motor, case, seed, extra):
    raw, identity = sim.call("reset", case)
    rng = np.random.default_rng(seed)
    other = np.random.default_rng(seed + 5419)
    commands, motion, reward, diversity = [], [], [], []
    started = time.monotonic()
    while True:
        item = policy_input(raw, identity["prompt"])
        chunk = motor.propose(item, rng.standard_normal((10, 32), dtype=np.float32))
        if extra:
            motor.features(item)
            second = motor.propose(item, other.standard_normal((10, 32), dtype=np.float32))
            diversity.append(np.abs(chunk[:5] - second[:5]).mean(axis=0).tolist())
        step = sim.call("step", chunk[:5])
        count = len(step["rewards"])
        commands.extend(chunk[:count])
        motion.extend(step["positions"])
        reward.extend(step["rewards"])
        raw = step["observation"]
        if step["done"]:
            return dict(reset=identity, controls=len(commands), success=bool(step["success"]),
                        action_sha=fingerprint(np.array(commands)), wrist_sha=fingerprint(np.array(motion)),
                        reward_sha=fingerprint(np.array(reward)), seconds=time.monotonic()-started,
                        mean_candidate_abs_difference=np.mean(diversity, axis=0).tolist() if extra else None)


def main(destination):
    receipt = dict(runtime=runtime_identity())
    row = json.loads((SISTER / "cases.json").read_text())[0]
    receipt["case"] = row
    motor = FrozenMotor()
    case = dict(task_id=row["task_id"], init_state_id=row["query_id"], seed=row["seed"])
    with Simulator() as sim:
        for name in ("warmup", "native", "discard_second"):
            receipt[name] = episode(sim, motor, case, row["seed"]+3, name=="discard_second")
            destination.write_text(json.dumps(receipt, indent=2)+"\n")
            print(name, receipt[name]["controls"], receipt[name]["success"], flush=True)
    for field in ("reset", "controls", "success", "action_sha", "wrist_sha", "reward_sha"):
        if receipt["native"][field] != receipt["discard_second"][field]:
            raise ValueError(f"Extra candidate changed base trace: {field}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
