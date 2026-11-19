"""Run original JAX OpenPI in its qualified interpreter."""
from multiprocessing.connection import Connection
from pathlib import Path
import sys

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config

CAMPAIGN = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1"
sys.path.insert(0, str(CAMPAIGN))
from reference import CHECKPOINT, observation


def main(fd):
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise RuntimeError("Require the assigned GPU for the frozen motor")
    policy = policy_config.create_trained_policy(config.get_config("pi05_libero"), str(CHECKPOINT))
    with Connection(fd) as channel:
        while True:
            try:
                raw, noise = channel.recv()
            except EOFError:
                return
            actions = policy.infer(observation(raw, raw["instruction"]), noise=noise)["actions"]
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Invalid output from the frozen motor")
            channel.send(actions)


if __name__ == "__main__":
    main(int(sys.argv[1]))
