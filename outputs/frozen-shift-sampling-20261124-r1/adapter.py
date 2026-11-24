"""Sampling-lane adapters: independent noise streams, and K=4 draws chosen by a source-fitted critic."""
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "frozen-vla-proposal-shift-20261124-r1"):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

from harness import Adapter
import critic

K = 4
STREAMS = 8


class Draws(Adapter):
    """Every variant consumes the start's candidate stream identically: K arrays and one integer per replan."""

    keep = K

    def reset(self, support, first):
        self.origin = np.array(first["observation"]["robot0_eef_pos"], dtype=np.float64)
        self.ticket = 0

    def noises(self, base, rng):
        drawn = [rng.standard_normal(base.shape).astype(np.float32) for _ in range(K)]
        self.ticket = int(rng.integers(K))
        return drawn[:self.keep]


class Single(Draws):
    name = "single-k1"
    keep = 1


class Random(Draws):
    name = "random-k4"

    def choose(self, raw, proposals):
        return self.ticket


class Critic(Draws):
    name = "critic-k4"

    def __init__(self, artifact=None):
        self.artifact = artifact or json.loads(critic.ARTIFACT.read_text())

    def reset(self, support, first):
        super().reset(support, first)
        self.gain = critic.response_gain(support["inputs"], support["response"], self.artifact["source_gain"])
        self.scores = []

    def choose(self, raw, proposals):
        displacement = np.asarray(raw["robot0_eef_pos"], dtype=np.float64) - self.origin
        scores = [critic.score(self.artifact["weights"], displacement, p, self.gain) for p in proposals]
        self.scores.append(scores)
        return int(np.argmax(scores))


class Stream(Adapter):
    """Untreated loop with policy noise from independent stream `index`, fixed per start."""

    def __init__(self, index):
        self.index, self.name = index, f"noise-{index}"

    def reset(self, support, first):
        self.rng = None

    def noises(self, base, rng):
        if self.rng is None:
            # The first word of the candidate stream is the same for every branch of a start.
            self.rng = np.random.default_rng([int(rng.integers(2 ** 62)), self.index])
        return [self.rng.standard_normal(base.shape).astype(np.float32)]


ADAPTERS = [Critic, Random, Single, *(Stream(index) for index in range(STREAMS))]
