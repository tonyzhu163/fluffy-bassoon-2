"""Harness adapters that sample the frozen policy with a fitted input prefix.

The adapter tags the prompt. A wrapper on the policy class's `infer` removes the tag and,
for that call only, samples through the same weights with the prefix injected after
`embed_prefix`. Untagged calls reach the stock method unchanged. The tag is what the
harness records per replan and what keeps a prefixed query apart from an untreated one.
"""
import contextlib
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
for folder in (HERE, HERE.parent / "frozen-vla-proposal-shift-20261124-r1"):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

from harness import Adapter
import prefix as px

FIT = "FROZEN_SHIFT_PREFIX_FIT"  # directory holding prefix-shift.npz and prefix-tokens.npz
TAG = re.compile(r" \[\[embodiment-prefix:([a-z-]+)\]\]$")
REGISTRY = {}   # adapter name -> prefix arrays
SAMPLERS = {}   # (policy id, adapter name) -> prefixed sampler
BUILD = None    # (policy, prefix) -> replacement for policy._sample_actions; OpenPI by default


def tag(name):
    return f" [[embodiment-prefix:{name}]]"


def build(policy, prefix):
    if BUILD is not None:
        return BUILD(policy, prefix)
    import openpi_engine
    return openpi_engine.sampler(policy, prefix)


def install(cls=None):
    """Wrap `cls.infer` once. Returns the stock function."""
    if cls is None:
        from openpi.policies.policy import Policy as cls
    current = cls.infer
    if hasattr(current, "stock"):
        return current.stock

    def infer(self, obs, *args, **kwargs):
        found = TAG.search(obs["prompt"]) if isinstance(obs.get("prompt"), str) else None
        if found is None:
            return current(self, obs, *args, **kwargs)
        name = found.group(1)
        key = (id(self), name)
        if key not in SAMPLERS:
            SAMPLERS[key] = build(self, REGISTRY[name])
        stock = self._sample_actions
        self._sample_actions = SAMPLERS[key]
        try:
            return current(self, dict(obs, prompt=obs["prompt"][:found.start()]), *args, **kwargs)
        finally:
            self._sample_actions = stock

    infer.stock = current
    cls.infer = infer
    return current


def uninstall(cls):
    if hasattr(cls.infer, "stock"):
        cls.infer = cls.infer.stock
    SAMPLERS.clear()


class Prefixed(Adapter):
    arm = None
    policy_class = None  # tests set a stand-in class; None is openpi.policies.policy.Policy

    def __init__(self):
        folder = os.environ.get(FIT)
        if not folder:
            raise RuntimeError(f"Set {FIT} to a completed fit directory")
        self.prefix, self.meta = px.load(Path(folder) / f"prefix-{self.arm}.npz")
        self.prefix = self.transform(self.prefix)

    def transform(self, prefix):
        return prefix

    def reset(self, support, first):
        held = REGISTRY.setdefault(self.name, self.prefix)
        if {k: px.digest(v) for k, v in held.items()} != {k: px.digest(v) for k, v in self.prefix.items()}:
            raise RuntimeError(f"{self.name} changed within one process")
        install(self.policy_class)

    def prompt(self, raw, original):
        return original + tag(self.name)


class Shift(Prefixed):
    name, arm = "prefix-shift", "shift"


class Tokens(Prefixed):
    name, arm = "prefix-tokens", "tokens"


class Random(Prefixed):
    """Seeded random direction with the fitted shift's norm."""
    name, arm = "prefix-random", "shift"

    def transform(self, prefix):
        return px.random_like(prefix, self.meta["seed"])


ADAPTERS = [Shift, Tokens, Random]
