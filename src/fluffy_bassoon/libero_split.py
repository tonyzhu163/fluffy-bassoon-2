"""Frozen LIBERO train/validation/test split (outputs/libero-pi05-corpus-20261029-r1).

Loads split.json, checks its content hash and exposes the tasks each stage may use.
The motor SFT of outputs/libero-noise-icrl-20261029-r1/PLAN.md (Stage M0) trains on
the demonstrations of the training tasks only: the split forbids gradients from
validation tasks (`supervised_validation_gradients`) and every use of test tasks.
"""
import hashlib
import json
from pathlib import Path


DEFAULT = Path(__file__).resolve().parents[2]/"outputs/libero-pi05-corpus-20261029-r1/split.json"
SPLITS = ("train", "validation", "test", "test_auxiliary_excluded_90")
COUNTS = dict(train=78, validation=10, test=10, test_auxiliary_excluded_90=2)


def canonical_hash(value):
    """Same rule as build_split.py: sorted keys, compact separators."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path=DEFAULT):
    split = json.loads(Path(path).read_text())
    claimed = split.get("metadata_sha256")
    if claimed != canonical_hash({k: v for k, v in split.items() if k != "metadata_sha256"}):
        raise ValueError("Split content does not match its metadata_sha256")
    counts = {name: sum(task["split"] == name for task in split["tasks"]) for name in SPLITS}
    if counts != COUNTS or len(split["tasks"]) != sum(COUNTS.values()):
        raise ValueError(f"Unexpected split counts {counts}")
    return split


def tasks(split, name):
    if name not in SPLITS:
        raise ValueError(f"Split is one of {SPLITS}")
    return [task for task in split["tasks"] if task["split"] == name]


def motor_training_tasks(split):
    """LIBERO-90 training tasks whose demonstrations the motor may fit."""
    chosen = tasks(split, "train")
    if any(task["suite"] != "libero_90" or not task["meta_gradient_allowed"] for task in chosen):
        raise ValueError("Motor training tasks must be LIBERO-90 tasks with gradients allowed")
    return chosen
