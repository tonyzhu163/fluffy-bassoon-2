"""Portable, benchmark-bound supervised initializations (separate from resume)."""

import hashlib
import json
import os
from pathlib import Path
import re

import torch

from fluffy_bassoon.evaluation_stages import sha256, write_json


FORMAT = "ctm_benchmark_warmstart_v1"
COMPACT = "compact_history_v1"
CACHED = "cached_rotation_history_v1"
NATIVE = "native_pi05_history_v1"


def compatibility(family, contract):
    """Deliberately exclude optimizer, batch, seed and source revision of the new phase."""
    if family == COMPACT:
        keys = ("task", "attempts", "horizon", "arms", "observation_encoding",
                "core", "core_config", "dtype")
    elif family == CACHED:
        keys = ("protocol", "cache", "attempts", "observation_encoding", "core", "core_config", "dtype")
    elif family == NATIVE:
        keys = ("task", "objects", "train_theta_degrees", "train_overlays", "attempts", "episode_steps",
                "action_chunk", "denoise_steps", "feedback", "reward", "core", "bridge", "gaussian_policy",
                "motor_sha256", "norm_stats_sha256", "rlinf_sources")
    else:
        raise ValueError("Unknown warm-start checkpoint family")
    # JSON round-trip produces a portable copy and rejects invalid numeric metadata.
    return json.loads(json.dumps({key: contract[key] for key in keys}, allow_nan=False))


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return sha256(path)


def export_warmstart(path, *, family, policy, contract, iteration, counters, data, evaluation):
    """Export weights only; the trainer separately retains a resumable latest.pt."""
    if type(iteration) is not int or iteration < 1:
        raise ValueError("Cannot export an untrained warm start")
    if not re.fullmatch(r"[0-9a-f]{40}", contract["source_commit"]):
        raise ValueError("Require a pinned source commit")
    if not data or not evaluation or not counters or any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError("Require training-data provenance, evaluation and nonnegative cost counters")
    weights = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    if any(not value.isfinite().all() for value in weights.values()):
        raise ValueError("Cannot export non-finite weights")
    manifest = json.loads(json.dumps(dict(family=family, compatibility=compatibility(family, contract),
        source_commit=contract["source_commit"], iteration=iteration, pretraining_counters=counters,
        data=data, evaluation=evaluation, status="candidate; adaptation requires behavioral evaluation"), allow_nan=False))
    digest = atomic_save(path, dict(format=FORMAT, manifest=manifest, policy=weights))
    write_json(Path(path).with_suffix(".json"), dict(**manifest, checkpoint_sha256=digest,
                checkpoint_file=Path(path).name))
    return digest


def checked_load(path, expected_sha256, *, weights_only=True):
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Require the declared checkpoint SHA256")
    if sha256(path) != expected_sha256:
        raise ValueError("Checkpoint checksum differs")
    saved = torch.load(path, map_location="cpu", weights_only=weights_only)
    if sha256(path) != expected_sha256:
        raise ValueError("Checkpoint changed while being read")
    return saved


def validate_weights(policy, actual):
    expected = policy.state_dict()
    if (not isinstance(actual, dict) or actual.keys() != expected.keys()
            or any(not isinstance(actual[key], torch.Tensor) or actual[key].shape != value.shape
                   or actual[key].dtype != value.dtype or not actual[key].isfinite().all()
                   for key, value in expected.items())):
        raise ValueError("Checkpoint tensor schema or finite-value check failed")
    # This buffer is checked again by the policy's load hook; reject before any mutation.
    if "saved_policy_config" in expected and not torch.equal(
            expected["saved_policy_config"].cpu(), actual["saved_policy_config"]):
        raise ValueError("Checkpoint Gaussian policy configuration differs")


@torch.no_grad()
def initialize_from_warmstart(path, expected_sha256, *, family, policy, contract):
    saved = checked_load(path, expected_sha256)
    manifest = saved.get("manifest", {})
    if (saved.get("format") != FORMAT or manifest.get("family") != family
            or manifest.get("compatibility") != compatibility(family, contract)
            or type(manifest.get("iteration")) is not int or manifest["iteration"] < 1
            or not manifest.get("data") or not manifest.get("pretraining_counters")):
        raise ValueError("Warm-start family, benchmark, interface or provenance differs")
    validate_weights(policy, saved.get("policy"))
    policy.load_state_dict(saved["policy"], strict=True)
    policy.critic.weight.zero_()
    policy.critic.bias.zero_()
    return dict(kind=FORMAT, checkpoint_sha256=expected_sha256, parent=manifest,
                transferred="recurrent core and actor; no trial memory", critic="zero",
                optimizer="fresh", rng="new phase seed; parent RNG not restored")


def add_warmstart_arguments(parser):
    parser.add_argument("--warmstart", type=Path, help="Benchmark-specific supervised weights, not a resume")
    parser.add_argument("--warmstart-sha256")


def validate_warmstart_arguments(args):
    if (args.warmstart is None) != (args.warmstart_sha256 is None):
        raise ValueError("Warmstart requires both a path and SHA256")
    if args.warmstart and getattr(args, "motor_warmstart", None):
        raise ValueError("Choose one initialization family")


def manifest_hash(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
