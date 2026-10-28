"""Audit saved JSON/text and Git blobs; no model imports or computation."""

import hashlib
import json
import math
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parent
evidence = root / "evidence"
source = "91c16961a02aa739c6cda6cb3c8627145d51aeef"
summary = json.loads((evidence / "summary.json").read_text())
probe = json.loads((evidence / "gpu-probe.json").read_text())
assert (evidence / "exit-code.txt").read_text().strip() == "0"
assert "142 passed" in (evidence / "qualification.log").read_text()
assert probe["status"] == "passed" and len(probe["devices"]) == 1
assert summary["source_commit"] == source
assert summary["allocation"]["job_id"] == "job-6zp4ujvgjbqw-master-0"
assert summary["core_config"]["ticks"] == 1
assert summary["training_transitions"] == 25000
updates = summary["updates"]
serial = updates["original_serial_full_trial"]
batched = updates["batched_heads_full_trial"]
chunked = updates["cached_chunks_100"]
for name, row in updates.items():
    assert math.isfinite(row["seconds"]) and row["seconds"] > 0
    assert max(row["replay_error"].values()) < 2e-5
    metrics = row["metrics"]
    assert metrics["optimizer_steps"] == 5
    assert metrics["forward_samples"] == metrics["active_action_samples"] == 25000
    assert math.isfinite(metrics["gradient_norm"]) and metrics["gradient_norm"] > 0
assert math.isclose(serial["metrics"]["loss"], batched["metrics"]["loss"], abs_tol=1e-9)
assert abs(serial["metrics"]["gradient_norm"] - batched["metrics"]["gradient_norm"]) < 1e-6
assert serial["gradient_horizon_observations"] == batched["gradient_horizon_observations"] == 2500
assert chunked["gradient_horizon_observations"] == 100

files = [
    "src/fluffy_bassoon/meta_rl/ppo.py",
    "src/fluffy_bassoon/meta_rl/metaworld_train.py",
    "src/fluffy_bassoon/meta_rl/metaworld_policy.py",
    "src/fluffy_bassoon/meta_rl/metaworld_env.py",
    "src/fluffy_bassoon/meta_rl/metaworld_ppo_timing.py",
    "tests/test_meta_rl_ppo.py",
    "tests/test_meta_rl_checkpoint.py",
    "tests/test_meta_rl_replay_windows.py",
    "tests/test_metaworld_pilot.py",
    "tests/test_meta_rl_cuda.py",
    "tests/test_meta_rl_structure.py",
    "scripts/run_metaworld_ppo_qualification.sh",
]
hashes = {}
repo = root.parents[1]
for name in files:
    qualified = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=repo)
    integrated = (repo / name).read_bytes()
    assert qualified == integrated, name
    hashes[name] = hashlib.sha256(integrated).hexdigest()
original_collection = summary["rollouts"]["original_collection"]["seconds"]
cached_collection = summary["rollouts"]["cached_collection"]["seconds"]
original_total = original_collection + serial["seconds"]
batched_total = original_collection + batched["seconds"]
chunked_total = cached_collection + chunked["seconds"]
result = dict(
    status="passed", source_commit=source,
    job_id="job-6zp4ujvgjbqw", model_computation_performed_by_audit=False,
    audit_basis="saved artifacts and identical qualified source blobs",
    active_allocation_seconds=int((evidence / "allocation-active-seconds.txt").read_text()),
    original_collection_and_ppo_seconds=original_total,
    batched_collection_and_ppo_seconds=batched_total,
    cached_collection_and_ppo_seconds=chunked_total,
    batched_end_to_end_speedup=original_total / batched_total,
    cached_end_to_end_speedup=original_total / chunked_total,
    source_file_sha256=hashes,
    evidence_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in evidence.iterdir() if p.name != "audit.json"},
)
(evidence / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if not key.endswith("sha256")}, indent=2))
