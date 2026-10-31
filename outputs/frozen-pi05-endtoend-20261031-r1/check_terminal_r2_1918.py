"""Bounded terminal metadata/sample check; no checkpoint load or journal replay."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import zipfile

root = Path(sys.argv[1])
out = root / "a1-r2/experiment"

def check(path, expected=None):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    return {"path": str(path), "bytes": len(data), "sha256": digest,
            "expected_sha256": expected,
            "matched": None if expected is None else digest == expected}

result_bytes = (out / "result.json").read_bytes()
result = json.loads(result_bytes)
rows = result["records"]
manifest_checks = []
for key, row in rows.items():
    ref = row["episode_manifest_ref"]
    path = Path(ref["root"]) / ref["path"]
    manifest = json.loads(path.read_bytes())
    manifest_checks.append({"episode": key, **check(path, ref["sha256"]),
        "completed_chunks_equal": manifest["completed_chunks"] == row["chunks"],
        "executed_actions_equal": manifest["executed_primitive_steps"] == row["primitive_steps"],
        "pending_decision": manifest["pending_decision"]})

last = rows["confirm-place_phone_stand-03-action-1"]
ref = last["episode_manifest_ref"]
corpus = Path(ref["root"])
manifest_path = corpus / ref["path"]
manifest = json.loads(manifest_path.read_bytes())
chunk_path = manifest_path.parent / manifest["chunks"][-1]["path"]
chunk = json.loads(chunk_path.read_bytes())
samples = []
for ref in [chunk["decision_arrays"], chunk["native_result"]["arrays"],
            chunk["observation"], chunk["next_observation"]]:
    path = corpus / ref["path"]
    row = check(path, ref["sha256"])
    with zipfile.ZipFile(path) as archive:
        row["zip_crc_passed"] = archive.testzip() is None
        row["array_members"] = archive.namelist()
    samples.append(row)

print(json.dumps({"format": "frozen_pi05_a1_terminal_bounded_v1",
    "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "result": check(out / "result.json", hashlib.sha256(result_bytes).hexdigest()),
    "recovery": check(out / "recovery.pt"),
    "episode_manifest_checks": manifest_checks,
    "newest_raw_samples": samples,
    "sampled_motor_rng_present": all(k in chunk["decision_metadata"]["coefficient_steering"]["audit_only"]
                                     for k in ("motor_rng_before", "motor_rng_after")),
    "sampled_optimizer_updates": chunk["decision_metadata"]["optimizer_updates"],
    "independent_full_journal_replay": False,
    "checkpoint_deserialized": False,
    "array_tensors_deserialized": False}))
