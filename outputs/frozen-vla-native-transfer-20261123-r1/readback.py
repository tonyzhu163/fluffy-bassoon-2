"""Re-read only native zero-transform cells from the two independently hash-bound panels."""
import csv
import hashlib
import json
import sys
from pathlib import Path


def read(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(path)
    d = json.loads(raw)
    (r,) = [row for row in d["rows"] if row["embodiment"] == "fam_nom"]
    if (r["true_yaw"], r["true_scale"], r["true_grip"]) != (0, 1, 1):
        raise ValueError(r)
    return d, r


def outcome(cell):
    if type(cell["success"]) is not bool or cell["controls"] < 1:
        raise ValueError(cell)
    return dict(success=cell["success"], controls=cell["controls"])


rows = []
for entry in csv.DictReader(sys.stdin):
    b, later = read(entry["seeds42_43"], entry["sha256_42_43"])
    _, earlier = read(entry["seed41_and_hs"], entry["sha256_41_hs"])
    u = b["unshifted"]
    if u["init_state_id"] != 13 or u["suite"] != later["suite"] or u["task_id"] != earlier["task_id"]:
        raise ValueError(u)
    comparisons = {"raw": outcome(u), "raw_repeat": outcome(later["arms"]["zero_shot_shifted"])}
    for name, prefix in [("symmeta", "ours_equiv_v2_passive"), ("perceiver", "perceiver_icba_passive")]:
        comparisons[f"{name}_s41"] = outcome(earlier["arms"][prefix])
        for seed in (42, 43):
            comparisons[f"{name}_s{seed}"] = outcome(later["arms"][f"{prefix}_s{seed}"])
    comparisons["gold_wire_identity_s41"] = outcome(earlier["arms"]["privileged_oracle"])
    rows.append(dict(panel=entry["panel"], suite=u["suite"], task=u["task_id"], layout=u["init_state_id"],
                     attempts=comparisons, source=entry))
if len(rows) != 32:
    raise ValueError(f"Unexpected row count {len(rows)}")
summary = {}
for panel in sorted({r["panel"] for r in rows}):
    selected = [r for r in rows if r["panel"] == panel]
    summary[panel] = {name: sum(r["attempts"][name]["success"] for r in selected)
                      for name in selected[0]["attempts"]}
print(json.dumps(dict(summary=summary, rows=rows), indent=2))
