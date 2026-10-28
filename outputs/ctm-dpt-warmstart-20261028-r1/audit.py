"""Audit saved compact results without loading models or running environments."""
import ast
import json
import math
from pathlib import Path
import struct
import zipfile


ROOT = Path(__file__).resolve().parent


def rewards(path, member):
    with zipfile.ZipFile(path) as archive:
        data = archive.read(member + ".npy")
    assert data[:8] == b"\x93NUMPY\x01\x00"
    size = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10:10 + size].decode("latin1"))
    assert len(header["shape"]) == 2
    dtype = {"<f4": "f", "<f8": "d", "<i8": "q", "<i4": "i"}[header["descr"]]
    flat = struct.unpack("<" + dtype * math.prod(header["shape"]), data[10 + size:])
    trials, attempts = header["shape"]
    if header["fortran_order"]:
        curve = [sum(flat[a * trials:(a + 1) * trials]) / trials for a in range(attempts)]
    else:
        curve = [sum(flat[a::attempts]) / trials for a in range(attempts)]
    return sum(flat) / len(flat), curve


index = json.loads((ROOT / "CHECKPOINTS.json").read_text())
hashes = {}
for line in (ROOT / "evidence/checkpoint-sha256.txt").read_text().splitlines():
    digest, filename = line.split()
    hashes[filename] = digest
rows = []
for candidate in index["candidates"]:
    task = candidate["benchmark"]
    directory = ROOT / "evidence/run-r1" / ("compact-" + task)
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "warmstart.json").read_text())
    initial = json.loads((directory / "metrics.jsonl").read_text().splitlines()[0])["evaluation"]
    assert candidate["sha256"] == summary["warmstart_sha256"] == manifest["checkpoint_sha256"]
    assert candidate["sha256"] == hashes[f"compact-{task}/warmstart.pt"]
    assert candidate["resume_sha256"] == summary["resume_checkpoint_sha256"]
    assert candidate["resume_sha256"] == hashes[f"compact-{task}/latest.pt"]
    assert summary["iteration"] == summary["counters"]["optimizer_steps"] == candidate["updates"]
    row = dict(benchmark=task, updates=summary["iteration"],
               completed_target=summary["completed"], elapsed_seconds=summary["elapsed_seconds"],
               initial_validation_nll=initial["validation_nll"],
               final_validation_nll=summary["evaluation"]["validation_nll"], conditions={})
    for condition, metric in summary["evaluation"]["behavior"].items():
        values = []
        for iteration, metrics in ((0, initial["behavior"]),
                                   (summary["iteration"], summary["evaluation"]["behavior"])):
            if task == "cached_rotation":
                path, member = directory / f"evaluation-{iteration}.npz", condition
            else:
                path, member = directory / f"evaluation-{iteration}-{condition}.npz", "rewards"
            mean, curve = rewards(path, member)
            expected = metrics[condition]
            assert abs(mean - expected[candidate["metric"]]) < 1e-6, (task, condition, iteration)
            expected_curve = expected.get("success_by_attempt", expected.get("reward_by_attempt"))
            assert len(curve) == len(expected_curve)
            assert all(abs(a - b) < 1e-6 for a, b in zip(curve, expected_curve))
            values.append(mean)
        row["conditions"][condition] = dict(initial=values[0], final=values[1])
    rows.append(row)
result = dict(status="passed", checks="7 export/resume digest pairs and 84 raw behavioral means/curves",
              interpretation="artifact audit only; no model computation", results=rows)
(ROOT / "evidence/compact-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
