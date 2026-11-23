"""Compare both completed matched CPU qualifications, including saved NPY bytes."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_case(path):
    (trace,) = (path / "matched/traces").glob("*.npz")
    with ZipFile(trace) as z:
        traces = {key: digest(z.read(key)) for key in z.namelist()}
    expected = {f"spatial_0__15__0__{field}.npy" for field in ("X", "A", "A_vla", "ref", "speed")}
    if traces.keys() != expected:
        raise ValueError("Incomplete native traces")
    with (path / "matched/results.csv").open() as f:
        row, = csv.DictReader(f)
    for name in ("wall_s", "ms_vla", "ms_per_step"):
        del row[name]
    ledger, = json.loads((path / "matched/calibration.json").read_text())
    for update in ledger["updates"]:
        del update["compute_s"], update["ipc_and_compute_s"]
    report = json.loads((path / "qualified.json").read_text())
    return dict(traces_npy_sha256=traces, row=row, controls=ledger["scored_controls"],
                queries=len(ledger["queries"]), updates=len(ledger["updates"]), reader=report["reader"],
                ledger_sha256=digest(json.dumps(ledger, sort_keys=True, allow_nan=False).encode()),
                native_replay=report["native_replay"], scene_seed=report["scene_seed"],
                native_revision=report["revision"], launcher_revision=report["code_revision"])


if __name__ == "__main__":
    shared, scratch = (read_case(Path(p)) for p in sys.argv[1:])
    if shared != scratch:
        sys.exit(f"Different fields: {[key for key in shared if shared[key] != scratch[key]]}")
    print(json.dumps(shared, indent=2, sort_keys=True))
