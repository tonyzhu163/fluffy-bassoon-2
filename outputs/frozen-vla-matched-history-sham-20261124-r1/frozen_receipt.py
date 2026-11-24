"""Independently read the exact fit, normalized stock checkpoint and source fitting identities on disk."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def receipt(parent):
    fit = parent/"model/gpu02"
    source = parent/"data/dataset-v1/manifest.json"
    stock = Path("/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1")
    fit_sha = sha(fit/"fit.json")
    if fit_sha != "4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc":
        raise ValueError("Sealed fit mismatch")
    report = json.loads((fit/"fit.json").read_bytes())
    data = json.loads(source.read_bytes())
    inputs = json.loads((fit/"train-inputs.json").read_bytes())
    windows = [[row["id"], t] for row in data["splits"]["train"]["trajectories"] for t in row["t"]]
    if (data["robot"] != "Panda" or not inputs["native_train"] or windows != inputs["identities"]
            or {i for i, _, _ in inputs["draws"]} != set(range(len(windows))) or sha(source) != report["data_manifest"]):
        raise ValueError("Frozen source fit provenance")
    heads = sha(fit/"heads.npz")
    ready = sha(stock/"READY.json")
    norm = sha(stock/"assets/physical-intelligence/libero/norm_stats.json")
    if (report["heads_sha256"], report["base"]["checkpoint_manifest"], report["base"]["normalization"]) != (heads, ready, norm):
        raise ValueError("Head archive or stock identity")
    factors = {}
    with zipfile.ZipFile(fit/"heads.npz") as z:
        for k in report["parameters"]["joint"]["fitted"]:
            with z.open(f"joint/{k}.npy") as f:
                arr = np.load(f, allow_pickle=False)
            h = hashlib.sha256(str((str(arr.dtype), arr.shape)).encode() + b"\0")
            h.update(arr.tobytes(order="C"))
            factors[k] = h.hexdigest()
    if factors != report["parameters"]["joint"]["fitted"]:
        raise ValueError("Joint factors differ")
    return dict(fit=fit_sha, heads_file=heads, joint_factors=factors,
                published_kernel=report["parameters"]["joint"]["action_out_proj/kernel"],
                checkpoint_manifest=ready, normalization=norm, source_manifest=sha(source),
                train_inputs=sha(fit/"train-inputs.json"), native_train=inputs["native_train"],
                source_robot=data["robot"], trajectories=len(data["splits"]["train"]["trajectories"]),
                distinct_windows=len(windows), draws=len(inputs["draws"]), code=report["code_revision"])


if __name__ == "__main__":
    print(json.dumps(receipt(Path(sys.argv[1])), indent=2))
