"""Inspect saved physical mismatch fingerprints on allocated Linux CPU only."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [r for key in sorted(a.keys() | b.keys())
                for r in differences(a.get(key), b.get(key), f"{path}.{key}".strip("."))]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        return [r for i, (x, y) in enumerate(zip(a, b)) for r in differences(x, y, f"{path}[{i}]")]
    if torch.is_tensor(a):
        a = a.numpy()
    if torch.is_tensor(b):
        b = b.numpy()
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return [dict(path=path, shapes=[list(a.shape), list(b.shape)])]
        indices = np.argwhere(a != b)
        if not len(indices):
            return []
        return [dict(path=path, shape=list(a.shape), differing=int(len(indices)),
                     maximum_absolute_delta=float(np.abs(a.astype(float) - b.astype(float)).max()),
                     values=[dict(index=index.tolist(), left=float(a[tuple(index)]), right=float(b[tuple(index)]))
                             for index in indices] if a.size <= 100 else None)]
    return [] if a == b else [dict(path=path, left=a, right=b)]


def main():
    pairs = json.loads(sys.argv[1])
    output = []
    for row in pairs:
        initial = []
        for ref in row["initial_refs"]:
            raw = Path(ref["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
            initial.append(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False))
        a, b = initial
        output.append(dict(case=row["case"], physical=differences(
            a["evidence"]["snapshot"]["physical"], b["evidence"]["snapshot"]["physical"]),
            public_states=differences(a["public"]["states"], b["public"]["states"])))
    print(json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()
