"""Compare the two frozen source readers on the previous pilot's fixed, labelled arm probe."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import tarfile

import numpy as np
import torch

from adapt_eval import K, ReaderAdapter, digest_file, write_json


@torch.inference_mode()
def score(model, tapes, refs, queries):
    def embed(path):
        a, m = tapes[path]
        history = torch.as_tensor(np.column_stack((a, m)), dtype=torch.float32).unsqueeze(0)
        return model.encode(history).numpy()[0]

    means = {body: np.stack([embed(path) for path in paths]).mean(0) for body, paths in refs.items()}
    rows = []
    for path, true in queries:
        z = embed(path)
        distances = {body: float(np.linalg.norm(z - mean)) for body, mean in means.items()}
        rows.append(dict(body=true, tape=path, nearest=min(distances, key=distances.get), distances=distances))
    tally = {b: dict(correct=sum(r["nearest"] == b and r["body"] == b for r in rows),
                     total=sum(r["body"] == b for r in rows)) for b in means}
    return dict(by_arm=tally, correct=sum(r["body"] == r["nearest"] for r in rows), total=len(rows), queries=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for f in ("archive", "first", "joint", "out"):
        p.add_argument("--" + f, type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "gpu01/pilot/"
    with tarfile.open(args.archive) as bundle:
        names = bundle.getnames()
        refs = {b: sorted(n for n in names if n.startswith(prefix + "calibration_check/")
                          and n.endswith(f"/{b}/prescribed/tape.npz")) for b in ("Panda", "Sawyer", "UR5e")}
        if any(len(paths) != 2 for paths in refs.values()):
            raise ValueError("Expected two calibration-check references per arm")
        pairs = json.load(bundle.extractfile(prefix + "pairs.json"))
        queries = [(prefix + row[k]["tape"], row[k]["body"]) for row in pairs for k in ("support", "donor")]
        if len(set(q for q, _ in queries)) != 8:
            raise ValueError("Expected eight distinct prescribed query histories")
        tapes = {}
        for path in [p for ps in refs.values() for p in ps] + [p for p, _ in queries]:
            with np.load(BytesIO(bundle.extractfile(path).read()), allow_pickle=False) as t:
                a, m = t["commands"], t["motion"]
                if a.shape != (K, 7) or m.shape != a.shape:
                    raise ValueError(f"Missing 16-transition paired tape: {path}")
                tapes[path] = a, m
        for i in range(0, len(queries), 2):
            if not np.array_equal(tapes[queries[i][0]][0], tapes[queries[i+1][0]][0]):
                raise ValueError("Support and donor commands differ")
    record = dict(archive_sha256=digest_file(args.archive), references=refs,
                  method="Euclidean nearest mean, two labelled reference probes per arm, no tuning")
    for stage, file in (("pretrained", args.first), ("joint", args.joint)):
        saved = torch.load(file, map_location="cpu", weights_only=True)
        net = ReaderAdapter().eval().requires_grad_(False)
        net.load_state_dict(saved["state_dict"])
        record[stage] = dict(checkpoint_sha256=digest_file(file), **score(net, tapes, refs, queries))
    write_json(args.out / "report.json", record)


if __name__ == "__main__":
    main()
