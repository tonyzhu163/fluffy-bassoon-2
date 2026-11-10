"""Export only uniquely successful executed actions in the native π0.5 training coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, native_flow_targets
from readout import label_chunks, read


def native_targets(data, stats):
    chunks, delivered = label_chunks(data)
    # The gym wrapper maps released gripper/mode scores to signed physical controls.
    physical = chunks.copy()
    physical[..., [6,11]] = np.where(physical[..., [6,11]] >= .5, 1., -1.)
    codec = DirectActionCodec.from_stats(stats)
    normalized = native_flow_targets(codec.encode(physical), codec, stats["actions"])
    assert normalized.shape == (len(chunks),50,32)
    assert np.isfinite(normalized).all() and delivered.any(1).all()
    return normalized, delivered


def common_boundary(left, right):
    """At the last common 5-control replan, the two labels must still share the actual view."""
    n = min(len(left["commands"]),len(right["commands"]))
    diverged = np.flatnonzero((left["commands"][:n] != right["commands"][:n]).any(axis=1))
    if not len(diverged):
        return None
    cursor = int(diverged[0])//5
    assert left["offsets"][cursor] == right["offsets"][cursor] == 5*cursor
    for name in ("states","frames"):
        np.testing.assert_array_equal(left[name][cursor],right[name][cursor])
    return cursor


def export(source, seeds, norm, out):
    out.mkdir()
    stats = read(norm)
    stats = stats.get("norm_stats",stats)
    summary = []
    for seed in seeds:
        scene = source/str(seed)
        pick = read(scene/"pickup.json")
        paths = [scene/f"{side}.npz" for side in ("left","right")]
        receipts = [read(path.with_suffix(".json")) for path in paths]
        assert all(r["decision"] == pick["decision"] and r["seed"] == seed for r in receipts)
        with np.load(paths[0]) as left, np.load(paths[1]) as right:
            decision = common_boundary(left,right)
            for goal in (0,1):
                hits = [a for a,r in enumerate(receipts) if r["placed"][goal]]
                if len(hits) != 1 or hits[0] != goal:
                    summary.append({"seed":seed,"goal":goal,"label":None,"successful_candidates":hits})
                    continue
                arm = hits[0]
                data = (left,right)[arm]
                targets, mask = native_targets(data,stats)
                file = f"{seed}-{goal}.npz"
                np.savez_compressed(out/file, target=targets, delivered=mask, offsets=data["offsets"][:-1])
                summary.append({"seed":seed,"goal":goal,"label":file,"successful_candidates":hits,
                                "query_archive":str(paths[arm]),"matched_decision_index":decision,
                                "valid_rows":len(targets),"covered_controls":int(len(data["commands"]))})
    manifest = {"normalizer_sha256":hashlib.sha256(norm.read_bytes()).hexdigest(),"labels":summary}
    (out/"index.json").write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for key in ("source","admission","norm","out"):
        ap.add_argument(key,type=Path)
    args = ap.parse_args()
    rows=export(args.source,read(args.admission)["admitted"],args.norm,args.out)["labels"]
    print("Unique goal labels",sum(bool(r["label"]) for r in rows),"/",len(rows))
