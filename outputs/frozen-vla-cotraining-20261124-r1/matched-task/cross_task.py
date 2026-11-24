"""Check all original demonstration resets underlying every source task against the four locked scene rows."""
import argparse
import math
from pathlib import Path

import h5py
from register import dump, original_xy, read, sha

HERE = Path(__file__).resolve().parent


def audit(inputs, manifest_path, output):
    index = read(HERE / "original-files.json")
    registered = read(HERE / "contract.json")
    manifest = read(manifest_path)
    bank = read(inputs / "release.json")
    if sha(manifest_path) != registered["data_manifest"]:
        raise ValueError("Staged manifest changed")
    staged = [r for s in manifest["splits"].values() for r in s["trajectories"]]
    if {r["prompt"] for r in staged} != set(index):
        raise ValueError("Original archives do not cover the staged split")
    comparisons = []
    for task, spec in index.items():
        path = inputs / Path(spec["path"]).name
        if sha(path) != spec["sha256"]:
            raise ValueError(f"Original archive bytes differ for {task}")
        with h5py.File(path) as f:
            sources = {k:original_xy(v) for k, v in f["data"].items()}
        for suite, target in registered["tasks"].items():
            for row_nr, fingerprint in target["layouts"].items():
                released = bank[suite]["rows"][int(row_nr)]
                if released["fingerprint"] != fingerprint:
                    raise ValueError("Released states changed since registration")
                xy = released["xy"]
                distances, different_sets = [], set()
                for demo, orig in sources.items():
                    if xy.keys() != orig.keys():
                        different_sets.add(tuple(sorted(orig.keys() ^ xy.keys())))
                        continue
                    joint = max(xy, key=lambda j: math.dist(orig[j], xy[j]))
                    distances.append((math.dist(orig[joint], xy[joint]), demo, joint))
                nearest = min(distances) if distances else None
                if nearest and nearest[0] <= target["comparison_tolerance_m"]:
                    raise ValueError(f"Cannot exclude original scene overlap for {task}/{suite}/{row_nr}/{nearest}")
                comparisons.append(dict(source_task=task, target=suite, released_row=int(row_nr),
                    original_demos=len(sources), staged_trajectories=[r["id"] for r in staged if r["prompt"] == task],
                    different_inventory_count=len(sources)-len(distances),
                    differing_joint_name_sets=sorted(different_sets),
                    nearest_same_inventory=dict(max_planar_separation_m=nearest[0], demo=nearest[1], joint=nearest[2])
                    if nearest else None))
    dump(output, dict(source_manifest=registered["data_manifest"],
         original_dataset="yifengzhu-hf/LIBERO-datasets", original_revision="f13aa24a3da8c43c7225569f28c562979fa0e35a",
         source_archives_sha256=sha(HERE / "original-files.json"), contract_sha256=sha(HERE / "contract.json"),
         comparisons=comparisons))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "manifest_path", "output"):
        p.add_argument(name, type=Path)
    audit(**vars(p.parse_args()))
