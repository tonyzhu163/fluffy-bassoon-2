"""Materialize the fixed development subset and inspect it in the simulator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

FILES = Path(__file__).resolve().parent
CODE = FILES.parents[1]
sys.path.insert(0, str(CODE / "exploration/libero-scenes"))
from scenes import to_bddl
from histories import label_scenes, placement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene_dir = args.output / "scenes"
    scene_dir.mkdir()
    manifest = json.loads((FILES / "SCENES.json").read_text())
    (scene_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for scene in manifest["scenes"]:
        text = to_bddl(scene)
        assert hashlib.sha256(text.encode()).hexdigest() == scene["bddl_sha256"]
        (scene_dir / scene["bddl"]).write_text(text)
    subprocess.run([sys.executable, "-Bu", str(CODE / "exploration/libero-scenes/admit.py"),
                    "--source", str(args.source), "--scenes", str(scene_dir),
                    "--output", str(args.output / "inspection"), "--image-size", "256"], check=True)
    inspected = json.loads((args.output / "inspection/result.json").read_text())
    records = {r["index"]: r for r in inspected["records"]}
    comparisons = []
    arrays = {}
    for scene in manifest["scenes"]:
        record = records[scene["index"]]
        if "world_positions" not in record:
            continue
        pos = record["world_positions"]
        bowls = np.array([[pos[f"akita_black_bowl_{i}"][:2] for i in (1, 2)]])
        rec = np.array([[pos[n][:2] for n in ("plate_1", "glazed_rim_porcelain_ramekin_1",
                                             "cookies_1", "flat_stove_1", "wooden_cabinet_1")]])
        actual, _ = placement.behaviours(bowls, rec, ("anchor-if",), False)
        actual = (actual // (placement.N_PLACE * placement.N_THEN)).astype(np.uint8)[0]
        nominal, _ = label_scenes([scene])
        arrays[f"scene_{scene['index']}"] = actual
        comparisons.append(dict(index=scene["index"], rules=len(actual),
                                changed_from_nominal=int(np.count_nonzero(actual != nominal[0])),
                                source_state_sha256=record["state_sha256"]))
    np.savez_compressed(args.output / "realized-rule-labels.npz", **arrays)
    result = dict(selection=manifest["selection"], policy_loaded=False,
                  rule_frame="settled world XY; translation-invariant table relations",
                  rule_comparisons=comparisons, inspection=inspected,
                  limitation="Predicate drop controls are not robot reachability demonstrations.")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SCENE_QUALIFICATION", json.dumps({"scenes":len(records), "admitted":inspected["admitted"],
          "rule_comparisons":comparisons, "rejection_reasons":inspected["rejection_reasons"]}), flush=True)


if __name__ == "__main__":
    main()
