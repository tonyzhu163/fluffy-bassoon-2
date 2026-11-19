"""Compare recognition on fitting and fresh poses with the native outcome ceiling."""
import argparse
import json
from pathlib import Path

import analyze
import readers


def recognition(panel, model):
    cells = {}
    for (pose, wire, condition), case in panel.items():
        key = f"w{wire:+d}_{condition}"
        counts = cells.setdefault(key, {})
        practice = case["support"]
        for kind in ("factored", "joint"):
            for seed in readers.SEEDS:
                sign, grip = model.predict(practice.commands, practice.states, kind, seed)
                row = counts.setdefault(f"{kind}/{seed}", dict(total=0, wire=0, grip=0, both=0))
                row["total"] += 1
                row["wire"] += int(sign == wire)
                row["grip"] += int(grip == int(condition == "D17"))
                row["both"] += int((sign, grip) == (wire, int(condition == "D17")))
    return cells


def main(out):
    pairing = analyze.Pairing()
    fit = analyze.read_stage(out, True, pairing)
    panel = analyze.read_stage(out, False, pairing)
    model = readers.load(out / "weights")
    groups = {}
    for (pose, wire, condition), case in panel.items():
        group = groups.setdefault(f"w{wire:+d}_{condition}", [])
        off = case[analyze.branch_name(wire, 0)].success
        on = case[analyze.branch_name(wire, 1)].success
        group.append(dict(pose=pose, query_pose=pose + 20, no_regrasp=off, regrasp=on,
                          measured_static_union=any(case[analyze.branch_name(*choice)].success
                                                    for choice in analyze.CHOICES)))
    spec = dict(fit_resubstitution=recognition(fit, model), fresh_poses=recognition(panel, model),
                native_grasp_pairs=groups, unused_reset_image_differences=len(pairing.reset_camera_differences),
                post_probe_differences=len(pairing.opening_differences))
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
