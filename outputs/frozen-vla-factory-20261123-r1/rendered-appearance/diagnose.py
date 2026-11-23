"""Oracle controls run after deployment sprites have been fixed."""

import argparse
import csv
from pathlib import Path

import numpy as np
from rig import PandaRig, render


def iou(a, b):
    return (a & b).sum() / (a | b).sum()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    root, out = args.root, args.output
    (root / "diagnostic").mkdir(exist_ok=True)
    rig = PandaRig()
    rows = []
    try:
        for path in sorted((root / "sprites").glob("*.npz")):
            with np.load(path) as dep, np.load(root / "oracle" / path.name) as ref, np.load(root / "public" / path.name) as pub:
                rig.set_pose(ref["q"], pub["grip"])
                rgb, mask = render(rig.env)
                np.savez_compressed(root / "diagnostic" / path.name, rgb=rgb, mask=mask)
                scene = ref["sm"]
                assert not (np.roll(dep["mask"], -128, axis=0) & (scene | ref["tm"])).any()
                assert not dep["mask"][:128].any()  # fixed sham has no wraparound
                rows.append(dict(task=str(pub["task"]), joint_rms_rad=np.sqrt(np.mean((dep["q"]-ref["q"])**2)),
                                 tcp_err_m=np.linalg.norm(dep["tool_error"][:3]), ori_err_rad=np.linalg.norm(dep["tool_error"][3:]),
                                 ik_true_rig_iou=iou(dep["mask"], mask), true_rig_scene_iou=iou(mask, scene),
                                 ik_scene_iou=iou(dep["mask"], scene), fixed_rig_true_pixels=mask.sum(),
                                 scene_occluded_pixels=(mask & ~scene).sum(), extra_scene_pixels=(scene & ~mask).sum(),
                                 true_rgb_mae_on_shared=np.abs(rgb.astype(float)-ref["source"])[mask & scene].mean(),
                                 target_only_fraction=(ref["tm"] & ~dep["mask"]).mean(),
                                 insert_fraction=dep["mask"].mean()))
    finally:
        rig.close()
    assert len(rows) == 10
    with (out / "geometry.csv").open("w") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
