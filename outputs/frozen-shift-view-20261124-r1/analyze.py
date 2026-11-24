"""Archived-frame counts for the body-view edit. Reads the ten Kinova3 exposures of the source-view run."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "frozen-vla-source-view-20261123-r1"
CAMERAS = ("agentview", "robot0_eye_in_hand")


def colour_class(rgb):
    """Proxy scene classes for raw pixels; no segmentation is archived."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    table = (r > 150) & (g > 130) & (b > 100) & (r - b > 20)
    wall = (abs(r - g) < 12) & (abs(g - b) < 12) & (r > 90) & (r < 200)
    dark = rgb.max(axis=-1) < 70
    return table, wall & ~table, dark & ~table & ~wall


def main():
    policy = json.loads((SOURCE / "native-readback/frame-inputs.json").read_text())
    rows = []
    for layout in (14, 15):
        record = json.loads((SOURCE / f"evidence/cpu01/layout-{layout}.json").read_text())
        reset = np.load(SOURCE / f"evidence/cpu01/s{layout}-t000.npz")
        for frame in record["frames"]:
            step = frame["step"]
            data = np.load(SOURCE / f"evidence/cpu01/s{layout}-t{step:03d}.npz")
            row = dict(layout=layout, control=step, cameras={})
            for cam in CAMERAS:
                raw, sham, swap = (data[f"{kind}_{cam}"] for kind in ("raw", "control", "panda"))
                changed = np.any(raw != swap, axis=-1)
                # A changed pixel that equals the reset frame showed static content before the edit:
                # scene, or a Kinova link that has not moved (base, shoulder).
                static = changed & np.all(raw == reset[f"raw_{cam}"], axis=-1)
                table, wall, dark = colour_class(raw)
                stats = frame["images"][cam]
                row["cameras"][cam] = dict(
                    changed_256=int(changed.sum()), kinova_rerender_changed_256=int(np.any(raw != sham, axis=-1).sum()),
                    kinova_arm_px=stats["orig_arm_px"], panda_arm_px=stats["source_arm_px"],
                    scene_revealed_px=stats["uncovered_px"], scene_newly_hidden_px=stats["source_only_px"],
                    panda_px_behind_scene=stats["source_arm_hidden_px"],
                    changed_static_px=int(static.sum()), changed_static_table_px=int((static & table).sum()),
                    changed_static_wall_px=int((static & wall).sum()), changed_static_dark_px=int((static & dark).sum()),
                    changed_static_other_px=int((static & ~table & ~wall & ~dark).sum()),
                    changed_rows=[int(v) for v in np.flatnonzero(changed.any(axis=1))[[0, -1]]] if changed.any() else None,
                    wrist_camera_offset_m_rad=list(frame["panda"]["camera_before_lock"][cam].values()),
                )
            rows.append(row)
    (HERE / "frames.json").write_text(json.dumps(dict(frames=rows, policy_resolution=policy), indent=1) + "\n")
    keys = ("changed_256", "kinova_rerender_changed_256", "scene_newly_hidden_px", "scene_revealed_px",
            "panda_px_behind_scene", "changed_static_px", "changed_static_table_px", "changed_static_wall_px",
            "changed_static_dark_px", "changed_static_other_px")
    for cam in CAMERAS:
        print(cam, " ".join(keys))
        for row in rows:
            print(row["layout"], row["control"], *[row["cameras"][cam][k] for k in keys], row["cameras"][cam]["changed_rows"])
    offsets = np.array([row["cameras"]["robot0_eye_in_hand"]["wrist_camera_offset_m_rad"] for row in rows])
    print("wrist camera offset max m, rad", offsets.max(axis=0))


if __name__ == "__main__":
    main()
