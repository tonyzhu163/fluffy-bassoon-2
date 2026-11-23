"""Compare exact image quarter turns with previously saved native MuJoCo camera rolls."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for original in sorted(args.audit.glob("*/roll+0.png")):
        base = rgb(original)
        for actual in (-90, 90):
            observed = rgb(original.with_name(f"roll{actual:+d}.png"))
            row = dict(scene=original.parent.name, camera_angle=actual)
            for sign, label in ((1, "same_sign"), (-1, "opposite_sign"), (0, "no_rotation")):
                predicted = np.rot90(base, k=sign * actual // 90)
                row[label + "_rgb_mae"] = float(np.abs(predicted - observed).mean())
            rows.append(row)
    if len(rows) != 32:
        raise ValueError(f"Expected 16 scenes and two physical rolls, found {len(rows)} comparisons")
    means = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k.endswith("_mae")}
    result = {"mean_rgb_error_on_0_to_255_scale": means, "paired_renders": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(means)


if __name__ == "__main__":
    main()
