"""Split by whole human demonstrations. Keep the physical label in a separate scoring step."""
import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw

from common import save, sha, tiles

SHA = "1b50d99a053a7e4547c9ae9ccd2e3cd762543d27be78f240f13440d7ae487ae8"
ROWS = [14, 15, 16, 17, 30, 31, 32, 33, 36, 37, 38, 39]


def offsets(group):
    body = ET.fromstring(group.attrs["model_file"]).find("worldbody")
    result, addr = {}, 1  # flattened state begins with time
    for el in body.iter():
        if el.tag not in ("joint", "freejoint"):
            continue
        result[el.get("name")] = addr
        addr += 7 if el.tag == "freejoint" or el.get("type") == "free" else 1
    return result


def split(f, bank_path):
    import torch

    bank = np.asarray(torch.load(bank_path, map_location="cpu", weights_only=False))
    starts = {}
    for i in range(50):
        group = f[f"data/demo_{i}"]
        off = offsets(group)
        movable = [off[name] for name in ("akita_black_bowl_1_joint0", "cream_cheese_1_joint0",
                                         "wine_bottle_1_joint0", "plate_1_joint0")]
        assert movable == [10, 17, 24, 31]  # released Panda and HDF5 state order
        starts[i] = group["states"][0]
    ix = np.array([np.arange(j, j + 3) for j in movable])
    assert bank.ndim == 2 and bank.shape[1] == len(starts[0])
    def distance(a, b):
        return float(np.linalg.norm(a[ix] - b[ix], axis=-1).max())
    lineage = {}
    for i, start in starts.items():
        distances = {f"released-{r}": distance(start, bank[r]) for r in ROWS}
        if i >= 20:
            distances.update({f"held-demo-{j}": distance(start, starts[j]) for j in range(10)})
        nearest = min(distances, key=distances.get)
        lineage[i] = dict(nearest=nearest, max_object_separation_m=distances[nearest],
                          target_separation_m={r: distances[f"released-{r}"] for r in ROWS})
    return dict(fit=[i for i in range(20, 50) if lineage[i]["max_object_separation_m"] > 0.001],
                held=list(range(10)), quarantine=list(range(10, 20)), initial_scene_audit=lineage,
                bank_path=str(bank_path), bank_sha256=sha(bank_path))


def prepare(path, out):
    assert sha(path) == SHA
    bank = (Path(os.environ["CROSSARM_RELEASE"]) / "simulator-source/libero/libero/init_files/libero_goal"
            / "open_the_top_drawer_and_put_the_bowl_inside.pruned_init")
    with h5py.File(path, "r") as f:
        info = split(f, bank)
        fit = info["fit"]
        for i in fit + info["held"]:
            g = f[f"data/demo_{i}"]
            obs = g["obs"]
            t = np.arange(0, len(obs["agentview_rgb"]) - 1, 5)
            frames = [obs["agentview_rgb"][j] for j in t]
            np.savez_compressed(out/f"public-{i}.npz", t=t, tiles=np.stack([tiles(v) for v in frames]))
            if i in fit:
                # Source-annotation sheet includes no simulator state or target images.
                selection = [0, *range(20, 131, 5), -1]
                contact = Image.new("RGB", (len(selection)*94, 115), "white")
                draw = ImageDraw.Draw(contact)
                for col, k in enumerate(selection):
                    j = len(t)-1 if k == -1 else min(k//5, len(t)-1)
                    img = Image.fromarray(frames[j][::-1]).resize((94,94))
                    contact.paste(img, (col*94, 0))
                    draw.text((col*94+2,95), f"{i}/{t[j]}", fill="black")
                contact.save(out/f"demo-{i}.jpg", quality=94)
        info.update(original_sha256=SHA, original_name=path.name, public_source="obs/agentview_rgb",
                    affinity=sorted(os.sched_getaffinity(0)))
        save(out/"split.json", info)


def score_labels(path, out):
    import json
    info = json.loads((out/"split.json").read_text())
    with h5py.File(path, "r") as f:
        for i in info["held"]:
            g = f[f"data/demo_{i}"]
            states = g["states"][:]
            t = np.load(out/f"public-{i}.npz")["t"]
            joints = g["obs/joint_states"][:-1]
            dist = [float(np.sqrt(np.mean((states[k:k+len(joints),1:8] - joints)**2))) for k in (0,1)]
            offset = int(np.argmin(dist))
            drawer = states[:,offsets(g)["wooden_cabinet_1_top_level"]]
            ever = np.maximum.accumulate(drawer < -0.14)
            save(out/f"truth-{i}.json", dict(t=t.tolist(), y=ever[t+offset].astype(int).tolist(),
                 joint_alignment_rmse_rad=dist, state_offset=offset, q=drawer[t+offset].tolist(),
                 first=int(np.flatnonzero(ever)[0])-offset if ever.any() else None))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "score"])
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    a.out.mkdir(exist_ok=True)
    (prepare if a.mode == "prepare" else score_labels)(a.source, a.out)
