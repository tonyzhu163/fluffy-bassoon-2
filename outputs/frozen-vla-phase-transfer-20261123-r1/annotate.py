"""Label source fitting examples by visible drawer-front displacement. No simulator-state access."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import save


def front(frame):
    # Upright 128 x 128 source camera. Scan from inside the dark wooden cabinet
    # toward the bright table. The median across eight adjacent lines reduces handle/plate interference.
    assert frame.shape == (128,128,3)
    gray = frame[::-1, :, :].mean(axis=2)
    scan = gray[73:81,15:88] >= 90
    assert np.all(scan.any(axis=1))
    return float(np.median(15 + np.argmax(scan,axis=1)))


def annotate(source, directory):
    ids = json.loads((directory/"split.json").read_text())["fit"]
    labels, audit = {}, {}
    with h5py.File(source,"r") as archive:
        for i in ids:
            images = archive[f"data/demo_{i}/obs/agentview_rgb"]
            t = np.load(directory/f"public-{i}.npz")["t"]
            edge = np.array([front(images[j]) for j in t])
            end = front(images[-1])
            assert end > edge[0]
            # Near-full visible travel. The native cabinet's 14 cm threshold is seven
            # eighths of nominal 16 cm travel. Perspective and occlusion can bias this proxy.
            after = edge >= edge[0] + (end-edge[0]) * 7/8
            on = np.flatnonzero(after[1:] & after[:-1])
            assert len(on), i
            labels[i] = int(t[on[0]])
            audit[i] = dict(t=t.tolist(),edge_px=edge.tolist(),end_px=end,switch=labels[i])
    save(directory/"annotations.json",labels)
    save(directory/"annotation-audit.json",audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("directory",type=Path)
    a = p.parse_args()
    annotate(a.source,a.directory)
