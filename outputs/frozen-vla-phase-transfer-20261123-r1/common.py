"""Shared public camera representation. Raw LIBERO images have the same storage convention on both arms."""
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tiles(frame):
    """16 by 16 equal-area mean luminances from the unflipped public agent camera."""
    h, w, c = frame.shape
    assert c == 3 and h % 16 == w % 16 == 0
    rgb = frame.reshape(16, h // 16, 16, w // 16, 3).mean(axis=(1, 3))
    return (rgb @ np.array([0.299, 0.587, 0.114]) / 255).ravel()
