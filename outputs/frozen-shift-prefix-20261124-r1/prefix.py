"""Input-side embodiment prefix for a frozen policy. No framework import: JAX and PyTorch share this file."""
import hashlib
import json
from pathlib import Path

import numpy as np

TOKENS = 8
ARMS = {"shift": "one vector added to every valid prompt/state token embedding",
        "tokens": f"{TOKENS} tokens appended after the prompt tokens"}
SCHEMA = "frozen-shift-prefix-v1"


def digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(repr((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def initial(arm, reference, seed):
    """`reference` is the mean valid prompt embedding of the first fit row, shape (width,)."""
    reference = np.asarray(reference, dtype=np.float32)
    if arm == "shift":
        return {"shift": np.zeros_like(reference)}  # the first step is the stock policy exactly
    if arm == "tokens":
        scale = .01 * float(np.sqrt(np.mean(reference ** 2)))
        jitter = np.random.default_rng([seed, 41]).standard_normal((TOKENS, len(reference)))
        return {"tokens": (reference + scale * jitter).astype(np.float32)}
    raise ValueError(arm)


def count(prefix):
    return int(sum(np.asarray(value).size for value in prefix.values()))


def random_like(prefix, seed):
    """Seeded random direction with the norm of each fitted array: the perturbation control."""
    rng = np.random.default_rng([seed, 43])
    out = {}
    for key in sorted(prefix):
        value = np.asarray(prefix[key], dtype=np.float32)
        draw = rng.standard_normal(value.shape).astype(np.float32)
        out[key] = draw * np.float32(np.linalg.norm(value) / np.linalg.norm(draw))
    return out


def inject(cat, cast, tokens, mask, ar, prefix, text):
    """Apply a prefix to the output of `embed_prefix`.

    tokens (b, s, w), mask (b, s) bool, ar (s,) bool. The last `text` positions are the
    prompt tokens. `cat(list, axis)` and `cast(array, like)` come from the framework.
    """
    if set(prefix) - {"shift", "tokens"}:
        raise ValueError(f"Unknown prefix arrays {sorted(prefix)}")
    if not 0 < text <= tokens.shape[1]:
        raise ValueError("Prompt tokens are not the tail of the prefix")
    if "shift" in prefix:
        shift = cast(prefix["shift"], tokens)
        if tuple(shift.shape) != (tokens.shape[-1],):
            raise ValueError("Shift width differs from the embedding width")
        gate = cast(mask[:, -text:, None], tokens)
        tokens = cat([tokens[:, :-text], tokens[:, -text:] + gate * shift], 1)
    if "tokens" in prefix:
        extra = cast(prefix["tokens"], tokens)
        if extra.ndim != 2 or extra.shape[-1] != tokens.shape[-1]:
            raise ValueError("Token width differs from the embedding width")
        n = extra.shape[0]
        tokens = cat([tokens, extra[None] + 0 * tokens[:, :1]], 1)
        mask = cat([mask] + [mask[:, :1] | True] * n, 1)
        ar = cat([ar] + [ar[:1] & False] * n, 0)
    return tokens, mask, ar


def save(path, prefix, receipt):
    path = Path(path)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in prefix.items()}
    meta = dict(receipt, schema=SCHEMA, parameters=count(arrays),
                sha256={key: digest(value) for key, value in sorted(arrays.items())})
    tmp = path.with_suffix(".part.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta, allow_nan=False)), **arrays)
    tmp.replace(path)
    return meta


def load(path):
    with np.load(Path(path), allow_pickle=False) as data:
        meta = json.loads(str(data["meta"]))
        prefix = {key: data[key] for key in data.files if key != "meta"}
    if meta["schema"] != SCHEMA or meta["sha256"] != {key: digest(value) for key, value in sorted(prefix.items())}:
        raise ValueError(f"Prefix file {path} does not match its receipt")
    return prefix, meta
