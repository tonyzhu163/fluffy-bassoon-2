"""Opt-in, read-only tap on the released VLA. Caller owns scene capture and scoring."""
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

UPSTREAM = "16a98486f560676b9036d1a6d067b9dd51a5917a"
POLICY = "7152f13e34c6c115d808aea0e5d08aca74eac14b"


def pack(value, arrays):
    """Type-preserving manifest and lossless bytes, including torch.bfloat16."""
    import torch
    if isinstance(value, dict):
        return {k: pack(value[k], arrays) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [pack(item, arrays) for item in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if torch.is_tensor(value):
        t = value.detach().contiguous()
        data = t.reshape(-1).view(torch.uint8).cpu().numpy().copy()
        shape, dtype = list(t.shape), str(t.dtype)
    else:
        a = np.asarray(value)
        if a.dtype.hasobject:
            raise TypeError("Object arrays cannot be captured losslessly")
        data = np.ascontiguousarray(a).reshape(-1).view(np.uint8).copy()
        shape, dtype = list(a.shape), a.dtype.str
    key = f"array{len(arrays):04d}"
    arrays[key] = data
    return dict(array=key, shape=shape, dtype=dtype, sha256=sha256(data).hexdigest())


@contextmanager
def capture_first_query(vla, folder, batch_keys, state_receipt, upstream_revision):
    """Wrap ONE native batch. Preserve the method return value and every subsequent query.

    state_receipt() must return the per-lane native reset receipts captured by the worker.
    It is called at the actual first query, after the native reset. All four ordered keys
    include adaptation seed, task, init_id, rep and original slot. Supply a fresh directory.
    """
    if (upstream_revision,vla.revision)!=(UPSTREAM,POLICY) or len(batch_keys)!=4:
        raise ValueError("Unexpected source, VLA or batch size")
    folder = Path(folder)
    folder.mkdir(parents=False, exist_ok=False)
    outer, inner = vla.predict_chunk, vla.policy.predict_action_chunk
    count = 0

    def record(observations, tasks, seeds, query):
        nonlocal count
        if query != 0:
            return outer(observations,tasks,seeds,query)
        if count or not (len(observations)==len(tasks)==len(seeds)==4):
            raise ValueError("Expected one full opening")
        count += 1
        arrays = {}
        raw = pack(observations,arrays)
        calls = []

        def trace_inner(obs, *, noise):
            calls.append(dict(processed=pack(obs,arrays),noise=pack(noise,arrays)))
            return inner(obs,noise=noise)

        with patch.object(vla.policy,"predict_action_chunk",trace_inner):
            result=outer(observations,tasks,seeds,query)
        if len(calls)!=1 or result.shape!=(4,50,7) or result.dtype!=np.float32:
            raise ValueError("Unexpected native inference interface")
        manifest = dict(upstream=UPSTREAM,policy=POLICY,batch_keys=batch_keys,query=query,
                        tasks=list(tasks),seeds=list(map(int,seeds)),raw=raw,**calls[0],
                        reset=state_receipt(),proposal=pack(result,arrays))
        np.savez_compressed(folder/"arrays.npz",**arrays)
        (folder/"receipt.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
        return result

    with patch.object(vla,"predict_chunk",record):
        yield
    if count!=1:
        raise ValueError("Opening was not captured")
