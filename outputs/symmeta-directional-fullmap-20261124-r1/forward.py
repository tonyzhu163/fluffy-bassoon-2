"""Validation-only batched inference with the originally selected parameters."""
from pathlib import Path
import sys
import time

import jax
import numpy as np
from common import (REGISTRATION, SEEDS, archive_cases, copy_verified, digest, historical, key,
                    original_rows, source_identity, token_hash, write, law, metrics, ridge)
from learn import predict, restore


def main(out, model, bank):
    assert jax.default_backend() == "gpu" and len(jax.devices()) == 1
    start = time.monotonic()
    selected = source_identity(model, bank)
    # The model may continue its independent tests. selected.json already freezes all three weights.
    copy_verified(model / "selected.json", out / "upstream-selected.json", digest(model / "selected.json"))
    rows, paths = original_rows(bank, "val", historical())
    assert len(rows) == 320
    tokens = np.stack([r["tok"] for r in rows])
    pred = {}
    for seed in SEEDS:
        file = f"seed-{seed}.pkl"
        assert digest(model / file) == selected["checkpoints"][file]
        copy_verified(model / file, out / file, selected["checkpoints"][file])
        weights = restore(out / file)
        values = np.asarray(predict(weights, tokens))
        assert values.shape == (320, 2, 2) and np.isfinite(values).all()
        pred[str(seed)] = values
    prior = np.asarray(selected["prior"])
    readback = []
    for i, row in enumerate(rows):
        maps = {f"direct-{s}":pred[str(s)][i] for s in SEEDS}
        maps["incumbent"] = ridge(row["x"], row["y"], selected["residual"])
        maps["ridge"] = ridge(row["x"], row["y"], selected["residual"], selected["ridge_multiplier"])
        physical = {}
        for name, op in maps.items():
            forecast = law(row["hc"],op,prior)[1:] if len(row["hd"]) else np.empty((0,2))
            score = metrics(row,op,prior)
            physical[name] = dict(estimate=op.tolist(), predicted_xy_m=forecast.tolist(),
                                  command=score["command"], motion=score["motion"])
        readback.append(dict(key=key(row), observed_xy_m=row["hd"].tolist(), predictions=physical))
    write(out / "validation-forecasts.json", readback)
    archive_cases(out / "validation-cases.npz", rows)
    np.savez_compressed(out / "validation-direct.npz", **pred)
    write(out / "validation-index.json", [dict(key=key(r), token_sha256=token_hash(r["tok"])) for r in rows])
    write(out / "validation-inputs.json", paths)
    files = ["upstream-selected.json", "validation-cases.npz", "validation-direct.npz",
             "validation-index.json", "validation-inputs.json", "validation-forecasts.json", *selected["checkpoints"]]
    write(out / "forward-receipt.json",
          dict(registration=REGISTRATION, model=str(model), bank=str(bank), selected_sha256=digest(model / "selected.json"),
               checkpoints=selected["checkpoints"], files={f: digest(out / f) for f in files},
               source_elapsed_seconds=time.monotonic()-start, device=str(jax.devices()[0]),
               gate_inputs="D from the 3x100 causal tokens; R and G from the same design and 3 observed XY motions"))
    print("Archived all original validation inputs and three frozen readbacks.", flush=True)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
