"""Two-draw source-only CPU check of the compiled adapter loss, optimizer, and inference routes."""
import argparse
from pathlib import Path
import os
from unittest.mock import patch

import jax
import numpy as np

import fit_plate
from plate_policy import FITS, PlatePolicy, write_json
from cotrain_policy import CHECKPOINT, array_sha
from flow_head import initial_params
from openpi.policies import policy_config
from openpi.training import config
from preflight import CPUFixture
from source_data import SourceDataset
from train import samples_for_split


def main(out):
    assert jax.default_backend() == "cpu" and len(os.sched_getaffinity(0)) <= 2
    cfg = config.get_config("pi05_libero")
    with patch("openpi.models.model.restore_params", return_value={}), patch.object(
            type(cfg.model), "load", return_value=CPUFixture()):
        stock = policy_config.create_trained_policy(cfg, CHECKPOINT)
    ds = SourceDataset("/mowenjie/fluffy-bassoon-2/outputs/frozen-vla-cotraining-20261124-r1/data/dataset-v1")
    rows = [row for row in (ds[i] for i in range(12)) if np.any(row["history"])][:2]
    assert len(rows) == 2
    engine = fit_plate.engine(stock._model)
    params = {c: initial_params(1024, "adapter_only") for c in FITS}
    train = {}
    for c, row in zip(FITS, rows, strict=True):
        d = out / c
        d.mkdir()
        train[c] = samples_for_split(stock, [row], "train", d)
    with patch.object(fit_plate, "EPOCHS", 1):
        params, progress = fit_plate.optimize(engine, params, train, out, "fixture")
    val = samples_for_split(stock, rows, "heldout", out)
    scores = fit_plate.evaluate(engine, params, val, [{"body":"fixture", "layout":i} for i in range(2)])
    motor = PlatePolicy(stock, params)
    parity = fit_plate.source_sample(motor, rows[0], np.random.default_rng(11).normal(size=(10,32)).astype(np.float32))
    assert len({array_sha(params[c]["history_out"]) for c in FITS}) == 2
    assert len({parity["nonempty_history"][c]["route"]["hlo_sha256"] for c in FITS}) == 1
    write_json(out / "qualify.json", dict(native=parity, score=scores, update=progress,
               cpu_affinity=sorted(os.sched_getaffinity(0)), model="pilot lightweight tower plus restored stock projection"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    main(p.parse_args().out)
