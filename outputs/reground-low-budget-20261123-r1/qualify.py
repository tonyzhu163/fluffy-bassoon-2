"""CPU readback through the native controller; a synthetic state/chunk fixture, outside scoring."""

import os

import numpy as np
import torch
from common import CELL, ref


def cpu_check(inputs, directory, deployment, verify_fitted):
    from reground.method import build

    receipts = {k: verify_fitted(inputs, directory, k, "cpu") for k in ("fwd", "idm")}
    # No real target fixture is exposed to this smoke. Use a synthetic identity orientation and a 50-action proposal.
    x = np.zeros((1, 25), np.float32)
    x[0, [3, 7]] = 1
    chunk = np.zeros((1, 50, 7), np.float32)
    chunk[:, :, 0] = 0.1
    chunk[:, :, 6] = -1
    with deployment(inputs, directory, "cpu"):
        ctrl = build.make_controller(
            "reground",
            inputs.cfg,
            CELL,
            inputs.seed,
            "cpu",
            fit_ok=False,
            policy="pi05",
        )
        ref.require(
            not ctrl.needs_sim and ctrl.shift == 0 and ctrl.cfg["anchor"] == "idm",
            "Wrong controller",
        )
        ctrl.reset(x)
        ctrl.on_chunk(x, chunk)
        action = ctrl.act(x, 0)
    ref.require(
        action.shape == (1, 7)
        and np.isfinite(action).all()
        and action[0, 6] == chunk[0, 0, 6],
        action,
    )
    rows = {}
    for kind, r in receipts.items():
        d = directory / "models" / inputs.record["windows"][kind]["spec_hash"]
        rows[kind] = {
            "model_sha256": r["model_sha256"],
            "metrics": ref.read(d / "fit_metrics.json"),
        }
    ref.write(
        directory / "cpu.json",
        {
            "seed": inputs.seed,
            "budget": inputs.budget,
            "torch": torch.__version__,
            "affinity": sorted(os.sched_getaffinity(0)),
            "threads": torch.get_num_threads(),
            "models": rows,
            "action": action.tolist(),
            "extra_target_fit": 0,
            "extra_target_deployment": 0,
            "simulator_episodes": 0,
            "subset_sha256": inputs.record["sha256"],
        },
    )
    print(f"Qualified seed {inputs.seed}, budget {inputs.budget}", flush=True)
