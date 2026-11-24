"""One source-only choice from the archived original validation forward, without target access."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from algebra import ALPHAS, MULTIPLIERS, Support
from common import (REGISTRATION, ROW_KEYS, SEEDS, SOURCE_ARMS, VAL_MAPS, MODES, correct, digest,
                    historical, read, token_hash, validation_tasks, write)


def grid_cost(row, operators, prior):
    """Parent's bounded inverse and measured five-control forecast, vectorized over candidates."""
    q = row["q"]
    pred = np.clip(q @ np.linalg.pinv(operators).swapaxes(-1,-2), -1, 1)
    actual = correct(q, row["a"])
    command = 50*np.sqrt(np.mean(np.sum((pred-actual)**2, axis=-1), axis=1))
    motion = np.full(len(operators), np.nan)
    hc = row["hc"]
    if len(row["hd"]):
        xy = hc[:,:2] @ operators.swapaxes(-1,-2)
        current = np.concatenate((xy, np.broadcast_to(hc[:,2:3], (*xy.shape[:2], 1))), axis=2)
        lag = np.concatenate((np.zeros_like(current[:,:1]), current[:,:-1]), axis=1)
        forecast = (np.concatenate((current, lag), axis=2) @ prior[:,:2])[:,1:]
        motion = 1000*np.sqrt(np.mean(np.sum((forecast-row["hd"])**2, axis=2), axis=1))
    return np.stack([command, motion], axis=1)


def main(out, forward):
    receipt = read(forward/"forward-receipt.json")
    assert receipt["registration"] == REGISTRATION
    for file, sha in receipt["files"].items():
        assert digest(forward/file) == sha
    hp = read(forward/"upstream-selected.json")
    assert receipt["checkpoints"] == hp["checkpoints"]
    fit = historical()
    assert hp["residual"] == fit["residual_power"] and hp["prior"] == fit["prior"]
    ids = read(forward/"validation-index.json")
    val_tasks = validation_tasks(fit)
    expected = {(arm, f"{s}/{t}", family, mode) for s,t in val_tasks.items() for arm in SOURCE_ARMS
                for family in VAL_MAPS for mode in MODES}
    assert len(ids) == len(expected) and {tuple(r["key"]) for r in ids} == expected
    prior = np.asarray(hp["prior"])
    groups = defaultdict(list)
    with np.load(forward/"validation-cases.npz") as rows, np.load(forward/"validation-direct.npz") as direct:
        assert set(direct.files) == {str(s) for s in SEEDS}
        estimates = {s: direct[str(s)] for s in SEEDS}
        assert all(a.shape == (320,2,2) and np.isfinite(a).all() for a in estimates.values())
        for i, meta in enumerate(ids):
            arm, task, family, mode = meta["key"]
            if not family.startswith("general") or mode.startswith("along"):
                continue
            row = {k: rows[f"{i}_{k}"] for k in ("tok","x","y","q","a","hc","hd")}
            assert token_hash(row["tok"]) == meta["token_sha256"]
            support = Support(row["x"], row["y"], hp["residual"])
            for seed in SEEDS:
                d = estimates[seed][i]
                spectral = [support.blended(d, m) for m in MULTIPLIERS]
                scalar_grid = support.r + ALPHAS[:,None,None]*(d-support.r)
                scalar_grid[0], scalar_grid[-1] = support.r, d
                cost = grid_cost(row, np.concatenate((spectral, scalar_grid)), prior)
                # Original D was evaluated with float32 pinv, R and blended maps use float64.
                command_d = 50*np.sqrt(np.mean(np.sum((correct(row["q"], d)-correct(row["q"], row["a"]))**2, axis=1)))
                cost[[len(MULTIPLIERS)-1,-1],0] = command_d
                groups[task,arm,family,"passive" if mode == "passive" else "lateral"].append(cost)
    # Equal weight for each of the 4 × 2 × 3 × 2 cells, averaging seeds and pulse signs inside.
    assert len(groups) == 48 and all(len(v)==(3 if k[-1]=="passive" else 6) for k,v in groups.items())
    values = np.stack([np.nanmean(v, axis=0) for v in groups.values()])
    assert np.isfinite(values[:,:,0]).all()
    available = np.isfinite(values[:,0,1])
    assert np.all(np.isfinite(values[:,:,1]) == available[:,None])
    scores = np.stack((values[:,:,0].mean(0), values[available,:,1].mean(0)),axis=-1)
    assert np.isfinite(scores).all()
    n = len(MULTIPLIERS)
    curves = {}
    for name, grid, costs in (("lambda", MULTIPLIERS, scores[:n]), ("alpha", ALPHAS.tolist(), scores[n:])):
        curves[name] = [dict(parameter=v, command=float(c), motion=float(m)) for v, (c,m) in zip(grid, costs)]
    winners = {name: min(entries, key=lambda e: (e["command"], e["motion"])) for name, entries in curves.items()}
    np.testing.assert_array_equal(scores[[0,n-1]], scores[[n,-1]])
    write(out/"validation-curves.json", curves)
    write(out/"source-selection.json", dict(registration=REGISTRATION, winners=winners,
          selected_lambda_multiplier=winners["lambda"]["parameter"], selected_alpha=winners["alpha"]["parameter"],
          normalized_k=hp["residual"]/0.05**2, ridge_multiplier=hp["ridge_multiplier"],
          checkpoints=hp["checkpoints"], upstream_selected=receipt["selected_sha256"],
          source_forward=str(forward), validation_receipt=digest(forward/"forward-receipt.json"),
          validation_files=receipt["files"], curve_sha256=digest(out/"validation-curves.json"),
          arm_task_family_strategy_cells=48, motion_cells=int(available.sum()),
          source_validation_reuse="Perceiver training recipe and original standalone ridge were selected here."))
    # Write completion marker last, before this program exits. No target path is accepted.
    (out/"source-selection.sha256").write_text(digest(out/"source-selection.json")+"\n")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
