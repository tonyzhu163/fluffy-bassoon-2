"""After an independently read-back source seal, replay original fresh D on matching evidence."""
from pathlib import Path
import sys

import numpy as np
from algebra import Support, scalar
from common import (REGISTRATION, SEEDS, LAW, archive_cases, copy_verified, digest, historical, key, law,
                    metrics, original_rows, read, ridge, source_identity, token_hash, write)
from summarize import summarize


def references(model, rows, paths, select):
    published = read(model/"fresh-files.json")
    assert all(paths.get(p) == h for p,h in published.items())
    assert set(published) == {p for p in paths if not Path(p).name.startswith("runtime-")}
    data = read(model/"fresh-rows.json")
    direct, physical = {}, {}
    for r in data:
        if r["method"] == "direct":
            k = (*key(r), r["seed"])
            assert k not in direct and r["split"] == "target"
            direct[k] = r
        if r["method"] in ("incumbent","ridge","true_map"):
            k = (*key(r), r["method"])
            assert k not in physical and r["split"] == "target"
            physical[k] = r
    keys = [key(r) for r in rows]
    assert set(direct) == {(*k,s) for k in keys for s in SEEDS}
    assert set(physical) == {(*k,m) for k in keys for m in ("incumbent","ridge","true_map")}
    return direct, physical


def main(out, selection, model, bank, source_hash):
    assert digest(selection/"source-selection.json") == source_hash
    assert (selection/"source-selection.sha256").read_text().strip() == source_hash
    choice = read(selection/"source-selection.json")
    assert choice["registration"] == REGISTRATION
    assert digest(selection/"validation-curves.json") == choice["curve_sha256"]
    forward = Path(choice["source_forward"])
    assert digest(forward/"forward-receipt.json") == choice["validation_receipt"]
    receipt = read(forward/"forward-receipt.json")
    assert receipt["files"] == choice["validation_files"]
    assert Path(receipt["model"]).resolve() == model.resolve() and Path(receipt["bank"]).resolve() == bank.resolve()
    select = source_identity(model, bank, require_seal=True)
    assert select["checkpoints"] == choice["checkpoints"] and digest(model/"selected.json")==choice["upstream_selected"]
    prior, residual = np.array(select["prior"]), select["residual"]
    assert select["ridge_multiplier"] == choice["ridge_multiplier"]
    # All target data access is after the verified independent source freeze.
    rows, paths = original_rows(bank, "target", historical())
    direct, physical = references(model, rows, paths, select)
    checksum = {}
    for name in ("sealed.json","selected.json","fresh-rows.json","fresh-files.json","source-test.json"):
        checksum[str(model/name)] = digest(model/name)
        copy_verified(model/name,out/f"upstream-{name}",checksum[str(model/name)])
    archive_cases(out/"target-cases.npz", rows)
    readback = []
    max_diff = {m:{v:0. for v in ("command","motion")} for m in ("direct","incumbent","ridge","true_map")}
    for i,r in enumerate(rows):
        k = key(r)
        s = Support(r["x"],r["y"],residual)
        weights = s.weights(choice["selected_lambda_multiplier"])
        pulse = r["issued"][:,:2] - np.repeat(r["baseline"][:,:2].astype(np.float32),5,axis=0)
        rr = ridge(r["x"],r["y"],residual,choice["ridge_multiplier"])
        for seed in SEEDS:
            drow = direct[(*k,seed)]
            d = np.array(drow["estimate"], dtype=np.float32)
            np.testing.assert_array_equal(d, drow["estimate"])
            estimators = dict(direct=d, incumbent=s.r, ridge=rr, scalar=scalar(s.r,d,choice["selected_alpha"]),
                              true_map=r["a"])
            for variant in ("directional","trace","reversed"):
                estimators[variant] = s.blended(d,choice["selected_lambda_multiplier"],variant)
            scores = {m:metrics(r,op,prior) for m,op in estimators.items()}
            for m in max_diff:
                original = drow if m == "direct" else physical[(*k,m)]
                assert (original["query_controls"],original["held_chunks"]) == (len(r["q"]),len(r["hd"]))
                np.testing.assert_array_equal(r["a"],original["actual"])
                np.testing.assert_allclose(estimators[m],original["estimate"],rtol=0,atol=2e-12)
                for v in ("command","motion"):
                    value, published = scores[m][v], original[v]
                    if value is None:
                        assert published is None
                    else:
                        max_diff[m][v] = max(max_diff[m][v],abs(value-published))
                        # NumPy/LAPACK may differ by runtime; direct pinv is float32.
                        np.testing.assert_allclose(value,published,rtol=2e-5 if m=="direct" else 1e-10,
                                                   atol=1e-5 if m=="direct" else 1e-10)
            forecasts = {m:law(r["hc"],op,prior)[1:].tolist() if len(r["hd"]) else []
                         for m,op in estimators.items()}
            readback.append(dict(**{n:r[n] for n in ("arm","task","family","mode","strategy","prefix_drift",
                                                     "rank_ratio","effort","bound_hits")}, seed=seed,
                case_index=i, token_sha256=token_hash(r["tok"]), held_chunks=len(r["hd"]), query_controls=len(r["q"]),
                pulse_energy=float((pulse**2).sum()),pulse2=float(np.linalg.norm(pulse[5])),
                pulse3=float(np.linalg.norm(pulse[10])), g=s.g.tolist(),weights=weights.tolist(),
                log_min_information=float(np.log1p(s.g[0]/s.k)),weight_range=float(np.ptp(weights)),
                maps={m:op.tolist() for m,op in estimators.items()},scores=scores, predicted_xy_m=forecasts))
    assert len(readback) == 600
    write(out/"target-rows.json",readback)
    write(out/"target-summary.json",summarize(readback))
    write(out/"target-inputs.json",paths)
    names = ("target-rows.json","target-summary.json","target-inputs.json","target-cases.npz")
    write(out/"target-receipt.json",dict(source_sha256=source_hash, registration=REGISTRATION, source_law=LAW,
          original=checksum, selected_checkpoints=select["checkpoints"], upstream_score_max_abs_difference=max_diff,
          files={f:digest(out/f) for f in names}, supports=len(rows), seed_rows=len(readback),
          validation_tasks=sorted({r["key"][1] for r in read(forward/"validation-index.json")}),
          target_tasks=sorted({r["task"] for r in rows})))


if __name__ == "__main__":
    main(*map(Path,sys.argv[1:5]),sys.argv[5])
