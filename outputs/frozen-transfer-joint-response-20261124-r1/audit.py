"""Read saved controls independently and reproduce recent/stale forecasts by augmented OLS."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def audit(out):
    results = json.loads((out/"results.json").read_text())
    fits = json.loads((out/"source-fits.json").read_text())
    with (out/"forecasts.csv").open() as f:
        forecasts = {(r["key"], r["arm"], int(r["zero_based_control"]), r["method"]): r
                     for r in csv.DictReader(f)}
    receipts = []
    for p in sorted(out.glob("libero*.json")):
        r = json.loads(p.read_text())
        key = p.stem
        cfg = fits[str(r["case"]["layout"])]
        selector = cfg["model"]
        assert len(cfg["source_keys"]) == 4 and all(
            k.rsplit("-", 1)[1] != str(r["case"]["layout"]) for k in cfg["source_keys"])
        pair = {}
        for arm in ("Panda", "Kinova3"):
            tape = out/f"{key}-{arm}.npz"
            assert hashlib.sha256(tape.read_bytes()).hexdigest() == r[arm]["raw_sha256"]
            with np.load(tape, allow_pickle=False) as a:
                n = len(a["actions"])
                assert a["actions"].shape == (n, 8) and np.max(abs(a["actions"])) <= 1
                assert np.array_equal(a["actions"][:, -1], a["proposal"][:, -1])
                assert np.allclose(np.diff(a["time"]), .05, rtol=0, atol=1e-12)
                assert a["rewards"].shape == (n,) and not a["rewards"][:-1].any()
                assert a["rewards"][-1] == r[arm]["success"]
                queries = np.arange(10, n, 11)
                inputs = np.stack((.05*a["actions"][:, :7], .05*a["v"][:-1]), axis=-1)
                dq = np.diff(a["q"], axis=0)
                prior = np.array(cfg["coefficients"][selector])
                scale = np.array(cfg["feature_rms_rad"])
                errors = {k: [] for k in ("source", "recent", "stale")}
                difference = []
                for i in queries:
                    for meth in errors:
                        m = prior.copy()
                        if meth != "source":
                            start = i-5 if meth == "recent" else i-10
                            h = np.arange(start, start+5)
                            assert np.all(h < i)
                            for j in range(7):
                                x = inputs[h, j]
                                # Solve for unscaled absolute coefficients with a ridge pseudo-observation.
                                regularizer = np.diag(scale*np.sqrt(cfg["weight"]))
                                design = np.vstack([x, regularizer])
                                target = np.r_[dq[h, j], regularizer @ prior[j]]
                                m[j] = np.linalg.lstsq(design, target, rcond=None)[0]
                        pred = a["J"][i, :3] @ (inputs[i]*m).sum(axis=1)
                        observed = a["xyz"][i+1] - a["xyz"][i]
                        err = 1e3*np.linalg.norm(pred-observed)
                        archived = forecasts[key, arm, i, meth]
                        difference.append(abs(err-float(archived["mm"])))
                        errors[meth].append(err)
                assert max(difference) < 1e-8
                for meth, values in errors.items():
                    assert np.isclose(np.mean(values), results[arm]["trajectories"][key][meth]["mm"],
                                      rtol=0, atol=1e-8)
                sat = np.any(abs(a["actions"][:, :7]) == 1, axis=1)
                contacts = a["external_contacts"] > 0
                pair[arm] = dict(controls=n, queries=len(queries), saturated=int(sat.sum()),
                                 contacts=int(contacts.sum()), torque_caps=int(sum(a["torque_limited"])),
                                 query_contacts=int(contacts[queries].sum()), query_sat=int(sat[queries].sum()),
                                 query_mean_actual_mm=np.mean(np.linalg.norm(np.diff(a["xyz"], axis=0)[queries], axis=1))*1000,
                                 max_error_reproduction_mm=max(difference), success=r[arm]["success"],
                                 q_clock_rad=r[arm]["q_clock_max_rad"], v_clock_rad_s=r[arm]["velocity_clock_max_rad_s"],
                                 xyz_clock_m=r[arm]["tcp_clock_max_m"])
                pair[arm]["proposal"] = a["proposal"]
        x, y = pair["Panda"].pop("proposal"), pair["Kinova3"].pop("proposal")
        n = min(len(x), len(y))
        assert np.array_equal(x[:n], y[:n])
        receipts.append(dict(key=key, arms=pair, opening_position_m=r["alignment"]["position_error_m"],
                             opening_rotation_rad=r["alignment"]["rotation_error_rad"],
                             opening_twist_gap=(np.array(r["start"]["Panda"]["twist"]) -
                                                np.array(r["start"]["Kinova3"]["twist"])).tolist(),
                             panda_reward_match=r["Panda"]["archive_native_reward_match"],
                             panda_action_difference=r["Panda"]["archive_action_max_abs"]))
    assert len(receipts) == 6
    return receipts


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
