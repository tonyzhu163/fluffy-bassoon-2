"""Select only physically measured native branches using the fixed reader and its controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / "libero-axis-factor-20261119-r1"))
from factor import NAMES, SIGNS, predict_support


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    native = json.loads(args.summary.read_text())
    assert native["status"] == "complete" and (native["probe_episodes"], native["query_episodes"]) == (64, 64)
    assert set(native["pairs"]) == {p["trial"] for p in native["schedule"]} and len(native["pairs"]) == 16
    lookup = {tuple(signs): name for signs, name in zip(SIGNS.astype(int), NAMES)}
    methods = ("reader", "analytic", "identity", "cleared", "opposite", "true_convention")
    scores = {m: {name: dict(success=0, signs_correct=0, trials=0) for name in NAMES} for m in methods}
    repro = []
    detail = []
    for key, pair in native["pairs"].items():
        probes, branches = pair["probes"], pair["branches"]
        assert set(probes) == set(branches) == set(NAMES)
        assert len({x["reset"]["physics"] for x in probes.values()}) == 1
        assert len({x["reset"]["physics"] for x in branches.values()}) == 1
        assert probes["nom"]["reset"]["physics"] != branches["nom"]["reset"]["physics"]
        repro.append(branches["nom"]["reproduction"])
        tokens = []
        for name in NAMES:
            sample = probes[name]["samples"]
            assert len(sample) == 10
            expected = [sample[i]["wire_xy"] + (np.array(sample[i]["position"][:2])
                        - np.array(sample[i - 1]["position"][:2])).tolist() for i in (4, 9)]
            assert np.array_equal(probes[name]["tokens"], expected)
            tokens.append(expected)
            b = branches[name]
            assert b["controls"] <= b["budget_controls"] and len(b["rewards"]) == b["controls"]
            assert b["success"] == bool(b["rewards"][-1])
        tokens = np.array(tokens, dtype=np.float32)
        estimated = predict_support(args.checkpoint, tokens)
        cleared = predict_support(args.checkpoint, np.zeros_like(tokens))
        analytic = np.where((tokens[:, :, :2] * tokens[:, :, 2:]).sum(axis=1) < 0, -1, 1)
        for i, name in enumerate(NAMES):
            truth = SIGNS[i].astype(int)
            opposite = NAMES.index(lookup[tuple(-truth)])
            estimates = dict(reader=estimated[i], analytic=analytic[i], identity=(1, 1),
                             cleared=cleared[i], opposite=estimated[opposite], true_convention=truth)
            for method, estimate in estimates.items():
                net = lookup[tuple(truth * estimate)]
                assert net == native["branch_by_true_and_estimate"][name][lookup[tuple(estimate)]]
                success = branches[net]["success"]
                v = scores[method][name]
                v["success"] += int(success)
                v["signs_correct"] += int(np.array_equal(estimate, truth))
                v["trials"] += 1
                detail.append(dict(trial=key, archive_state_exact=pair["archive_reset"]["physics_match"],
                                   convention=name, method=method, estimated=np.asarray(estimate).tolist(),
                                   net=net, success=success))
    print(json.dumps(dict(scores=scores, detail=detail, source_archive_pin=native["source_revision"],
                          launch=(args.summary.parent.parent / "launch.txt").read_text().strip(),
                          exit=(args.summary.parent.parent / "exit.txt").read_text().strip(),
                          probe_episodes=native["probe_episodes"], query_episodes=native["query_episodes"],
                          source_sha256=file_hash(args.summary), checkpoint_sha256=file_hash(args.checkpoint),
                          archive_resets={k: pair["archive_reset"] for k, pair in native["pairs"].items()},
                          nominal_replay=dict(success_matches=sum(r["success_matches"] for r in repro),
                                              controls_match=sum(r["controls_match"] for r in repro),
                                              reward_mismatches=sum(r["reward_mismatches"] for r in repro),
                                              max_wrist_drift_m=max(r["max_position_drift_m"] for r in repro)),
                          elapsed_s=native["updated_unix_s"] - native["started_unix_s"]), indent=1))


if __name__ == "__main__":
    main()
