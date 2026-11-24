"""Verify the complete locked test and package its raw-file index and per-start effects on Linux CPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


from panel import MODES, cases, key, require
from verify import summarize

FIT_SHA = "4855546591381f149d7509a6af3d45db32e9e82a732788c8376d504f19417dbc"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def readback(out, job):
    rc, start, end = map(int, (out/"exit.txt").read_text().split())
    require(rc == 0 and end > start, "incomplete native attempt")
    run, probe = read(out/"run.json"), read(out/"gpu.json")
    device, = probe["devices"]
    require(probe["status"] == "passed" and run["host"] == probe["hostname"] == f"{job}-master-0"
            and device["cuda_name"] == "NVIDIA GeForce RTX 4090", "scheduler/device identity")
    fit = Path(run["fit"])
    require(FIT_SHA == digest(fit/"fit.json") == run["fit_sha256"] == run["offline_fit"]["fit_sha256"],
            "wrong frozen trainer")
    summary = summarize(out, False)
    require(summary == read(out/"native-summary.json"), "independent complete readback differs")
    before = read(out/"frozen-before.json")
    require(before == read(out/"frozen-after.json") and before["fit_file"] == FIT_SHA and
            before["heads_file"] == digest(fit/"heads.npz"), "frozen before/after and fit files")
    effective = {c:g["action_out_proj/kernel"] for c,g in read(fit/"fit.json")["parameters"].items()}
    require(before["resident_effective_kernels"] == effective, "effective trained projection identities")
    expected = {f"episodes/{key(q)}-{m}.json" for q in cases() for m in MODES}
    require(expected == {str(p.relative_to(out)) for p in (out/"episodes").glob("*.json")}, "missing or extra tapes")
    paths = sorted(expected | {"console.log","exit.txt","launch.txt","storage.txt","gpu.json","run.json",
                               "historical-noop.json","frozen-before.json","frozen-after.json","native-summary.json"})
    hashes = {rel:digest(out/rel) for rel in paths}
    rows, attempts, steps = [], Counter(), 0
    for row in summary["rows"]:
        case = row["case"]
        entries = {}
        for mode, metrics in row["modes"].items():
            tape = read(out/"episodes"/f"{key(case)}-{mode}.json")
            r = tape["native"]
            rewards = r["native_rewards"]
            require(r["native_return"] == int(r["success"]) and
                    ((rewards[-1] == 1 and not any(rewards[:-1]) and r["terminated"]) if r["success"]
                     else r["controls"] == r["reset"]["horizon"] and not any(rewards)), "sparse endpoint")
            attempts[tape["accepted_reset_attempt"]] += 1
            steps += len(rewards)
            entries[mode] = [int(metrics["success"]), metrics["controls"]]
        diffs = {m:[v["first_different_proposal_replan"],v["first_different_issued_control"]]
                 for m,v in row["first_changes"].items()}
        rows.append(dict(case=case, outcomes=entries, first_divergences=diffs, mechanism=row["mechanism"],
                         reset=dict(physics=row["physics"],released_row=row["released_row"],noise=row["noise"]),
                         opening_parity=row["parity"]))
    return dict(job=job, pfs=str(out), pin=run["pin"], device=device["cuda_name"], uuid=device["cuda_uuid"],
                start_epoch=start, finish_epoch=end, gpu_hours=(end-start)/3600, fit=FIT_SHA,
                frozen_parameters=before, historical=read(out/"historical-noop.json"),
                registered_cases=len(rows), checked_terminals=len(expected), checked_control_rewards=steps,
                accepted_reset_attempt_counts=attempts, arms=summary["arms"], matched_arms=summary["matched_arms"],
                rows=rows, outcome_order=["native_success_and_return","executed_controls"],
                divergence_order=["first_different_proposal_replan","first_different_issued_control"],
                divergence_index_base=0, compared_to="joint", hashes=hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("job")
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    results = readback(args.out, args.job)
    with args.evidence.open("x") as f:
        json.dump(results, f, separators=(",", ":"), allow_nan=False)
        f.write("\n")
    print(json.dumps({k:results[k] for k in ("job","pin","checked_terminals","checked_control_rewards","arms")}))
