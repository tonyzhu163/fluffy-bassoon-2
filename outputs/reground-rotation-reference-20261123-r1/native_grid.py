"""Audit original as-run outcomes on the released grid and measure opening agreement."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from compare import summarize_assignments


RUN = "outputs/reground-rotation-reference-20261123-r1"
STREAM_PIN = "cab855a03c8ff1a3f1926513369099bf11c4b859"
KEYS = ("policy", "task", "suite", "shift", "level", "adapt_seed", "split", "init_id", "rep", "noise_seed")


def equal_proposal(left, right, steps):
    return (np.array_equal(left["X"][0], right["X"][0])
            and np.array_equal(left["A_vla"][:steps], right["A_vla"][:steps]))


def equal_stream_prefix(stream, base, steps):
    return all(np.array_equal(stream[k][:n], base[k][:n])
               for k, n in (("X", steps + 1), ("A", steps), ("A_vla", steps), ("speed", steps)))


def read_grid():
    root = Path(os.environ["REGROUND_RUNTIME_ROOT"])
    code = root / f"src-{STREAM_PIN}"
    if subprocess.check_output(["git", "-C", code, "rev-parse", "HEAD"], text=True).strip() != STREAM_PIN:
        raise ValueError("Streaming producer pin differs")
    sys.path[:0] = [str(code / RUN), str(code / "src")]
    import symmeta_adapter as adapter
    ref = adapter.reference
    ref.configure()
    from reground.eval.jobs import load_traces

    rows = []
    for job in adapter.units(ref.SEEDS):
        s, shard = job["seed"], job["shard"]
        paths = {"symmeta": adapter.STORE / "shards" / ref.label(job)}
        adapter.validate(paths["symmeta"], job)
        for method in ("base", "reground"):
            native = next(j for j in ref.units(method, [s]) if j["shard"] == shard)
            paths[method] = ref.output(native)
            ref.check_eval(paths[method], native)
        tables = {}
        traces = {}
        for method, path in paths.items():
            with (path / "results.csv").open() as f:
                tables[method] = list(csv.DictReader(f))
            traces[method] = load_traces(path)
        for st, ba, rg in zip(tables["symmeta"], tables["base"], tables["reground"], strict=True):
            if not all(st[k] == ba[k] == rg[k] for k in KEYS):
                raise ValueError(f"Native assignment differs in seed {s}, shard {shard}")
            key = (st["task"], int(st["init_id"]), int(st["rep"]))
            a, b, g = (traces[m][key] for m in ("symmeta", "base", "reground"))
            prefix = min(10, int(ba["steps"]), int(st["steps"]))
            if prefix < 10 and (ba["steps"], ba["success"]) != (st["steps"], st["success"]):
                raise ValueError(f"Different stopping in opening {s} {key}")
            row = {k: st[k] for k in KEYS}
            row.update(shard=shard, opening_length=prefix,
                       symmeta_base_opening_exact=equal_stream_prefix(a, b, prefix),
                       reground_base_opening_exact=equal_proposal(g, b, min(10, int(ba["steps"]), int(rg["steps"]))),
                       symmeta_reground_opening_exact=equal_proposal(a, g, min(10, int(st["steps"]), int(rg["steps"]))),
                       all_initial_robot_states_exact=np.array_equal(a["X"][0], b["X"][0])
                       and np.array_equal(a["X"][0], g["X"][0]))
            for name, data in (("symmeta", st), ("base", ba), ("reground", rg)):
                for metric in ("success", "unsafe", "steps"):
                    row[f"{name}_{metric}"] = int(data[metric])
            rows.append(row)
    return rows


def opening_sensitivity(cases):
    result = {}
    for a, b in (("symmeta", "base"), ("reground", "base"), ("symmeta", "reground")):
        mask = cases[f"{a}_{b}_opening_exact"]
        unmatched = int((~mask).sum())
        item = dict(exact=int(mask.sum()), unmatched=unmatched,
                    affected_shards=cases.loc[~mask, ["adapt_seed", "shard"]].drop_duplicates().values.tolist())
        for metric in ("success", "unsafe"):
            x, y = cases.loc[mask, f"{a}_{metric}"], cases.loc[mask, f"{b}_{metric}"]
            diff = int(x.sum() - y.sum())
            item[metric] = dict(a_events_on_exact=int(x.sum()), b_events_on_exact=int(y.sum()),
                                mean_difference_on_exact=float((x - y).mean()),
                                all_grid_mean_difference_bounds=[(diff - unmatched) / len(cases),
                                                                 (diff + unmatched) / len(cases)])
        result[f"{a}_minus_{b}"] = item
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows = read_grid()
    cases = pd.DataFrame(rows)
    summary = summarize_assignments(cases)
    summary["all_initial_robot_states_exact"] = int(cases.all_initial_robot_states_exact.sum())
    summary["opening_sensitivity"] = opening_sensitivity(cases)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with gzip.open(args.output_dir / "native-cases.json.gz", "wt") as f:
        json.dump(rows, f, allow_nan=False)
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
