"""Keyed input contract, immutable pair records and CPU-only reduction."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile

PANELS = ("gpu09test", "gpu09iiwa", "gpu06kinova3", "gpu08pi0fix")
SEEDS = (41, 42, 43)
READERS = ("tuned", "symmeta")
OOD = {"compositional_ood", "continuous_angle_ood"}
SETTINGS = ("arm", "model_class", "joint_count", "scene_fingerprint", "perturbation_plan",
            "worker_step_index", "action_schema", "noise_seed", "noise_sha256", "horizon", "prompt")
HASH = re.compile("[0-9a-f]{64}")
HERE = Path(__file__).resolve().parent


def require(ok, msg):
    if not ok:
        raise ValueError(str(msg))


def parse(raw):
    def unique(pairs):
        obj = dict(pairs)
        require(len(obj) == len(pairs), "Duplicate JSON field")
        return obj

    def invalid(number):
        raise ValueError(f"Invalid JSON number {number}")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked(path, sha):
    raw = path.read_bytes()
    require(digest(raw) == sha, f"SHA-256 mismatch at {path}")
    return raw


def atomic(path, raw, *, replace=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".writing-", delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
            if replace:
                os.replace(tmp, path)
            else:
                try:
                    os.link(tmp, path)
                except FileExistsError:
                    require(path.read_bytes() == raw, f"Conflicting completed file {path}")
        finally:
            tmp.unlink(missing_ok=True)


def load_predictions(path, expected_sha, *, allow_fixture=False):
    obj = parse(checked(path, expected_sha))
    require(type(obj["schema"]) is int and obj["schema"] == 1 and type(obj["rows"]) is list, "Prediction schema")
    require(not obj.get("qualification_fixture") or allow_fixture, "Qualification fixture used as production input")
    by_key = {}
    for p in obj["rows"]:
        panel, i, seed = (p[k] for k in ("panel", "index", "seed"))
        key = panel, i, seed
        require(panel in PANELS and type(i) is int and 0 <= i < 96 and type(seed) is int and seed in SEEDS
                and key not in by_key and type(p["task_id"]) is int and isinstance(p["suite"], str)
                and isinstance(p["embodiment"], str) and HASH.fullmatch(p["tape_sha256"]), f"Prediction key {key}")
        for a in READERS:
            c = p[a]
            require(set(c) == {"yaw_deg", "xy_scale", "gripper_sign"} and
                    all(type(c[k]) in (int, float) and math.isfinite(c[k]) for k in ("yaw_deg", "xy_scale"))
                    and c["xy_scale"] > 0 and type(c["gripper_sign"]) is int and c["gripper_sign"] in (-1, 1),
                    f"Invalid correction {key} {a}")
        by_key[key] = p
    require(len(by_key) == 1152, f"Expected 1152 distinct prediction keys, got {len(by_key)}")
    return by_key


def join(panel, cases, predictions):
    rows = []
    require(len(cases) == 96 and len({(c["suite"], c["task_id"], c["embodiment"]) for c in cases}) == 96,
            f"Invalid archive panel {panel}")
    for i, c in enumerate(cases):
        require(c["index"] == i and c["panel"] == panel, "Archive index")
        for seed in SEEDS:
            p = predictions[panel, i, seed]
            for k in ("panel", "index", "suite", "task_id", "embodiment", "tape_sha256"):
                require(c[k] == p[k], f"Prediction/archive {k} mismatch at {(panel, i, seed)}")
            require(p.get("regime", c["regime"]) == c["regime"], f"Wrong regime {(panel, i, seed)}")
            rows.append(dict(c, seed=seed, **{a: p[a] for a in READERS}))
    return rows


def pair_path(out, row):
    return out / "pairs" / row["panel"] / f"s{row['seed']}" / f"{row['index']:02d}.json"


def seal(out, predictions_sha):
    panels = {p: digest((out/p/"panel.json").read_bytes()) for p in PANELS}
    panels_data = [parse(checked(out/p/"panel.json", v)) for p, v in panels.items()]
    ids = None
    for p, data in zip(PANELS, panels_data, strict=True):
        rows = data["rows"]
        keys = [(r["index"], r["seed"], r["suite"], r["task_id"], r["embodiment"], r["regime"]) for r in rows]
        require(data["predictions_sha256"] == predictions_sha and data["panel"] == p and
                [(k[0], k[1]) for k in keys] == [(i, s) for i in range(96) for s in SEEDS]
                and len([r for r in rows if r["regime"] in OOD]) == 192 and (ids is None or ids == keys),
                f"Cross-panel alignment at {p}")
        ids = keys
    code = {f: digest((HERE/f).read_bytes()) for f in
            ("live.py", "paired_records.py", "paired_inputs.py", "paired_devices.py", "live.sh")}
    for path in (HERE.parent/"reground-rotation-reference-20261123-r1/reference.py",
                 HERE.parents[1]/"scripts/[train]gpu_probe.py", HERE.parents[1]/"src/fluffy_bassoon/baige_gpu.py"):
        code[str(path.relative_to(HERE.parents[1]))] = digest(path.read_bytes())
    atomic(out/"contract.json", encode(dict(schema=1, predictions_sha256=predictions_sha,
                                            panels=panels, evaluator_sha256=code)))


def contract(out):
    raw = (out/"contract.json").read_bytes()
    d = parse(raw)
    require(d["schema"] == 1 and set(d["panels"]) == set(PANELS), "Invalid attempt contract")
    return digest(raw), d


def anchor(arms):
    a = arms["tuned"]
    return a["reset"], a["opening_proposal_sha256"]


def validate_arms(r, arms, ref):
    key = r["panel"], r["index"], r["seed"]
    require(set(arms) == set(READERS), f"Incomplete readers {key}")
    reset, opening = anchor(arms)
    require(reset == arms["symmeta"]["reset"] and all(reset[k] == ref[k] for k in SETTINGS)
            and HASH.fullmatch(reset["physics"]) and reset["physics"] != r["support_physics"]
            and reset["renderer_vendor"] == "NVIDIA Corporation" and reset["renderer"]
            and reset["noise_seed"] == r["query"]["seed"] + 817117 and
            HASH.fullmatch(opening) and opening == arms["symmeta"]["opening_proposal_sha256"],
            f"Query pairing mismatch {key}")
    for k in ("public", "motor_frame0", "reader_frame0"):
        require(reset[k].keys() == ref[k].keys() and all(HASH.fullmatch(v) for v in reset[k].values()),
                f"Incomplete {k} receipt {key}")
    for a in READERS:
        x = arms[a]
        n = x["controls"]
        require(x["inferred"] == r[a] and x["released_prompt"] == reset["prompt"]
                and type(n) is int and 0 < n <= reset["horizon"] and len(x["native_rewards"]) == n
                and all(type(v) in (int, float) and math.isfinite(v) for v in x["native_rewards"])
                and math.isfinite(x["native_return"]) and sum(x["native_rewards"]) == x["native_return"]
                and all(type(x[k]) is bool for k in ("success", "terminated", "truncated"))
                and x["terminated"] != x["truncated"] and (not x["truncated"] or n == reset["horizon"])
                and (not x["success"] or x["terminated"]), f"Invalid native trajectory {key} {a}")


def receipt(out, sha):
    require(HASH.fullmatch(sha), "Invalid device receipt digest")
    return parse(checked(out/"receipts"/(sha+".json"), sha))


def read_pair(out, row, ref, contract_sha, receipts):
    path = pair_path(out, row)
    raw = path.read_bytes()
    obj = parse(raw)
    body = obj["pair"]
    require(digest(encode(body)) == obj["sha256"] and body["row"] == row and
            body["contract_sha256"] == contract_sha, f"Pair provenance mismatch at {path}")
    sha = body["execution_sha256"]
    if sha not in receipts:
        receipts[sha] = receipt(out, sha)
    device = receipts[sha]["devices"][body["lane"]]
    require(device["allocation_ordinal"] == body["lane"] and receipts[sha]["contract_sha256"] == contract_sha
            and receipts[sha]["hostname"].startswith("job-"), f"Wrong execution origin {path}")
    validate_arms(row, body["arms"], ref)
    return raw, body


def write_pair(out, row, arms, ref, contract_sha, execution_sha, lane):
    validate_arms(row, arms, ref)
    body = dict(row=row, arms=arms, contract_sha256=contract_sha, execution_sha256=execution_sha, lane=lane)
    atomic(pair_path(out, row), encode(dict(pair=body, sha256=digest(encode(body)))))


def clustered_interval(task_totals, observations):
    # Equal-sized tasks. Convolve integer paired differences for eight whole-task draws.
    distribution = Counter({0: 1})
    for _ in task_totals:
        updated = Counter()
        for value, count in distribution.items():
            for delta in task_totals:
                updated[value + delta] += count
        distribution = updated
    cuts = []
    total = sum(distribution.values())
    cum = 0
    for v, multiplicity in sorted(distribution.items()):
        cum += multiplicity
        while len(cuts) < 2 and cum * 40 >= (1, 39)[len(cuts)] * total:
            cuts.append(100 * v / observations)
    return cuts


def counts(group):
    result = {a: sum(b["arms"][a]["success"] for b in group) for a in READERS}
    diffs = [int(b["arms"]["tuned"]["success"]) - int(b["arms"]["symmeta"]["success"]) for b in group]
    pct = {a: 100*v/len(group) if group else None for a, v in result.items()}
    return dict(n=len(group), successes=result, success_pct=pct, wins=diffs.count(1), losses=diffs.count(-1),
                ties=diffs.count(0), tuned_minus_symmeta_pp=100*sum(diffs)/len(diffs) if group else None)


def describe(group, expected_per_seed):
    result = counts(group)
    by_seed = {s: counts([b for b in group if b["row"]["seed"] == s]) for s in SEEDS}
    result.update(per_seed=by_seed, complete=False, seed_sd_pp=None, task_cluster_ci95_pp=None)
    if all(x["n"] == expected_per_seed for x in by_seed.values()):
        tasks = defaultdict(list)
        for b in group:
            tasks[b["row"]["suite"], b["row"]["task_id"]].append(b)
        require(len(tasks) == 8 and all(len(v)*8 == len(group) for v in tasks.values()), "Unbalanced tasks")
        result.update(complete=True, seed_sd_pp={
            **{a: statistics.stdev(100*x["successes"][a]/x["n"] for x in by_seed.values()) for a in READERS},
            "tuned_minus_symmeta": statistics.stdev(x["tuned_minus_symmeta_pp"] for x in by_seed.values())},
            task_cluster_ci95_pp=clustered_interval(
                [counts(v)["wins"]-counts(v)["losses"] for v in tasks.values()], len(group)))
    return result


def reduce_attempt(out, partial=False):
    sha, c = contract(out)
    panels, receipts, expected = {}, {}, set()
    for p in PANELS:
        d = parse(checked(out/p/"panel.json", c["panels"][p]))
        records, missing = [], []
        task_diagnostics = defaultdict(lambda: {"reset": set(), "opening": set(), "execution": set()})
        for r in d["rows"]:
            path = pair_path(out, r)
            expected.add(path)
            if not path.is_file():
                missing.append([r["index"], r["seed"]])
                continue
            _, b = read_pair(out, r, d["refs"][r["shard"]], sha, receipts)
            reset, opening = anchor(b["arms"])
            diag = task_diagnostics[f"{r['suite']}/{r['task_id']}"]
            diag["reset"].add(digest(encode(reset)))
            diag["opening"].add(opening)
            diag["execution"].add(b["execution_sha256"])
            records.append(b)
        panels[p] = dict(missing_index_seed=missing,
                         task_distinct_fingerprints={k: {field: len(v) for field, v in ds.items()}
                                                     for k, ds in task_diagnostics.items()},
                         overall=describe(records, 96),
                         ood=describe([b for b in records if b["row"]["regime"] in OOD], 64))
    extra = set((out/"pairs").rglob("*.json")) - expected
    require(not extra, f"Unexpected pair paths {extra}")
    n = sum(d["overall"]["n"] for d in panels.values())
    summary = dict(contract_sha256=sha, expected_pairs=1152, completed_pairs=n, complete=n == 1152, panels=panels)
    require(partial or n == 1152, f"Incomplete reduction {n}/1152")
    if not partial:
        atomic(out/"summary.json", encode(summary))
    atomic(out/"partial.json", encode(summary), replace=True)
    print(f"Validated {n}/1152 paired records", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("seal", "reduce"))
    p.add_argument("out", type=Path)
    p.add_argument("--predictions-sha")
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    if args.mode == "seal":
        seal(args.out, args.predictions_sha)
    else:
        reduce_attempt(args.out, args.partial)
