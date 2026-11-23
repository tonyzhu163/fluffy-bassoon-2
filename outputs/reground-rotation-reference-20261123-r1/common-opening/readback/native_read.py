"""Read released data with NumPy and the standard library, without evaluator imports."""
import ast
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs/reground-rotation-reference-20261123-r1")
ARMS = ("base", "symmeta", "reground")
PAIRS = (("symmeta", "base"), ("reground", "base"), ("symmeta", "reground"))
PIN = "16a98486f560676b9036d1a6d067b9dd51a5917a"
POLICY = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
STREAM = "cab855a03c8ff1a3f1926513369099bf11c4b859"
FIELDS = ("X", "A", "A_vla", "ref", "speed")
LIMITS = {"10": 520, "goal": 300, "spatial": 280, "object": 280}
SEEN = {(0, 4), (1, 15)}
KEYS = ("policy", "task", "suite", "shift", "level", "adapt_seed", "split", "init_id", "rep", "noise_seed")


def check(ok, message):
    if not ok:
        raise ValueError(str(message))


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def seed_for(i, r):
    return int(hashlib.sha1(f"{i}:{r}".encode()).hexdigest()[:8], 16)


def load_functions(path, names):
    # Execute only the registered pure predicates, without any evaluator/policy imports.
    tree = ast.parse(path.read_bytes())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    check({n.name for n in tree.body} == set(names), names)
    scope = {"np": np}
    exec(compile(tree, str(path), "exec"), scope)
    return [scope[n] for n in names]


def grid():
    schedule = read(ROOT / "rotation-jobs.json")
    with gzip.open(ROOT / f"src-{STREAM}" / "outputs" / ROOT.name / "evidence/rotation-jobs.json.gz") as f:
        check(json.load(f) == schedule, "Pinned and shared schedules differ")
    plans = {}
    for j in schedule:
        episodes = sorted(tuple(k) for k in j["episodes"])
        check(len(episodes) == len(set(episodes)) == 280, "schedule")
        for i in range(0, len(episodes), 16):
            plans[j["seed"], i // 16] = episodes[i:i+16]
    check(set(plans) == {(s, h) for s in range(5) for h in range(18)}, "shard grid")
    return plans


def original(s, h, arm):
    store = "symmeta-native" if arm == "symmeta" else "reference-native"
    return ROOT / store / "shards" / f"seed{s}/rot_z/90/{arm}/shard{h:03d}"


def row_key(row):
    return row["task"], int(row["init_id"]), int(row["rep"])


def read_arm(path, s, arm, keys, speed_limit):
    man = read(path / "manifest.json")
    expected = dict(n_done=len(keys), n_rows=len(keys), n_grid=280, git_sha=PIN, vla_revision=POLICY,
                    vla_compiled=False, adapt_seed=s, cell="rot_z/90", method=arm, policy="pi05", split="test")
    check(all(man[k] == v for k, v in expected.items()), path)
    with (path / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    check([row_key(r) for r in rows] == keys, f"assignment order {path}")
    files = sorted((path / "traces").glob("*.npz"))
    check(len(files)*4 == len(keys), f"trace batch count {path}")
    records, prefixes = [], []
    for batch, fp in enumerate(files):
        with np.load(fp, allow_pickle=False) as z:
            names = ["__".join(map(str,k)) for k in keys[4*batch:4*batch+4]]
            check(set(z.files) == {f"{n}__{k}" for n in names for k in FIELDS}, fp)
            for lane, n in enumerate(names):
                row = rows[4*batch+lane]
                task, i, r = row_key(row)
                suite = task.split("_")[0]
                check(row["suite"] == suite and row["method"] == arm and row["policy"] == "pi05"
                      and row["shift"] == "rot_z" and float(row["level"]) == 90 and row["split"] == "test"
                      and int(row["adapt_seed"]) == s and int(row["noise_seed"]) == seed_for(i, r), row)
                tr = {k:z[f"{n}__{k}"] for k in FIELDS}
                t = int(row["steps"])
                check(0 < t <= LIMITS[suite] and row["success"] in ("0", "1")
                      and (row["success"] == "1" or t == LIMITS[suite]), row)
                for k, shape in zip(FIELDS, ((t+1,25), (t,7), (t,7), (t,3), (t,))):
                    a = tr[k]
                    values_ok = np.isnan(a).all() if k == "ref" and arm != "reground" else np.isfinite(a).all()
                    check(a.shape == shape and a.dtype == np.float32 and values_ok, (fp, n, k))
                counts = {k:int(row[k]) for k in ("contact_steps","displaced","fallen","speed_viol","table_steps")}
                check(all(v >= 0 for v in counts.values()), row)
                sv = int((tr["speed"] > speed_limit).sum())
                unsafe = int(any(counts[k] > 0 for k in ("contact_steps","displaced","fallen")) or sv > 0)
                check(sv == counts["speed_viol"] and int(row["unsafe"]) == unsafe, row)
                prefix = {k:tr[k][:11 if k == "X" else 10].copy() for k in ("X","A","A_vla","speed")}
                records.append(dict(metrics={k:int(row[k]) for k in ("success","unsafe","steps")}, safety=counts,
                                    trace_sha256={k:digest(a) for k,a in tr.items()}, x0=digest(tr["X"][0]),
                                    prefix_sha256={k:digest(a) for k,a in prefix.items()},
                                    csv={k:v for k,v in row.items() if k not in ("ms_vla","ms_per_step","wall_s")}))
                prefixes.append(prefix)
    return records, prefixes


def label(s,h):
    return f"s{s}-{h:03d}"
