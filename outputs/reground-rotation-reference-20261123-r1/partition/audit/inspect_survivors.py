"""Read only production collect/babble files for cross-host keys. Run on Linux CPU."""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from zipfile import ZipFile

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def stat_tuple(path):
    try:
        s = path.stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    except FileNotFoundError:
        return None


def read_file(path):
    blob = path.read_bytes()
    return blob, dict(path=str(path), size=len(blob), sha256=digest(blob))


def arr_id(arr):
    return dict(dtype=arr.dtype.str, shape=list(arr.shape), sha256=digest(arr.tobytes(order="C")))


def array_equal_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def literal_assignment(path, field):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == field for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(field)


class CollectPayload(pickle.Unpickler):
    # The pinned production collect checkpoints store NumPy traces in a SimpleNamespace.
    # No torch tensors are needed to read this specific payload.
    allowed = {("types", "SimpleNamespace"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"), ("_codecs", "encode")}

    def find_class(self, module, name):
        if (module, name) not in self.allowed:
            raise ValueError((module, name))
        return super().find_class(module, name)


def read_checkpoint(path):
    blob, fp = read_file(path)
    with ZipFile(io.BytesIO(blob)) as z:
        bad = z.testzip()
        entry, = [name for name in z.namelist() if name.endswith("/data.pkl")]
        value = CollectPayload(io.BytesIO(z.read(entry))).load()
    pid = re.fullmatch(r"batch\d+\.(\d+)/data.pkl", entry)
    return value, dict(**fp, zip_crc_ok=bad is None, data_member=entry,
                       archive_pid=int(pid[1]) if pid else None)


def owner_lane(meta):
    m = re.search(r" --attempt \S+/([^/ ]+) .*?--rank (\d+)(?: |$)", meta.get("argv", ""))
    return (m[1], "collect-gpu" + m[2]) if m else (None, None)


def batch_evidence(key, seed, shard, owners, arrays):
    paths = sorted((store / "batches" / key).glob("batch*.pt"))
    batch_rows = []
    traces = []
    rng = np.random.default_rng([seed, shard, 4242])
    for index, path in enumerate(paths):
        current, row = read_checkpoint(path)
        episodes = current["episodes"]
        trace_batch = current["result"].traces
        expected = [(tasks[(shard + 4*index + i) % len(tasks)], None, 0, int(rng.integers(2**31)))
                    for i in range(4)]
        row.update(batch=index, episodes=episodes, lengths=[len(t["A"]) for t in trace_batch],
                   order_ok=path.name == f"batch{index:04d}.pt",
                   identities_ok=current["upstream"] == pin and current["policy"] == policy
                   and current["rng"] is None and episodes == expected,
                   trace_contract_ok=len(trace_batch) == 4 and all(
                       t["A"].ndim == 2 and t["A"].shape[1] == 7 and len(t["A"]) > 0
                       and t["X"].shape == (len(t["A"])+1,25) and np.isfinite(t["A"]).all()
                       and np.isfinite(t["X"]).all() and t["task"] == e[0]
                       for t, e in zip(trace_batch, expected)))
        row["pid_lane_candidates"] = sorted(pids[row["archive_pid"]])
        traces.extend(trace_batch)
        batch_rows.append(row)
    counts = {}
    cp_logs = {}
    for owner in owners:
        attempt, lane = owner
        path = str(store / "attempts" / attempt / (lane + ".log"))
        stream = rec["logs"].get(path, {}).get("lines", [])
        rows = []
        latest = None
        for line_no, msg in stream:
            event = cp_msg.match(msg)
            if event:
                mode, unit, idx = event.groups()
                latest = dict(batch=int(idx), event=mode, checkpoint_line=line_no) if unit == key else None
                if latest:
                    rows.append(latest)
            elif msg.startswith("[collect]") and latest:
                progress = collect_msg.match(msg)
                if progress:
                    cell, s, sh, total = progress.groups()
                    if f"seed{s}/{cell}/collect/shard{int(sh):03d}" == key:
                        latest.update(cumulative=int(total), count_line=line_no)
                latest = None
        cp_logs[attempt+"/"+lane] = rows
        counts[attempt+"/"+lane] = [r.get("cumulative") for r in rows]
    rebuilt = {}
    if traces and arrays is not None:
        rebuilt = dict(X=np.concatenate([t["X"] for t in traces]), A=np.concatenate([t["A"] for t in traces]),
                       lens=np.array([len(t["A"]) for t in traces], dtype=np.int64),
                       task=np.array([t["task"] for t in traces]))
    re_npz = io.BytesIO()
    if rebuilt:
        np.savez_compressed(re_npz, **rebuilt)
    return dict(checkpoints=batch_rows, lane_log_batches=cp_logs, lane_cumulative_counts=counts,
                rebuilt_equals_final={k: array_equal_bytes(v, arrays[k]) for k, v in rebuilt.items()},
                rebuilt_npz_sha256=digest(re_npz.getvalue()) if rebuilt else None)


def check_key(key, group):
    seed_part, shift, level, phase, shard_part = key.split("/")
    seed, cell, shard = int(seed_part[4:]), shift+"/"+level, int(shard_part[5:])
    babble = phase == "babble"
    leaf = "babble" if babble else "vla_pi05"
    dest = store / "runs/data" / cell / seed_part / leaf / (shard_part + ".npz")
    meta_path = dest.with_suffix(".json")
    owners = sorted({(r["attempt"], r["lane"]) for r in group["rows"]})
    entries = [dest, meta_path, *sorted((store / "batches" / key).glob("batch*.pt"))]
    before = {str(p): stat_tuple(p) for p in entries}
    info = dict(read_utc=now(), comparable=group["comparable"], publication_counts=group["counts"],
                partial_files={f"{a}/{l}":[str(p) for p in sorted((store / "attempts" / a /
                                "partial" / key).rglob("*")) if p.is_file()] for a,l in owners},
                final_exists=dest.is_file(), sidecar_exists=meta_path.is_file())
    arr = None
    if info["final_exists"] and info["sidecar_exists"]:
        blob, fp = read_file(dest)
        mblob, mfp = read_file(meta_path)
        meta = json.loads(mblob)
        with np.load(io.BytesIO(blob), allow_pickle=False) as d:
            arr = {k:d[k] for k in d.files}
        with ZipFile(io.BytesIO(blob)) as z:
            crc_ok = z.testzip() is None
        lens, task = arr["lens"], arr["task"]
        n, e = int(lens.sum()), len(lens)
        target, quantum = (2000, 1) if babble else (4000, 4)
        expected = [tasks[shard % len(tasks)] if babble else tasks[(shard+i)%len(tasks)] for i in range(e)]
        checks = dict(zip_crc=crc_ok, schema=set(arr) == {"X","A","lens","task"}, positive_int_lengths=
                      lens.ndim == 1 and lens.dtype.kind in "iu" and bool((lens>0).all()),
                      native_dtypes=arr["X"].dtype == arr["A"].dtype == np.float32 and lens.dtype == np.int64,
                      shape=arr["X"].shape == (n+e,25) and arr["A"].shape == (n,7) and task.shape == (e,),
                      finite=bool(np.isfinite(arr["X"]).all() and np.isfinite(arr["A"]).all()),
                      task_order=task.tolist() == expected, stopping=n>=target and e%quantum==0 and
                      int(lens[:-quantum].sum()) < target, sidecar_counts=meta["n"] == n,
                      sidecar_identity=meta["seed"] == seed and meta["cell"] == cell)
        if babble:
            checks.update(shard=meta["shard"]==shard, task_list=meta["tasks"]==tasks,
                          native_cfg=meta["cfg"]==babble_cfg, wall_s_valid=meta["wall_s"]>=0)
        else:
            checks.update(sidecar_model=meta["git_sha"]==pin and meta["vla_revision"]==policy and
                          meta["policy"]=="pi05" and meta["x0_fixed"] is True)
            owner = owner_lane(meta)
            info["sidecar_owner"] = list(owner)
            checks["sidecar_owner_in_receipts"] = (owner in owners and
                rec["attempts"][owner[0]]["reservation"]["host"]==meta["host"])
        info.update(npz=fp, sidecar=mfp, metadata=meta, episodes=e, transitions=n, checks=checks,
                    all_reported_counts_match=all(pair==[n,e] for pair in group["counts"]),
                    arrays={k:arr_id(v) for k,v in arr.items()}, lengths=lens.tolist(), tasks=task.tolist())
    if not babble:
        info["batch_evidence"] = batch_evidence(key, seed, shard, owners, arr)
    entries_after = [dest, meta_path, *sorted((store / "batches" / key).glob("batch*.pt"))]
    info["stat_before"] = before
    info["stat_stable"] = before == {str(p): stat_tuple(p) for p in entries_after}
    return info


def local_snapshot_comparison():
    snap = inputs["local_snapshot"]
    compared = defaultdict(int)
    errors = []
    for name, attempt in snap["attempts"].items():
        if name not in rec["attempts"] or rec["attempts"][name]["reservation"]["phase"] not in ("collect","babble"):
            continue
        reservation = rec["attempts"][name]["reservation"]
        if attempt["host"] != reservation["host"] or attempt["pin"] != reservation["code"]:
            errors.append(dict(attempt=name, reason="reservation"))
        for lane, stream in attempt["lanes"].items():
            source = str(store / "attempts" / name / (lane+".jsonl"))
            actual = rec["streams"][source]["rows"]
            for item in stream["published"]:
                if any(r["event"]=="published" and all(r[k]==v for k,v in item.items()) for r in actual):
                    compared[name] += 1
                else:
                    errors.append(dict(source=source, item=item))
    return dict(read_utc=snap["read_utc"], source="speed/active-snapshot.json",
                matched_publications=dict(compared), differences=errors)


inputs = json.load(sys.stdin.buffer)
rec = inputs["receipts"]
assert platform.system() == "Linux" and os.environ["CUDA_VISIBLE_DEVICES"] == ""
assert str(Path.cwd()) == rec["root"]
store = Path("reference-native")
up = Path("upstream")
ref = Path("src-23a6641562a4d833837f27ab0535a20ad94e8ec8")/"outputs"/Path.cwd().name/"reference.py"
speed = Path("speed/src-516a2f75bd9a213609d60858e2a33c3c7743cf84")/"outputs"/Path.cwd().name/"speed/babble.py"
tasks = literal_assignment(up/"util.py","TASKS")
pin, policy = (literal_assignment(ref,field) for field in ("PIN","POLICY"))
babble_src = up/"reground/data/babble.py"
config_class, = [n for n in ast.parse(babble_src.read_text()).body if isinstance(n, ast.ClassDef) and n.name=="BabbleCfg"]
babble_cfg = {n.target.id:ast.literal_eval(n.value) for n in config_class.body if isinstance(n,ast.AnnAssign)}
cp_msg = re.compile(r"^\[(batch-checkpoint|replay)\] .*/reference-native/batches/(.+)/batch(\d+)\.pt\b")
collect_msg = re.compile(r"^\[collect\] collect pi05 (\S+) seed(\d+) shard(\d+) (\d+)/4000 ")
pids = defaultdict(set)
pid_examples = {}
for path in sorted((store/"runs/data").glob("*/*/*/vla_pi05/shard*.json")):
    meta = json.loads(path.read_bytes())
    attempt, lane = owner_lane(meta)
    if attempt and rec["attempts"].get(attempt,{}).get("reservation",{}).get("host")==meta["host"]:
        key = attempt+"/"+lane
        pids[meta["pid"]].add(key)
        pid_examples.setdefault((key,meta["pid"]),str(path))
out = dict(started_utc=now(), receipt_cutoff_utc=rec["receipt_cutoff_utc"], numpy=np.__version__,
           python=platform.python_version(), local_comparison=local_snapshot_comparison(),
           source_sha256={str(p):digest(p.read_bytes()) for p in (ref,speed,up/"util.py",babble_src,
                         up/"reground/eval/jobs.py",up/"reground/utils.py")},
           lane_pid_evidence=[dict(lane=k, pid=p, metadata=v) for (k,p),v in sorted(pid_examples.items())],
           keys={})
for unit, item in rec["cross_host_keys"].items():
    out["keys"][unit] = check_key(unit,item)
# Look for residual temporaries across production data, batches and the relevant staging trees only.
temp_roots = [store/"runs/data",store/"batches"]
temp_roots += [store/"attempts"/name/"partial" for name, v in rec["attempts"].items()
               if (v.get("reservation") or {}).get("phase") in ("collect","babble","all")]
out["temp_scan_roots"] = [str(p) for p in temp_roots]
out["temp_files"] = [str(Path(base)/d) for top in temp_roots for base, _, files in os.walk(top)
                     for d in files if ".tmp" in d]
out["finished_utc"] = now()
print(json.dumps(out, indent=1, allow_nan=False))
