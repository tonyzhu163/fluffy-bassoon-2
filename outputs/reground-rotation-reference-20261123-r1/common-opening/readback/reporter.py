"""Poll only missing final paths. Persist each original shard once. Emit a live request feed."""
import argparse
from datetime import datetime, timezone
import gzip
import os
import sys
import time

from native_read import *


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record(s, h, keys, limit, predicates):
    proposal, stream_prefix = predicates
    by_arm = {arm:read_arm(original(s,h,arm), s, arm, keys, limit) for arm in ARMS}
    rows = []
    for i, key in enumerate(keys):
        rec = {a:by_arm[a][0][i] for a in ARMS}
        prefixes = {a:by_arm[a][1][i] for a in ARMS}
        check(all(all(rec[a]["csv"][k] == rec["base"]["csv"][k] for k in KEYS) for a in ARMS), key)
        equal = {}
        for a,b in PAIRS:
            p,q = prefixes[a],prefixes[b]
            n = min(10, rec[a]["metrics"]["steps"], rec[b]["metrics"]["steps"])
            px = proposal(p, q, n)
            state = np.array_equal(p["X"][0], q["X"][0])
            matches = np.equal(p["A_vla"][:n], q["A_vla"][:n]).all(axis=1)
            check(px == (state and matches.all()), key)
            stream = (a,b) == ("symmeta","base")
            if stream and n < 10:
                check(all(rec[a]["metrics"][k] == rec[b]["metrics"][k] for k in ("steps","success")), key)
            equal[f"{a}/{b}"] = dict(start=state, steps=n, different_indices=np.flatnonzero(~matches).tolist(),
                                     native_grid=stream_prefix(p,q,n) if stream else px)
        rows.append(dict(seed=s, shard=h, key=key, pairs=equal, arms=rec))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("work", type=Path)
    ap.add_argument("native_grid", type=Path)
    ap.add_argument("--deadline", type=float, required=True)
    args = ap.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    plans = grid()
    predicates = load_functions(args.native_grid, ("equal_proposal", "equal_stream_prefix"))
    limit = read(ROOT/"upstream/configs/speed_envelope.json")["pi05"]
    completed = args.work / "original"
    completed.mkdir(exist_ok=True)
    progressfile = args.work / "observed.json"
    progress = read(progressfile) if progressfile.exists() else {}
    def update():
        extra, known = [], []
        for tag, row in sorted(progress.items()):
            if row["discordant"]:
                item = {"seed":row["seed"],"shard":row["shard"],"id":tag,
                        "original_discordant_assignments":row["discordant"]}
                (known if (row["seed"],row["shard"]) in SEEN else extra).append(item)
        print(json.dumps(dict(observed_utc=now(), original_shards=len(progress),
                              assignments=sum(r["n"] for r in progress.values()), extra_requests=extra,
                              preassigned=["s0-004","s1-015"], confirmed_preassigned=known,
                              missing_originals=[label(*k) for k in plans if label(*k) not in progress])), flush=True)
    update()
    while True:
        for (s,h), keys in plans.items():
            tag = label(s,h)
            if tag in progress or not (original(s,h,"reground")/"manifest.json").is_file():
                continue
            rows = record(s,h,keys,limit,predicates)
            path = completed / f"{tag}.json.gz"
            tmp = path.with_suffix(".part")
            with gzip.open(tmp, "wt") as f:
                json.dump(rows, f, separators=(",",":"), allow_nan=False)
            tmp.replace(path)
            discordant = [dict(task=r["key"][0],init_id=r["key"][1],rep=r["key"][2],
                               pairs=[k for k,p in r["pairs"].items() if p["different_indices"]])
                          for r in rows if any(p["different_indices"] for p in r["pairs"].values())]
            progress[tag] = dict(seed=s, shard=h, n=len(rows), discordant=discordant)
            save(progressfile, progress)
            update()
        if len(progress) == 90 or time.time() >= args.deadline:
            break
        time.sleep(9)


if __name__ == "__main__":
    main()
