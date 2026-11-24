"""Re-score the portable step records without simulator, policy or numeric dependencies."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import tarfile


def read(ar, member):
    return json.load(ar.extractfile(member))


def action_bytes(rows):
    return b"".join(struct.pack("<7f", *row) for row in rows)


def paired_interval(d):
    """Exact percentile endpoints for the empirical paired-layout bootstrap."""
    distribution = {0: 1}
    for _ in d:
        nxt = Counter()
        for subtotal, multiplicity in distribution.items():
            for value in d:
                nxt[subtotal+value] += multiplicity
        distribution = nxt
    cumulative = 0
    bounds = []
    for total, count in sorted(distribution.items()):
        cumulative += count
        while len(bounds) < 2 and 40*cumulative >= (1,39)[len(bounds)]*len(d)**len(d):
            bounds.append(100*total/len(d))
    return bounds


def score(path, donor_archive):
    modes = ("raw","tuned","IIWA34","IIWA35")
    prefixes = {}
    with tarfile.open(donor_archive) as ar:
        for m in modes[2:]:
            obj = read(ar, f"gpu01/episodes/IIWA-{m[-2:]}-raw.json")
            commands = [a for c in obj["calls"] for a in c["sent"]]
            switch = next(i for i,a in enumerate(commands) if a[6] > 0)
            prefixes[m] = commands[:5*(switch//5)]
    rows, hashes = [], {}
    with tarfile.open(path) as ar:
        for row in range(40,48):
            group = {}
            for m in modes:
                name = f"episodes/{row}-{m}.json"
                data = ar.extractfile(name).read()
                hashes[name] = hashlib.sha256(data).hexdigest()
                obj = json.loads(data)
                assert not obj["fixture"] and obj["mode"] == m and obj["query"]["init_state_id"] == row
                assert obj["query"]["seed"] == 42710300+row and obj["query"]["arm"] == "Kinova3"
                calls = obj["calls"]
                steps = [s for c in calls for s in c["native_steps"]]
                n = len(steps)
                assert [s["index"] for s in steps] == list(range(1,n+1))
                assert all(not (s["terminated"] or s["truncated"] or s["reward"] or s["success"]) for s in steps[:-1])
                s = steps[-1]
                assert n <= 300 and s["terminated"] == s["success"] == bool(s["reward"])
                assert s["reward"] in (0.,1.) and s["truncated"] == (n == 300 and not s["success"])
                assert s["terminated"] != s["truncated"] and n == obj["native"]["controls"]
                assert obj["native"]["success"] == s["success"] and s["reward"] == obj["native"]["native_return"]
                assert all(c["motor"] == "pi05_libero" and c["prompt"] == obj["native"]["released_prompt"]
                           and c["t"] == 5*i for i,c in enumerate(calls))
                for prev, cur in zip(calls, calls[1:]):
                    assert cur["public"] == prev["after_public"] and cur["inputs"] == cur["expected_inputs"]
                donor = prefixes.get(m, [])
                prefix_n = min(n,len(donor))
                assert action_bytes(v["action"] for v in steps[:prefix_n]) == action_bytes(donor[:prefix_n])
                assert all(c["prefix"] == (c["t"] < len(donor)) for c in calls)
                handoff = None
                if donor and n > len(donor):
                    at = calls[len(donor)//5]
                    assert at["inputs"] != calls[0]["inputs"] and at["public"] != calls[len(donor)//5-1]["public"]
                    handoff = dict(control_index=at["t"], noise_sha256=at["noise"], proposal_sha256=at["proposal"])
                if m == "raw":
                    ref = obj
                assert obj["native"]["reset"] == ref["native"]["reset"] and obj["env"] == ref["env"]
                assert all(calls[0][k] == ref["calls"][0][k] for k in ("proposal","inputs","public","noise"))
                assert all(a["noise"] == b["noise"] for a,b in zip(calls,ref["calls"]))
                group[m] = dict(success=int(s["success"]), controls=n, source_controls=prefix_n, handoff=handoff)
            rows.append(dict(row=row, branches=group))
    contrasts = {}
    for m,b in [(m,b) for m in modes[2:] for b in ("tuned","raw")] + [("tuned","raw")]:
        d = [r["branches"][m]["success"]-r["branches"][b]["success"] for r in rows]
        contrasts[f"{m}-{b}"] = dict(rescues=[r["row"] for r,v in zip(rows,d) if v==1],
                                     losses=[r["row"] for r,v in zip(rows,d) if v==-1], ties=d.count(0),
                                     effect_pp=100*sum(d)/len(d), paired_layout_bootstrap_95_pp=paired_interval(d))
    return dict(rows=rows, successes={m:sum(r["branches"][m]["success"] for r in rows) for m in modes},
                contrasts=contrasts, episode_sha256=hashes)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episodes", type=Path)
    p.add_argument("--donors", type=Path, default=Path(__file__).resolve().parent.parent /
                   "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz")
    a = p.parse_args()
    print(json.dumps(score(a.episodes, a.donors), indent=2))
