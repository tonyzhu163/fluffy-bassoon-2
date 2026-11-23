"""Paired arithmetic effects conditional on fixed tasks, and physical attempt-ledger accounting."""
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import reference as ref
import retry
from worker import selected

ARCHIVE_SHA = "8aa0fdc6eee00c69779bb43911746032e18a7dd951ecdacaea6774a3a3195263"
N_BOOT = 10000
BOOT_SEED = 4115
METHODS = (*retry.ARMS, "stream0", "stream5")
CONTRASTS = ((1,0),(0,2),(1,2),(0,3),(1,3))
METRICS = ("success","unsafe")


def archive():
    path = ref.ROOT/"final-native-v1/native-cases.json.gz"
    data = path.read_bytes()
    ref.require(hashlib.sha256(data).hexdigest()==ARCHIVE_SHA, "Historical case table differs")
    rows = json.loads(gzip.decompress(data))
    by_key = defaultdict(dict)
    for row in rows:
        key = (row["task"],int(row["init_id"]),int(row["rep"]))
        seed = int(row["adapt_seed"])
        ref.require(seed not in by_key[key] and row["policy"]=="pi05" and row["split"]=="test"
                    and row["shift"]=="rot_z" and float(row["level"])==90, "Historical case differs")
        by_key[key][seed] = row
    grid = {k for batch in retry.schedule() for k in batch}
    ref.require(len(rows)==1400 and set(by_key)==grid and all(set(v)==set(ref.SEEDS) for v in by_key.values()),
                "Incomplete historical comparison")
    return by_key


def interval(draws):
    return np.quantile(draws,[.025,.975]).tolist()


def analysis(cases):
    """Use the same paired resample for all methods, metrics and arithmetic contrasts."""
    rng = np.random.default_rng(BOOT_SEED)
    groups = defaultdict(list)
    for c in cases:
        groups[c["task"]].append(c)
    metrics = {}
    for metric in METRICS:
        tables = {}
        for task, cases_t in sorted(groups.items()):
            vals = np.array([[c[m][metric] for m in METHODS] for c in cases_t],dtype=float)
            n = len(vals)
            draw = rng.integers(0,n,(N_BOOT,n))
            start_samples = vals[draw].mean(axis=1)
            groups_s = defaultdict(list)
            for i,c in enumerate(cases_t):
                groups_s[c["support_id"]].append(i)
            members = list(groups_s.values())
            totals = np.stack([vals[g].sum(axis=0) for g in members])
            sizes = np.array(list(map(len,members)))
            selects = rng.integers(len(members),size=(N_BOOT,len(members)))
            support_samples = totals[selects].sum(axis=1)/sizes[selects].sum(axis=1)[:,None]
            tables[task] = (vals,start_samples,support_samples)
        all_vals = np.concatenate([g[0] for g in tables.values()])
        tables["all"] = (all_vals, *[sum(len(g[0])*g[i] for g in tables.values())/len(all_vals) for i in (1,2)])
        results = {}
        for task,(values,bs,sb) in tables.items():
            methods = {}
            for i,m in enumerate(METHODS):
                methods[m] = dict(rate=float(values[:,i].mean()), start_ci95=interval(bs[:,i]))
                if m!="stream5":
                    methods[m]["count"] = int(values[:,i].sum())
            differences = {}
            for i,j in CONTRASTS:
                delta = values[:,i]-values[:,j]
                stats = dict(delta=float(delta.mean()), start_ci95=interval(bs[:,i]-bs[:,j]),
                             support_cluster_ci95=interval(sb[:,i]-sb[:,j]))
                if METHODS[j]!="stream5":
                    stats.update(only_left=int((delta==1).sum()),only_right=int((delta==-1).sum()),
                                 both=int(((values[:,i]==1)&(values[:,j]==1)).sum()),
                                 neither=int(((values[:,i]==0)&(values[:,j]==0)).sum()))
                differences[f"{METHODS[i]}_minus_{METHODS[j]}"] = stats
            results[task] = dict(n_unique_starts=len(values),methods=methods,contrasts=differences)
        metrics[metric] = results
    return metrics


def journals(paths):
    complete, uncertain = Counter(), Counter()
    for path in paths:
        pending = None
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):  # a kill can interrupt a single terminal write
                break
            event,stage,op,n = json.loads(line)
            if event=="begin" and pending is None:
                pending = (stage,op,n)
            else:
                ref.require(event=="end" and pending==(stage,op,n),f"Invalid native journal {path}")
                complete[(stage,op)] += n
                pending = None
        if pending:
            stage,op,n = pending
            uncertain[(stage,op)] += n
    def encode(tally):
        out = {}
        for (stage,op),v in sorted(tally.items()):
            out.setdefault(stage,{})[op]=v
        for row in out.values():
            row["settle_controls_from_resets"] = retry.N_SETTLE*row.get("reset",0)
        return out
    return dict(confirmed=encode(complete),inflight_at_most=encode(uncertain))


def report(store, phase):
    from reground.eval.jobs import load_traces
    grid = retry.schedule()
    history = archive()
    @lru_cache(maxsize=1)
    def prior(shard):
        return load_traces(ref.ROOT/f"symmeta-native/shards/seed0/rot_z/90/symmeta/shard{shard:03d}")
    cases = []
    all_paths = []
    for b in sorted(selected(grid,phase)):
        path = store/"batches"/f"batch{b:03d}"
        retry.validate(path,grid[b])
        all_paths.extend(path.glob("*.jsonl"))
        support = ref.read(path/"support.json")
        arms = [ref.read(path/(arm+".json"))["rows"] for arm in retry.ARMS]
        with np.load(path/"opening.npz") as opening:
            opening_sha = ref.read(path/(retry.ARMS[0]+".json"))["opening_sha256"]
            props,x0 = opening["proposal"],opening["X0"]
            for i,k in enumerate(grid[b]):
                old = history[k]
                orig = prior(int(old[0]["shard"]))[k]
                ref.require(all(int(r["noise_seed"])==int(arms[0][i]["noise_seed"]) for r in old.values()),
                            "Historical query nonce differs")
                ref.require(np.array_equal(x0[i],orig["X"][0]),"Historical reset robot state differs")
                prefix = min(10,len(orig["A_vla"]))
                record = dict(task=k[0], init_id=k[1], rep=k[2], label=0, support_id=support[i]["support"][1],
                              support_safety=support[i]["support_safety"],
                              estimate=support[i]["estimate"], noise_seed=arms[0][i]["noise_seed"],batch=b,
                              paired_opening_sha256=opening_sha, archived_seed0_first_proposal_prefix=prefix,
                              archived_seed0_proposal_exact=np.array_equal(props[i,:prefix],orig["A_vla"][:prefix]),
                              historical_five={s:{k:r[k] for k in r if k.startswith(("symmeta_","reground_","base_"))
                                                  and not k.endswith("exact")} for s,r in old.items()})
                for m,rows in zip(retry.ARMS,arms,strict=True):
                    record[m]=rows[i]
                record["stream0"]={m:old[0]["symmeta_"+m] for m in METRICS}
                record["stream5"]={m:sum(r["symmeta_"+m] for r in old.values())/5 for m in METRICS}
                cases.append(record)
    metrics = analysis(cases)
    matched = [c for c in cases if c["archived_seed0_proposal_exact"]]
    partition = [r for b in (store/"batches").glob("batch*") for r in b.glob("*.jsonl")]
    partial = list((store/"attempts").glob("*/batch*/*.jsonl"))
    receipts = {}
    for path in sorted((store/"attempts").glob("*/launch.txt")):
        attempt = path.parent
        receipts[attempt.name] = dict(launch=path.read_text().split(),
            exit=(attempt/"exit.txt").read_text().split() if (attempt/"exit.txt").exists() else None)
    if phase=="full":
        ref.require(len(cases)==280 and len({(c["task"],c["support_id"]) for c in cases})==120,"Incomplete full panel")
    result = dict(phase=phase, code=os.environ["RETRY_PIN"], historical_sha256=ARCHIVE_SHA, new_seed_labels=[0],
                  original_seed_labels=list(ref.SEEDS), original_rows_per_arm=1400, new_pairs=len(cases),
                  distinct_support_layouts=len({(c["task"],c["support_id"]) for c in cases}),
                  conditional_on="Fixed tasks, reader, policy and preassigned support mapping.",
                  intervals=dict(draws=N_BOOT,seed=BOOT_SEED,method="paired percentile bootstrap within task",
                                 sensitivity="resample complete support-layout clusters within task"),
                  paired=metrics, archived_prefix_exact=len(matched),
                  archived_seed0_matched_prefix_subset=analysis(matched) if 0<len(matched)<len(cases) else None,
                  support_controls_per_deployment=15, support_seconds_at_20hz=.75,
                  explicit_resets_per_deployment=2, settle_controls_per_deployment=20,
                  support_unsafe=sum(c["support_safety"]["unsafe"] for c in cases),
                  either_support_or_query_unsafe={arm:sum(bool(c["support_safety"]["unsafe"] or c[arm]["unsafe"])
                                                        for c in cases) for arm in retry.ARMS},
                  intended_physical_interactions=journals(all_paths),
                  all_sealed_and_partial_physical_interactions=journals(partition+partial),attempts=receipts,
                  native_budget_note="Count confirmed reset calls ×10 native settling controls. Failed native reset "
                     "retries can add up to ten settling controls per internal retry. Inflight counts bound "
                     "unacknowledged calls. Simulator construction is runtime setup. Opening capture adds zero resets.")
    dest = store/("readback-"+phase)
    ref.write(dest/"cases.json",cases)
    ref.write(dest/"summary.json",result)
    print(f"{phase}: {len(cases)} pairs, {len(matched)} exact historical proposal prefixes; {dest}",flush=True)


def selftest(out):
    retained = archive()
    fixtures = []
    for (task,i,rep),seeds in retained.items():
        v = seeds[0]["symmeta_success"]
        row = dict(task=task,support_id=(i-15)%15)
        for m in METHODS:
            row[m]=dict(success=v,unsafe=0)
        row["translation_wrist"]=dict(success=1,unsafe=1)
        fixtures.append(row)
    r=analysis(fixtures)
    n = sum(c["stream0"]["success"] for c in fixtures)
    allrow=r["success"]["all"]["contrasts"]
    dx=allrow["translation_wrist_minus_translation"]
    assert dx["only_left"]==280-n and dx["only_right"]==0
    assert np.isclose(dx["delta"],(280-n)/280)
    assert allrow["translation_minus_stream0"]["start_ci95"]==[0,0]
    assert r["unsafe"]["all"]["contrasts"]["translation_wrist_minus_translation"]["support_cluster_ci95"]==[1,1]
    log=out/"journal-fixture.jsonl"
    log.write_text('["begin","support","reset",4]\n["end","support","reset",4]\n["begin","support","step",4]\n')
    tally=journals([log])
    assert tally["confirmed"]["support"]["settle_controls_from_resets"]==40 and tally["inflight_at_most"]["support"]["step"]==4
    ref.write(out/"passed.json",dict(code=os.environ["RETRY_PIN"], historical_unique_starts=len(retained),
              original_rows=sum(map(len,retained.values())),original_seed0_streaming_success=n,
              fixture_effect=dx, ledger_fixture=tally, gpu_hours=0))
