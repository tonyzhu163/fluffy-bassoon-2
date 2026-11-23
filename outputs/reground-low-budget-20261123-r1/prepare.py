"""CPU-only materialization. This is the only production stage allowed to open source pools."""
import argparse
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from common import (BUDGETS, CELL, REF_CODE, SOURCE_STORE, WORK, configure, digest,
                    ref, sha, subset_hash)

REF_FITS = ref.read(REF_CODE / "evidence/fitted-models.json")


def source_pool(seed, kind, rows):
    from reground.data import babble
    directory = babble.data_dir(CELL, seed) / kind
    paths = sorted(directory.glob("*.npz"))
    ref.require([p.name for p in paths] == [f"shard{r['shard']:03d}.npz" for r in rows], directory)
    per = []
    for p, r in zip(paths, rows, strict=True):
        ref.require(sha(p) == r["sha256"], p)
        eps = babble.load_episodes(p)
        ref.require(babble.n_transitions(eps) == r["transitions"] and len(eps) == r["episodes"], p)
        per.append([(e, dict(kind=kind, shard=p.name, episode=i, task=e["task"], length=len(e["A"])))
                    for i, e in enumerate(eps)])
    return [v[i] for i in range(max(map(len, per))) for v in per if i < len(v)]


def prefix(pool, n):
    from reground.data import babble
    eps = babble.take([p[0] for p in pool], n)
    return eps, [{**p[1], "start": 0, "stop": len(e["A"])} for p, e in zip(pool, eps)]


def prepare(seed, tmp):
    from reground.data import babble
    from reground.models import cache
    from reground.models.dynamics import split_episodes
    from reground.utils import stable_hash
    import json

    pools, sources = {}, {}
    for kind, file in (("babble", "native-babble-manifest"), ("vla_pi05", "native-policy-manifest")):
        with gzip.open(REF_CODE / f"evidence/{file}.json.gz", "rt") as f:
            rows = [r for r in json.load(f)["files"] if r["seed"] == seed and r["cell"] == CELL]
        rows.sort(key=lambda x: x["shard"])
        pool = source_pool(seed, kind, rows)
        pools[kind] = pool
        sources[kind] = dict(shards=len(rows), manifest_rows_sha256=digest(rows), episodes=len(pool),
                             transitions=sum(len(e["A"]) for e, key in pool))
    held = pools["vla_pi05"][-babble.HOLDOUT_VLA_EPS:]
    onp = babble.heldout_vla_pool(CELL, seed, "pi05")
    full_record = REF_FITS[seed]
    held_n = babble.n_transitions(onp)
    ref.require(len(onp) == len(held) == 5 and subset_hash(onp) == subset_hash([e for e, k in held])
                and full_record["data"][CELL]["heldout_vla_transitions"] == held_n, "Wrong holdout")
    eligible = pools["vla_pi05"][:-len(held)]
    report = dict(seed=seed, upstream=ref.PIN, policy=ref.POLICY, source=sources,
                  holdout=dict(episodes=[k for e, k in held], transitions=held_n, sha256=subset_hash(onp)),
                  reference_models={}, budgets={})
    for m in full_record["models"]:
        spec, directory = m["spec"], SOURCE_STORE / m["dir"]
        ref.require(ref.read(directory / "spec.json") == spec and sha(directory / "model.pt") == m["model_sha256"]
                    and ref.read(directory / "fit_metrics.json") == m["metrics"], directory)
        key = m["role"] + "_" + m["kind"]
        report["reference_models"][key] = dict(cache=directory.name, model_sha256=m["model_sha256"])
        if m["role"] == "dep":
            report["reference_models"][key]["init_sha256"] = sha(SOURCE_STORE / "fit-starts" / (directory.name + ".pt"))
    for n in (*BUDGETS, 10000):
        nb = round(0.6 * n)
        b, kb = prefix(pools["babble"], nb)
        v, kv = prefix(eligible, n - nb)
        eps, keys = b + v, kb + kv
        native = babble.adaptation_data(CELL, seed, n, 0.6)
        h = subset_hash(native)
        ref.require(subset_hash(eps) == h and babble.n_transitions(eps) == n, "Native prefix differs")
        for smaller, previous in report["budgets"].items():
            small_b = round(int(smaller) * .6)
            ref.require(subset_hash(babble.take(b, small_b)) == previous["babble_sha256"]
                        and subset_hash(babble.take(v, int(smaller) - small_b)) == previous["vla_sha256"],
                        "Source-specific prefixes are not nested")
        train, val = split_episodes(list(range(len(eps))), 0.1, seed=seed)
        record = dict(sha256=h, babble_sha256=subset_hash(b), vla_sha256=subset_hash(v),
                      episodes=keys, train_indices=list(map(int, train)), val_indices=list(map(int, val)),
                      train_transitions=sum(len(eps[i]["A"]) for i in train),
                      val_transitions=sum(len(eps[i]["A"]) for i in val), windows={})
        for kind in ("fwd", "idm"):
            m = next(m for m in full_record["models"] if m["role"] == "dep" and m["kind"] == kind)
            spec = cache.make_spec(kind, "dep", CELL, seed, n, .6)
            ref.require(spec == {**m["spec"], "budget": n}, "Spec drift")
            k = spec["cfg"]["unroll"] if kind == "fwd" else 1
            windows = lambda ids: sum(max(0, len(eps[i]["A"]) - k + 1) for i in ids)
            record["windows"][kind] = dict(K=k, L=spec["cfg"]["L"], training=windows(train),
                                           validation=windows(val), spec_hash=stable_hash(spec))
            if n == 10000:
                metrics=m["metrics"]
                ref.require(len(train) == metrics["n_train_eps"] and len(val) == metrics["n_val_eps"]
                            and (kind != "fwd" or windows(train) == metrics["fit"]["n_train"]), "Full split differs")
        if n != 10000:
            d = tmp / f"b{n}"
            d.mkdir()
            babble.save_episodes(d / "prefix.npz", eps)
            ref.require(subset_hash(babble.load_episodes(d / "prefix.npz")) == h, "Tape readback")
            record["tape_sha256"] = sha(d / "prefix.npz")
        report["budgets"][str(n)] = record
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", choices=ref.SEEDS, default=ref.SEEDS)
    args=parser.parse_args()
    configure()
    from reground.data import babble
    base = WORK / "inputs"
    base.mkdir(exist_ok=True, parents=True)
    with patch.object(babble, "RUNS", SOURCE_STORE / "runs"):
        for seed in args.seeds:
            dest = base / f"seed{seed}"
            with TemporaryDirectory(dir=base) as temp:
                staged = Path(temp) / "seed"
                staged.mkdir()
                report = prepare(seed, staged)
                ref.write(staged / "manifest.json", report)
                if dest.exists():
                    ref.require(report == ref.read(dest / "manifest.json") and all(
                        sha(dest / f"b{n}/prefix.npz") == report["budgets"][str(n)]["tape_sha256"]
                        for n in BUDGETS), "Published inputs differ")
                else:
                    staged.rename(dest)
                print(f"Verified seed {seed} {digest(report)}", flush=True)


if __name__ == "__main__":
    main()
