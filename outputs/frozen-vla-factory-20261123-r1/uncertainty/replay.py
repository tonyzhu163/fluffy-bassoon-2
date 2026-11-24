"""Replay pre-query routes against paired historical native outcomes; no motor runtime."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "757a44cd4374833cad790852df8b87ddf412e638"
SOURCE = "ur5e-pi05"
PANELS = {"gpu09test": SOURCE, "gpu09iiwa": "iiwa-pi05",
          "gpu06kinova3": "kinova3-pi05", "gpu08pi0fix": "ur5e-pi0"}
SEEDS = (41, 42, 43)
PRED_DIR = "outputs/symmeta-perceiver-retry-20261123-r1/evidence"
STATS_DIR = "outputs/symmeta-paired-statistics-20261123-r1"
REF_DIR = "outputs/symmeta-fk-crossarm-20261123-r1/evidence"
FALLBACKS = ("hs", "medoid")
# Operating points for descriptive coverage sensitivity, never used to tune the primary.
TAILS = (0.10, 0.25, 0.50)


def blob(path):
    return subprocess.check_output(["git", "-C", ROOT, "show", f"{BASE}:{path}"])


def read(path):
    raw = blob(path)
    return json.loads(gzip.decompress(raw) if path.endswith(".gz") else raw)


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, order=True)
class Key:
    panel: str
    index: int
    suite: str
    task_id: int
    transform: str

    @property
    def task(self):
        return (self.suite, self.task_id)

    @classmethod
    def from_row(cls, panel, row):
        return cls(panel, int(row["index"]), row["suite"], int(row["task_id"]), row["embodiment"])


@dataclass(frozen=True)
class Support:
    digest: str
    query_seed: int
    regime: str
    score: float
    medoid: str


def operator(yaw, gain, sign):
    """Isometric vectorization of the unclipped inverse action map, ignoring constant axes."""
    assert math.isfinite(yaw) and math.isfinite(gain) and gain > 0 and sign in (-1, 1)
    inverse = 1 / min(4.0, max(0.25, gain))
    angle = math.radians(-yaw)
    # The two planar basis commands produce the same squared deviation.
    return (math.sqrt(2) * inverse * math.cos(angle), math.sqrt(2) * inverse * math.sin(angle),
            inverse, sign)


def diagnostics(predictions):
    maps = [operator(p["yaw_deg"], p["xy_scale"], p["gripper_sign"]) for p in predictions]
    sums = [sum(math.dist(a, b) ** 2 for b in maps) for a in maps]
    return math.sqrt(sums[0] / 2), f"s{SEEDS[sums.index(min(sums))]}"


def load_supports():
    """Join signed reader outputs to verified passive tape identities. Ignore all query outcomes."""
    export = f"{PRED_DIR}/train02pro/predictions.json"
    contract = read(f"{PRED_DIR}/prepare01pro/contract.json")
    assert sha(blob(export)) == contract["predictions_sha256"]
    by_key = defaultdict(dict)
    for r in read(export)["rows"]:
        key = Key.from_row(PANELS[r["panel"]], r)
        seed = r["seed"]
        assert seed in SEEDS and seed not in by_key[key]
        by_key[key][seed] = (r["symmeta"], r["tape_sha256"], r["regime"])

    splits = read("outputs/libero-meta-learning-20261117-r1/SPLITS.json")
    tasks = {(suite, i) for suite, ids in splits["test"].items() for i in ids}
    suites = sorted(splits["test"])
    cross = read(f"{REF_DIR}/summary.json")
    refs = {}
    for panel, name in PANELS.items():
        if name == SOURCE:
            roll = read("outputs/symmeta-camera-command-20261123-r1/evidence/roll.json.gz")
            entries = []
            for support in roll["supports"]:
                r = support["case"]
                assert support["controls"] == 15 and support["selected_chunk_indices"] == [0, 1, 2]
                entries.append((r, support["npz"]["sha256"], r["support_init_state_id"],
                                r["query_init_state_id"], r["support_seed"], None))
        else:
            path = f"{REF_DIR}/{cross[panel]['artifact']}"
            assert sha(blob(path)) == cross[panel]["sha256"]
            evidence = read(path)
            assert evidence["panel"] == panel and evidence["native_reader_qualified"]
            entries = []
            for r in evidence["rows"]:
                assert r["reader_available"] and r["native_reader_qualified"]
                entries.append((r, r["support_sha256"], r["support_layout"], r["query_layout"],
                                r["support_seed"], r["query_seed"]))
        assert sorted(r["index"] for r, *_ in entries) == list(range(96))
        for r, digest, start, query, start_seed, query_seed in entries:
            key = Key.from_row(name, r)
            offset = 10000 * suites.index(key.suite) + 100 * key.task_id
            assert (start, query, start_seed) == (11, 13, 31800011 + offset)
            assert query_seed in (None, 41800013 + offset) and key.task in tasks
            rows = by_key[key]
            assert set(rows) == set(SEEDS) and all(p[1] == digest for p in rows.values())
            regimes = {p[2] for p in rows.values()}
            assert len(regimes) == 1 and key not in refs
            score, medoid = diagnostics([rows[s][0] for s in SEEDS])
            refs[key] = Support(digest, 41800013 + offset, regimes.pop(), score, medoid)
    assert refs.keys() == by_key.keys() and len({s.digest for s in refs.values()}) == len(refs) == 384
    return refs


def project_labels(raw_rows, supports, source_only):
    labels = {}
    for r in raw_rows:
        if (r["panel"] == SOURCE) != source_only:
            continue
        key = Key.from_row(r["panel"], r)
        assert key not in labels and r["regime"] == supports[key].regime
        assert r["variant"] == "BASE" and int(r["query_init_state_id"]) == 13
        assert int(r["query_reset_seed"]) == supports[key].query_seed
        arm = {s: int(r[s]) for s in ("s41", "s42", "s43", "hs")}
        assert set(arm.values()) <= {0, 1}
        labels[key] = arm
    assert labels.keys() == {k for k in supports if (k.panel == SOURCE) == source_only}
    return labels


def alternative(kind, support):
    return support.medoid if kind == "medoid" else "hs"


def select(score, cutoff, fallback):
    return fallback if score >= cutoff else "s41"


def calibrate(training, labels, kind):
    """Fit one high-ambiguity threshold to paired source success. Prefer fewer switches on ties."""
    scores = sorted({s.score for s in training.values()})
    cutoffs = [math.inf, *reversed([(a + b) / 2 for a, b in zip(scores, scores[1:])]), -math.inf]
    best = None
    for cutoff in cutoffs:
        switches = [k for k, s in training.items() if s.score >= cutoff and alternative(kind, s) != "s41"]
        gain = sum(labels[k][alternative(kind, training[k])] - labels[k]["s41"] for k in switches)
        value = (gain, -len(switches))
        if best is None or value > best[0]:
            best = value, cutoff
    return best[1], best[0][0], -best[0][1]


def make_routes(supports, source_labels):
    targets = {k: s for k, s in supports.items() if k.panel != SOURCE}
    assert set(source_labels) == {k for k in supports if k.panel == SOURCE}
    routes, folds = {}, {}
    for task in sorted({k.task for k in targets}):
        train = {k: supports[k] for k in source_labels if k.task != task}
        test = {k: s for k, s in targets.items() if k.task == task}
        assert len(train) == 84 and len(test) == 36
        assert {k.task for k in train}.isdisjoint(k.task for k in test)
        assert {s.digest for s in train.values()}.isdisjoint(s.digest for s in test.values())
        scores = [s.score for s in train.values()]
        quantiles = statistics.quantiles(scores, n=100, method="inclusive")
        thresholds = {f"hs_top_{round(100*t)}pct": quantiles[round(100*(1-t))-1] for t in TAILS}
        fold = {}
        for name in FALLBACKS:
            cutoff, gain, n_switches = calibrate(train, source_labels, name)
            thresholds[name] = cutoff
            fold[name] = dict(cutoff=cutoff if math.isfinite(cutoff) else str(cutoff),
                              source_net=gain, source_switches=n_switches)
        folds["/".join(map(str, task))] = fold
        for k, s in test.items():
            routes[k] = {name: select(s.score, c, alternative(name, s)) for name, c in thresholds.items()}
    assert routes.keys() == targets.keys()
    return routes, folds


def task_interval(net_by_task, denominator):
    """Exact percentile interval over 8^8 ordered whole-task resamples. Equal task denominators."""
    groups = Counter(net_by_task.values())
    dist = Counter({0: 1})
    for _ in net_by_task:
        new = Counter()
        for a, n in dist.items():
            for b, m in groups.items():
                new[a+b] += n*m
        dist = new
    cumulative = 0
    bounds = []
    for effect, count in sorted(dist.items()):
        cumulative += count
        while len(bounds) < 2 and cumulative * 40 >= (1, 39)[len(bounds)] * sum(dist.values()):
            bounds.append(round(100 * effect / denominator, 3))
    return bounds


def summarize(keys, labels, supports, routes, name):
    nets = Counter()
    sizes = Counter()
    random_groups = defaultdict(list)
    fixed = Counter()
    wins = losses = switches = successes = ceiling = selected_fallback_success = 0
    for key in keys:
        y = labels[key]
        branch = routes[key][name]
        alt = alternative(name, supports[key])
        chosen, base = y[branch], y["s41"]
        fixed.update(y)
        fixed["always_medoid"] += y[supports[key].medoid]
        delta = chosen - base
        successes += chosen
        wins += int(delta > 0)
        losses += int(delta < 0)
        switches += int(branch != "s41")
        selected_fallback_success += chosen * int(branch != "s41")
        ceiling += max(base, y[alt])
        nets[key.task] += delta
        sizes[key.task] += 1
        random_groups[key.panel, key.task, alt].append((branch != "s41", y[alt] - base))
    assert len(sizes) == 8 and len(set(sizes.values())) == 1
    expected_gain = sum(sum(sw for sw, _ in group) * statistics.mean(d for _, d in group)
                        for group in random_groups.values())
    return dict(n=len(keys), successes=successes, fixed_successes=dict(fixed), switches=switches,
                reader_coverage=(len(keys)-switches)/len(keys), successes_in_switches=selected_fallback_success,
                wins=wins, losses=losses, delta_pp=round(100*(wins-losses)/len(keys), 3),
                task_ci95_pp=task_interval(nets, len(keys)), oracle_two_branch_successes=ceiling,
                random_switch_expected_successes=round(fixed["s41"]+expected_gain, 4),
                net_successes_by_task={"/".join(map(str, k)): v for k, v in sorted(nets.items())})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE/"metrics.json")
    args = parser.parse_args()
    supports = load_supports()
    raw_rows = list(csv.DictReader(io.StringIO(blob(f"{STATS_DIR}/outcomes.csv").decode())))
    source = project_labels(raw_rows, supports, source_only=True)
    routes, folds = make_routes(supports, source)
    decisions = [[k.panel, k.index, r] for k, r in sorted(routes.items())]
    digest = sha(json.dumps(decisions, sort_keys=True).encode())
    # Make target outcomes unparseable and rerun the source projection and routing.
    poison = dict.fromkeys(("hs", "s41", "s42", "s43"), "masked")
    masked = [r if r["panel"] == SOURCE else {**r, **poison} for r in raw_rows]
    again, _ = make_routes(supports, project_labels(masked, supports, source_only=True))
    assert again == routes
    # Project the actual target success columns only after all decisions are fixed.
    target = project_labels(raw_rows, supports, source_only=False)
    for held in {k.task for k in target}:
        # Independently perturb every held-task source outcome. Its target fold must stay fixed.
        altered = {k: ({a: 1-y for a, y in arm.items()} if k.task == held else arm)
                   for k, arm in source.items()}
        changed, _ = make_routes(supports, altered)
        assert all(changed[k] == routes[k] for k in target if k.task == held)
    results = {}
    for name in (*FALLBACKS, *(f"hs_top_{round(100*t)}pct" for t in TAILS)):
        sets = {"all": list(target), "ood": [k for k in target if supports[k].regime != "familiar"]}
        if name in FALLBACKS:
            sets.update({p: [k for k in target if k.panel == p] for p in sorted({k.panel for k in target})})
        results[name] = {group: summarize(keys, target, supports, routes, name) for group, keys in sets.items()}
    evidence = dict(input_revision=BASE, gpu_hours=0, source_cases=len(source), target_cases=len(target),
                    support_npz_identities=len(supports), reader_predictions=3*len(supports),
                    scored_tasks=8, target_plan_sha256=digest, source_folds=folds, results=results)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print("384 support identities, 8 disjoint folds, target-outcome and held-task perturbations passed.")
    for name in FALLBACKS:
        r = results[name]["all"]
        print(f"{name}: {r['successes']}/{r['n']}; {r['switches']} switches; "
              f"{r['wins']} wins, {r['losses']} losses; task 95% {r['task_ci95_pp']} pp")
    print(f"Target decisions SHA-256 {digest}")


if __name__ == "__main__":
    main()
