"""Stage 0 gates for noise-arm selection over a frozen pi0.5 (CPU, saved artifacts only).

outputs/noise-mode-icrl-20261029-r1/PLAN.md. Reads the outcomes.jsonl and trajectory
files written by scripts/diagnostics/maniskill_pi05_noise_cache.py. A condition is a
physics variant (C2) or a (variant, object) pair (C1). Per condition, arm success
p(c, k) is estimated on two disjoint halves of its scenes: the oracle arm is selected
on one half and scored on the other, so taking a max does not inflate it. Arm -1 is
native sampling.

- G1 headroom: oracle arm minus native, on evaluated conditions.
- G2 heterogeneity: oracle arm minus the best static arm chosen on reference conditions.
- G3 consistency: split-half Spearman correlation of arm success within a condition.
- G4 structure: per-observation log-score gain of a posterior over reference-condition
  arm profiles against independent Beta(1, 1) arms, scored sequentially.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


THRESHOLDS = dict(g1_headroom=.15, g2_heterogeneity=.10, g3_consistency=.5, g4_structure=0.)
NATIVE = -1


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty outcomes file")
    seen = set()
    for row in rows:
        if type(row["success_once"]) is not bool or type(row["arm"]) is not int or row["arm"] < NATIVE:
            raise ValueError("Require binary success and integer arms >= -1")
        identity = (row["variant"], row["scene_index"], row["arm"], row["repeat"])
        if identity in seen:
            raise ValueError(f"Duplicate variant/scene/arm/repeat {identity}")
        seen.add(identity)
    return rows


def condition_of(row, condition):
    if condition == "variant":
        return row["variant"]
    if condition == "object":
        return row["variant"], row["object"]
    raise ValueError("Condition is 'variant' or 'object'")


def success_table(rows, condition):
    """{condition: {scene_index: {arm: mean success over repeats}}}."""
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        sums[condition_of(row, condition)][row["scene_index"]][row["arm"]].append(float(row["success_once"]))
    table = {c: {s: {a: float(np.mean(v)) for a, v in arms.items()} for s, arms in scenes.items()}
             for c, scenes in sums.items()}
    arms = {a for scenes in table.values() for per_arm in scenes.values() for a in per_arm}
    for c, scenes in table.items():
        if any(set(per_arm) != arms for per_arm in scenes.values()):
            raise ValueError(f"Condition {c} lacks some arm on some scene")
    return table, sorted(arms)


def split_halves(scenes, seed):
    order = np.random.default_rng(seed).permutation(sorted(scenes))
    half = len(order) // 2
    if half < 1:
        raise ValueError("Each condition needs at least two scenes")
    return [int(s) for s in order[:half]], [int(s) for s in order[half:2*half]]


def profile(table, c, scenes, arms):
    return np.array([np.mean([table[c][s][a] for s in scenes]) for a in arms])


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start
        while stop + 1 < len(values) and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2
        start = stop + 1
    return ranks


def spearman(a, b):
    """Spearman correlation with average ranks; nan when either side is constant."""
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sequential_log_scores(observations, references):
    """Mean log predictive score per observation under both models.

    observations: ordered (arm_position, success) pairs from one condition.
    references: [conditions, arms] success profiles from other conditions.
    """
    references = np.clip(np.asarray(references, dtype=float), .02, .98)
    log_weights = np.zeros(len(references))
    hits, trials = defaultdict(float), defaultdict(float)
    mixture = independent = 0.
    for arm, success in observations:
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        p_mix = float(weights @ references[:, arm])
        p_ind = (hits[arm] + 1) / (trials[arm] + 2)
        mixture += np.log(p_mix if success else 1 - p_mix)
        independent += np.log(p_ind if success else 1 - p_ind)
        log_weights += np.log(references[:, arm] if success else 1 - references[:, arm])
        hits[arm] += success
        trials[arm] += 1
    return mixture / len(observations), independent / len(observations)


def gates(rows, *, condition, train, evaluate, seed=0):
    """Gate statistics for evaluated conditions; references exclude the evaluated one."""
    table, arms = success_table(rows, condition)
    if NATIVE not in arms or len(arms) < 3:
        raise ValueError("Require the native arm and at least two noise arms")
    noise_arms = [a for a in arms if a != NATIVE]
    native = arms.index(NATIVE)
    missing = [c for c in list(train) + list(evaluate) if c not in table]
    if missing:
        raise ValueError(f"Conditions without rows: {missing}")
    per_condition = []
    for c in evaluate:
        references = [t for t in train if t != c]
        if not references:
            raise ValueError(f"No reference conditions for {c}")
        static = noise_arms[int(np.argmax(np.mean(
            [profile(table, t, table[t], noise_arms) for t in references], 0)))]
        halves = split_halves(table[c], [seed, len(per_condition)])
        g1 = g2 = 0.
        for select, score in (halves, halves[::-1]):
            chosen = noise_arms[int(np.argmax(profile(table, c, select, noise_arms)))]
            scored = profile(table, c, score, arms)
            g1 += (scored[arms.index(chosen)] - scored[native]) / 2
            g2 += (scored[arms.index(chosen)] - scored[arms.index(static)]) / 2
        g3 = spearman(*(profile(table, c, half, noise_arms) for half in halves))
        rng = np.random.default_rng([seed, 1, len(per_condition)])
        observations = [(noise_arms.index(a), table[c][s][a] >= .5) for s in table[c] for a in noise_arms]
        observations = [observations[i] for i in rng.permutation(len(observations))]
        mixture, independent = sequential_log_scores(
            observations, [profile(table, t, table[t], noise_arms) for t in references])
        per_condition.append(dict(condition=c, scenes=len(table[c]), static_arm=static,
            native=float(profile(table, c, table[c], [NATIVE])[0]),
            best_arm_all_scenes=float(profile(table, c, table[c], noise_arms).max()),
            g1_headroom=float(g1), g2_heterogeneity=float(g2), g3_consistency=g3,
            g4_structure=float(mixture - independent)))
    summary = {}
    for key, threshold in THRESHOLDS.items():
        values = np.array([r[key] for r in per_condition], dtype=float)
        valid = values[np.isfinite(values)]
        mean = float(valid.mean()) if len(valid) else float("nan")
        summary[key] = dict(mean=mean, threshold=threshold, undefined_conditions=int((~np.isfinite(values)).sum()),
                            passed=bool(np.isfinite(mean) and (mean > threshold if key == "g4_structure"
                                                               else mean >= threshold)))
    return dict(condition=condition, arms=arms, train=list(train), evaluate=list(evaluate),
                per_condition=per_condition, summary=summary,
                all_passed=all(v["passed"] for v in summary.values()))


def determinism(rows):
    """Repeated (variant, scene, arm >= 0) rows: do complete issued trajectories match?"""
    groups = defaultdict(list)
    for row in rows:
        if row["arm"] != NATIVE:
            groups[row["variant"], row["scene_index"], row["arm"]].append(row)
    repeated = [g for g in groups.values() if len(g) > 1]
    matched = sum(len({r["issued_actions_sha256"] for r in g}) == 1 for g in repeated)
    outcome_matched = sum(len({r["success_once"] for r in g}) == 1 for g in repeated)
    distinct_slots = sum(len({r["slot"] for r in g}) > 1 for g in repeated)
    return dict(repeated_groups=len(repeated), trajectory_matched=matched, outcome_matched=outcome_matched,
                groups_in_distinct_slots=distinct_slots,
                deterministic=bool(repeated) and matched == len(repeated))


def load_trajectories(directory):
    """{row: issued [chunks, CHUNK, 7] array} from the cache script's trajectory files."""
    trajectories = {}
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path) as data:
            for row, issued in zip(data["row"], data["issued"]):
                trajectories[int(row)] = issued
    return trajectories


def mode_effect(rows, trajectories):
    """Mean issued-action distance between arms on the same scene, against native-vs-native.

    Distances use only each trajectory's first chunk, which starts from the same reset
    state for every arm, so the difference comes from the noise rather than divergent states.
    """
    by_scene = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["repeat"] == 0 or row["arm"] == NATIVE:
            by_scene[row["variant"], row["scene_index"]][row["arm"]].append(trajectories[row["row"]][0])
    between, native = [], []
    for arms in by_scene.values():
        first = [chunks[0] for arm, chunks in sorted(arms.items()) if arm != NATIVE]
        between += [float(np.linalg.norm(a - b)) for i, a in enumerate(first) for b in first[i+1:]]
        natives = arms.get(NATIVE, [])
        native += [float(np.linalg.norm(a - b)) for i, a in enumerate(natives) for b in natives[i+1:]]
    return dict(first_chunk_distance_between_arms=float(np.mean(between)) if between else float("nan"),
                first_chunk_distance_native_repeats=float(np.mean(native)) if native else float("nan"),
                pairs=len(between))


def condition_list(text, condition):
    if condition == "variant":
        return [int(x) for x in text.split(",")]
    return [tuple(int(y) for y in x.split(":")) for x in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, required=True, help="output directory of the cache script")
    parser.add_argument("--condition", choices=("variant", "object"))
    parser.add_argument("--train", help="variants '0,1' or variant:object pairs '0:3,0:4'")
    parser.add_argument("--evaluate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    gate_arguments = (args.condition, args.train, args.evaluate)
    if any(x is None for x in gate_arguments) and any(x is not None for x in gate_arguments):
        parser.error("Pass --condition, --train and --evaluate together, or none for checks only")
    rows = load_rows(args.cache_dir/"outcomes.jsonl")
    if args.condition is None:
        result = dict(summary=None, all_passed=None)
    else:
        result = gates(rows, condition=args.condition, train=condition_list(args.train, args.condition),
                       evaluate=condition_list(args.evaluate, args.condition), seed=args.seed)
    result["determinism"] = determinism(rows)
    trajectories = args.cache_dir/"trajectories"
    if trajectories.is_dir():
        result["mode_effect"] = mode_effect(rows, load_trajectories(trajectories))
    args.output.write_text(json.dumps(result, indent=2, default=str)+"\n")
    print(json.dumps(dict(summary=result["summary"], all_passed=result["all_passed"],
                          determinism=result["determinism"]), default=str))


if __name__ == "__main__":
    main()
