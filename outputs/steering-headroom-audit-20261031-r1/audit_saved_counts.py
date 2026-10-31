"""Ordinary saved-byte/count analysis only; no repository, model or test imports."""

import csv
import hashlib
import io
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = "fd4a7c91207210951e5bbfc3ecb80c61fedd9052"
CSV = "outputs/libero40-fixed100-target-plan-20261031-r1/case-results.csv"
CORPUS = "outputs/libero40-train-corpus-statistics-20261030-r1/summary.json"
EXPECTED = {
    CSV: "0072f6de5a1351242e8f31b85b805181882d73a72fa69f7d3460954bdd6a49ad",
    CORPUS: "befd494cea8860afd786a07e9273456793c09834712faab7ade97a4fb85a6f81",
}


def source_bytes(relative):
    return subprocess.check_output(["git", "show", BASE + ":" + relative], cwd=ROOT)


def source_receipt(relative):
    data = source_bytes(relative)
    digest = hashlib.sha256(data).hexdigest()
    if relative in EXPECTED and digest != EXPECTED[relative]:
        raise ValueError("Saved source changed: " + relative)
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    return {"path": relative, "bytes": len(data), "sha256": digest,
            "git_blob_sha1": blob,
            "url": f"https://github.com/tonyzhu163/fluffy-bassoon-2/blob/{BASE}/{relative}"}


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return math.fsum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
                     for j in range(k + 1))


def clopper_pearson(k, n, alpha):
    """Two-sided exact binomial interval, numerical tail inversion."""
    tail = alpha / 2
    lower, upper = 0.0, 1.0
    if k:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if 1 - binomial_cdf(k - 1, n, mid) < tail:
                lo = mid
            else:
                hi = mid
        lower = (lo + hi) / 2
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if binomial_cdf(k, n, mid) > tail:
                lo = mid
            else:
                hi = mid
        upper = (lo + hi) / 2
    return [lower, upper]


def paired(rows, treatment, comparator):
    n = len(rows)
    cells = Counter((int(row[treatment]), int(row[comparator])) for row in rows)
    wins, losses = cells[1, 0], cells[0, 1]
    discordant = wins + losses
    # Each marginal interval has 97.5% coverage. Bonferroni joint coverage
    # >=95%; subtracting their bounds covers p(win)-p(loss). No independence
    # between the win/loss counts is assumed. Across pairs, iid is assumed.
    win_ci = clopper_pearson(wins, n, .025)
    loss_ci = clopper_pearson(losses, n, .025)
    rd_ci = [max(-1.0, win_ci[0] - loss_ci[1]),
             min(1.0, win_ci[1] - loss_ci[0])]
    pvalue = (1.0 if not discordant else min(
        1.0, 2 * sum(math.comb(discordant, j)
                     for j in range(min(wins, losses) + 1)) / 2 ** discordant))
    return {
        "treatment": treatment, "comparator": comparator, "n_case_pairs": n,
        "treatment_successes": cells[1, 1] + wins,
        "comparator_successes": cells[1, 1] + losses,
        "wins": wins, "losses": losses,
        "ties": cells[1, 1] + cells[0, 0],
        "both_success": cells[1, 1], "both_failure": cells[0, 0],
        "risk_difference": (wins - losses) / n,
        "risk_difference_pp": 100 * (wins - losses) / n,
        "risk_difference_95pct_conservative_ci": rd_ci,
        "risk_difference_95pct_conservative_ci_pp": [100 * x for x in rd_ci],
        "win_probability_97_5pct_CP": win_ci,
        "loss_probability_97_5pct_CP": loss_ci,
        "exact_two_sided_McNemar_p": pvalue,
        "discordant_task_ids": {
            "wins": [int(r["task_id"]) for r in rows if int(r[treatment]) > int(r[comparator])],
            "losses": [int(r["task_id"]) for r in rows if int(r[treatment]) < int(r[comparator])],
        },
    }


def holm(comparisons):
    running = 0.0
    for rank, entry in enumerate(sorted(comparisons, key=lambda x: x["exact_two_sided_McNemar_p"])):
        running = max(running, min(1.0, (len(comparisons) - rank) * entry["exact_two_sided_McNemar_p"]))
        entry["Holm_adjusted_p_within_audit_family"] = running


def main():
    provenance = [source_receipt(x) for x in [CSV, CORPUS,
        "outputs/libero40-fixed100-target-plan-20261031-r1/RESULT.md",
        "outputs/libero40-fixed100-target-plan-20261031-r1/cost-receipt.json",
        "outputs/libero40-train-corpus-statistics-20261030-r1/README.md",
        "docs/frozen-vla-libero/EVALUATION.md",
        "outputs/icrl-headroom-diagnosis-20261031-r1/PLAN.md"]]
    rows = list(csv.DictReader(io.StringIO(source_bytes(CSV).decode())))
    if len(rows) != 90 or sorted(int(r["task_id"]) for r in rows) != list(range(90)):
        raise ValueError("Panel is not the complete unique 90-case bank")
    k0 = ["native/k0", "ad_full/k0", "ad_static/k0", "dpt_full/k0", "dpt_static/k0"]
    k1 = ["ad_full/k1", "ad_episode_reset/k1", "dpt_full/k1", "dpt_episode_reset/k1"]
    if any(r[c] not in ("0", "1") for r in rows for c in k0):
        raise ValueError("Incomplete/nonbinary K0 panel")
    selected = [r for r in rows if r[k1[0]] != ""]
    if [int(r["task_id"]) for r in selected] != list(range(0, 90, 10)):
        raise ValueError("K1 fixed-nine bank changed")
    if any(r[c] not in ("0", "1") for r in selected for c in k1):
        raise ValueError("Incomplete/nonbinary K1 panel")
    if any(r[c] != "" for r in rows if r not in selected for c in k1):
        raise ValueError("Unexpected K1 records")
    primary = [paired(rows, c, "native/k0") for c in k0[1:]]
    k1_comparisons = [paired(selected, c, "native/k0") for c in k1]
    k1_comparisons += [paired(selected, "ad_full/k1", "ad_episode_reset/k1"),
                       paired(selected, "dpt_full/k1", "dpt_episode_reset/k1")]
    controls = [paired(rows, "ad_full/k0", "ad_static/k0"),
                paired(rows, "dpt_full/k0", "dpt_static/k0")]
    for family in (primary, k1_comparisons, controls):
        holm(family)
    source = json.loads(source_bytes(CORPUS))
    patterns = source["case_outcome_patterns"]
    cases = source["failure_then_later_success_cases"]
    total = sum(patterns.values())
    recovered_from_patterns = sum(n for pattern, n in patterns.items()
        if any(pattern[i] == "F" and pattern[j] == "S"
               for i in range(4) for j in range(i + 1, 4)))
    stage_groups = Counter()
    learned_ids, transition_learned_ids = [], []
    ordered_pairs = 0
    for case in cases:
        pattern, stages = case["pattern"], case["stages"]
        pairs = [[i, j] for i in range(4) for j in range(i + 1, 4)
                 if pattern[i] == "F" and pattern[j] == "S"]
        if pairs != case["ordered_failure_success_pairs"]:
            raise ValueError("Recorded recovery-pair mismatch")
        ordered_pairs += len(pairs)
        stage_set = set(stages)
        group = ("warmup_only" if stage_set == {"warmup"} else
                 "learned_only" if stage_set == {"learned"} else "transition")
        stage_groups[group] += 1
        if any(stages[i] == stages[j] == "learned" for i, j in pairs):
            learned_ids.append(case["case_id"])
            if group == "transition":
                transition_learned_ids.append(case["case_id"])
    expected = source["case_counts"]
    if not (total == 527 and recovered_from_patterns == len(cases) == 60
            and len({c["case_id"] for c in cases}) == 60
            and len(learned_ids) == expected["cases_with_learned_failure_then_learned_success"] == 10
            and stage_groups == {"warmup_only": 44, "learned_only": 8, "transition": 8}
            and len(transition_learned_ids) == 2 and ordered_pairs == 132):
        raise ValueError("Recovery reconciliation failed")
    same = Counter(r[k0[0]] for r in rows if len({r[c] for c in k0}) == 1)
    result = {
        "schema": "saved_steering_headroom_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "remote_main_reviewed_at_start": BASE,
        "scope": "Saved CSV/JSON/count arithmetic only on macOS; no tests, models, robot/simulator actions, GPU work, scheduler query, monitor or source expansion.",
        "sources": provenance,
        "panel": {"K0_task_ids": list(range(90)), "K1_task_ids": list(range(0, 90, 10)),
                  "K0_successes": {c: sum(int(r[c]) for r in rows) for c in k0},
                  "K1_successes": {c: sum(int(r[c]) for r in selected) for c in ["native/k0"] + k1},
                  "all_five_K0_same_outcome": {"cases": sum(same.values()),
                      "all_success": same["1"], "all_failure": same["0"]}},
        "inference": {
            "effect": "P(treatment success, comparator failure) - P(treatment failure, comparator success); observed estimate (wins-losses)/whole-panel n.",
            "interval": "Conservative pointwise >=95% paired risk-difference interval under iid pair sampling: subtract simultaneous bounds of two marginal 97.5% Clopper-Pearson intervals via Bonferroni. Not a Wald interval, equivalence bound, or CI obtained by conditioning on discordant cases.",
            "interval_formula": "[L_CP(w,n;alpha=.025)-U_CP(l,n;alpha=.025), U_CP(w,n;alpha=.025)-L_CP(l,n;alpha=.025)]",
            "coverage_argument": "Each marginal misses with probability <=.025. Union bound gives joint coverage >=.95; subtraction covers p_win-p_loss. The two category counts need not be independent; case pairs must satisfy the binomial sampling model.",
            "exact_test": "Two-sided conditional McNemar: min(1,2*P[Binomial(w+l,.5)<=min(w,l)]); D=0 returns1.",
            "multiplicity": "Holm p-adjustment separately within four K0/native, six K1, and two K0/full-static audit-defined families. CIs are pointwise, not simultaneous across comparisons.",
            "assumptions_and_limits": ["These are all90 fixed task/state0/seed123 cases, not random independent training seeds; K1 reuses nine tasks from that bank.",
                "Intervals are conditional sampling-model sensitivity summaries. Task-family dependence, heterogeneous effects, shared models and cross-process decoding mean they are not guaranteed population or across-training-seed confidence statements.",
                "Exact McNemar assumes independent case pairs and equal win/loss chance under its null. Its exactness does not cure the panel-design limitations.",
                "The empirical whole-panel risks/counts are exact descriptions of this saved bank; inferential uncertainty about unseen tasks/states/execution or training seeds is not identified by one observation per cell.",
                "K1 comparisons with native use native K0 on the same nine cases; K1 learned policies paid actual support, while native has no support. Full/reset is the closer equal-support-cost history control.",
                "No post-hoc informative-case-only denominator; no power analysis, equivalence test or claim that absent significance bounds all practically relevant effects."]},
        "K0_vs_native": primary, "K1_full_reset_native": k1_comparisons,
        "K0_full_vs_static_secondary": controls,
        "source_recovery_reconciliation": {
            "status": "RESOLVED_FROM_ORIGINAL_SAVED_RECEIPT",
            "denominator_complete_K4_cases": total,
            "any_failure_then_later_success_cases": 60,
            "any_failure_then_later_success_rate": 60 / total,
            "all_four_warmup_recovery_cases": stage_groups["warmup_only"],
            "all_four_learned_recovery_cases": stage_groups["learned_only"],
            "warmup_learned_transition_recovery_cases": stage_groups["transition"],
            "learned_failure_then_later_learned_success_cases": len(learned_ids),
            "learned_pair_rate_over_all_K4_cases": len(learned_ids) / total,
            "learned_pair_transition_case_ids": transition_learned_ids,
            "learned_pair_case_ids": learned_ids,
            "ordered_failure_success_episode_pairs": ordered_pairs,
            "first_episode_failure_then_later_success_cases": expected["first_episode_failure_then_later_success_cases"],
            "all_success_K4_cases": patterns["SSSS"],
            "aggregate": source["aggregate"], "stages": source["stages"],
            "interpretation": "10 is a subset of60:8 entirely learned cases plus2 transition cases. These are recorded within-K4 recoveries, not evidence of algorithmic improvement, a memory effect, or a causal reward response.44 recovery cases are entirely untrained warmup. SAC weights/exploration can change between episodes; stage rates use different chronological cases and sampling distributions.",
            "snapshot_limit": "527 cases/2108episodes is the historical source snapshot, not the later2332episode bank. Do not silently change denominators or treat nominal start hashes as measured post-settle physical states (all2108 missing)."},
        "repeat_uncertainty": {"source": "docs/frozen-vla-libero/EVALUATION.md: separate historical native student/noise-wave comparison",
            "reported_native_outcome_flips": 4, "historical_pairs": 90,
            "scope": "Reported matched initial scene/seed/input/noise but differing decoded actions across processes; not repeated fixed100 observations and not a calibrated iid flip probability for this panel.",
            "current_fixed100_replicates_per_method_case": 1,
            "claim": "Repeat and training-seed variance remain unresolved. Do not subtract4/90 from effects or equate it with an irreducible noise floor."},
        "method_references": [
            "https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC9447366/"],
        "implementation_note": "The conservative Bonferroni-CP construction is explicitly derived here; the matched-pairs reference is background, not a claim to implement its tighter melded interval.",
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for family in (primary, k1_comparisons, controls):
        for r in family:
            print(r["treatment"], "vs", r["comparator"], r["wins"], r["losses"], r["ties"],
                  round(r["risk_difference_pp"], 3),
                  [round(x, 3) for x in r["risk_difference_95pct_conservative_ci_pp"]],
                  r["exact_two_sided_McNemar_p"])


if __name__ == "__main__":
    main()
