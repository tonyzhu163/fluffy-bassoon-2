# Saved steering-headroom audit — 2026-10-31

**The complete fixed100 panel shows small paired differences, not equivalence or a ceiling on all steering interfaces.** AD full and static each add one success over native across 90 saved cases; DPT full and static match native totals. In the nine-case support panel, AD full/reset outcomes are identical and DPT full loses one case to reset. The historical source counts **60/527** and **10/527** describe different, nested recovery events and are reconciled below.

This is ordinary saved CSV/JSON/count analysis on macOS: no tests, model calls, GPU work, robot actions, scheduler queries, new monitors or source expansion. Existing execution owners and holds are unchanged; this audit is not a launch gate.

## Complete fixed100 paired outcomes

One state-0/scene-seed-123 case per LIBERO90 task, using the existing fixed100 models. Each comparison includes **all 90 task/case pairs**. W/L/T means treatment-only success / comparator-only success / equal outcomes. RD is treatment minus comparator in percentage points.

| Treatment vs native K0 | Successes | W / L / T | RD (pp) | Conservative 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| AD full | 29 vs 28 | 3 / 2 / 85 | +1.11 | [−8.19, +10.24] |
| AD static | 29 vs 28 | 4 / 3 / 83 | +1.11 | [−9.43, +11.50] |
| DPT full | 28 vs 28 | 2 / 2 / 86 | 0.00 | [−8.54, +8.54] |
| DPT static | 28 vs 28 | 1 / 1 / 88 | 0.00 | [−6.87, +6.87] |

All five methods agree on 81/90 cases: 24 common successes and 57 common failures. That is outcome agreement, not proof that the motor determines every outcome or that no other controller can succeed. Full versus trained-static controls give AD **3/3/84**, RD 0 [−9.89,+9.89]; DPT **1/1/88**, RD 0 [−6.87,+6.87].

K1 uses the **pre-existing nine tasks 0,10,…,80**, with an actual support followed by a query. Native below is its K0 result on those same nine cases, not a newly collected native K1 arm. K1 policies incur support cost; full/reset is the closer equal-support-cost history comparison.

| Treatment / comparator | Query successes | W / L / T | RD (pp) | Conservative 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: |
| AD full K1 / native | 6 vs 6 | 0 / 0 / 9 | 0.00 | [−38.55,+38.55] |
| AD same-weight reset K1 / native | 6 vs 6 | 0 / 0 / 9 | 0.00 | [−38.55,+38.55] |
| DPT full K1 / native | 5 vs 6 | 0 / 1 / 8 | −11.11 | [−52.99,+38.41] |
| DPT same-weight reset K1 / native | 6 vs 6 | 0 / 0 / 9 | 0.00 | [−38.55,+38.55] |
| AD full K1 / same-weight reset | 6 vs 6 | 0 / 0 / 9 | 0.00 | [−38.55,+38.55] |
| DPT full K1 / same-weight reset | 5 vs 6 | 0 / 1 / 8 | −11.11 | [−52.99,+38.41] |

Task60 is the only DPT full/reset discordance and favors reset. All **12** two-sided exact conditional McNemar p-values are **1.000**; Holm-adjusted p-values are also 1.000 within the four K0/native, six K1 and two secondary full/static audit families. These are not evidence of equivalence.

## Statistical interpretation and repeat uncertainty

RD is `(W−L)/n`, including ties. Sparse discordance makes an unqualified normal approximation unsuitable. We use a deliberately conservative construction: calculate two marginal **97.5% Clopper–Pearson intervals** for win and loss probabilities, then subtract opposite endpoints. By the union bound their joint coverage is at least 95%, hence so is coverage of their difference under the independent, identically distributed pair model. No independence between the win and loss counts is required. Intervals are pointwise, not simultaneous across all comparisons.

The exact paired test conditions on `D=W+L` and uses `min(1,2·P[Bin(D,0.5)≤min(W,L)])`; D=0 gives p=1. These intervals are not a power calculation or an inversion of that test. See [NIST’s paired-test assumptions](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm) and the [matched-pairs risk-difference discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC9447366/); the latter’s tighter melded interval is not implemented here.

This fixed bank is a census of the observed 90 task/cases, not a random sample of training seeds. Task families, shared models and reuse of the nine tasks create dependence. The intervals are conditional sampling-model sensitivity summaries, not guaranteed coverage for unseen tasks, states or training seeds. Query and support episodes are not additional independent tasks.

A separate historical student/noise-wave comparison recorded **4/90 native outcome flips** despite matched initial inputs/noise and differing decoded actions. It is not a repeated fixed100 experiment or a calibrated noise floor for this panel. Current fixed100 cells have one execution each; repeat and training-seed uncertainty remain unresolved. Do not subtract that historical flip rate from effects, condition the primary denominator on favorable cases, or translate nonsignificance into “no advantage larger than X.”

## Source-history reconciliation: resolved

The original saved TRAIN receipt covers **527 K4 cases / 2,108 episodes**, not the later expanded bank.

| Recorded event | Cases / all 527 |
| --- | ---: |
| Any failed episode followed by any later successful episode within the K4 | 60 / 527 (11.39%) |
| Entirely warmup among those recovery cases | 44 / 527 |
| Entirely learned among those recovery cases | 8 / 527 |
| Warmup/learned transition among those recovery cases | 8 / 527 |
| Learned failure followed by a later learned success | 10 / 527 (1.90%) |

The final row is a **subset** of 60: eight entirely learned cases plus two of the eight transition cases. The 60 cases contain 132 ordered failure/success episode pairs; these are not 132 independent trials. We recomputed the 60-case total from all 527 outcome patterns, then checked all 60 saved case IDs, stage sequences and ordered pairs. Stage labels use recorded behavior counters; a warmup episode’s post-episode optimizer clock can already be positive.

There are 454 all-success cases, 94 failed episodes total, and 18 failures among 776 learned-behavior episodes. These facts suggest sparse recovery content under learned behavior. They do not establish policy improvement, a memory effect or that the entire dataset contains no useful learning: 44 recovery cases occur entirely during warmup, and weights/exploration can change between episodes. Warmup and learned success rates concern different chronological cases and different sampling distributions. All 2,108 receipts lack measured post-settle state hashes; nominal hashes are not substituted for actual physical-state evidence.

## Sources and reproducibility

Reviewed remote-main snapshot: **`fd4a7c91207210951e5bbfc3ecb80c61fedd9052`**, fetched into an isolated detached worktree. The main clone’s existing coordination edit was left untouched.

- [Fixed100 case outcomes](../libero40-fixed100-target-plan-20261031-r1/case-results.csv), SHA256 `0072f6de5a1351242e8f31b85b805181882d73a72fa69f7d3460954bdd6a49ad`.
- [Original source-history receipt](../libero40-train-corpus-statistics-20261030-r1/summary.json), SHA256 `befd494cea8860afd786a07e9273456793c09834712faab7ade97a4fb85a6f81`, and its [definitions](../libero40-train-corpus-statistics-20261030-r1/README.md).
- [Machine-readable summary](summary.json): full cell counts, discordant IDs, intervals, tests, exact source hashes/URLs and limitations.
- [Reproducible stdlib analysis](audit_saved_counts.py) reads the pinned Git blobs. On macOS, enable `proxy_on` in the same interactive zsh shell, then run `python3 outputs/steering-headroom-audit-20261031-r1/audit_saved_counts.py`. It refuses changed input hashes and incomplete/nonbinary panels.

The audit supports a focused interface/data screen. It neither proves a universal steering limit nor supplies an improving-history or teacher-competence qualification.
