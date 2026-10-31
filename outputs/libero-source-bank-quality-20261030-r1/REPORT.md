# Selected source-bank quality, 2026-10-30

**No saved-data integrity blocker found. The teachers have not demonstrated
learned recovery or reliable improvement.** This is a read-only inspection of
the pinned one-goal (`a52a2317…`) and two-goal (`f1577bcb…`) banks. No model was
constructed or evaluated; CPU checkpoint loading extracted saved metadata/RNG
only. Full paths, hashes, counts and reproduction script: `receipt.json` and
`inspect_saved.py`. This adds no admission gate or source/target exclusion.

## Actual behavior and labels

“Post-update” below means the **executed episode's pre-action teacher clock
was positive**, not merely that an update occurred after that episode.

| Source goal / partition | Episodes: success/failure | Pre-update episodes; chunks | Post-update episodes; chunks | Cases / actual nominal IDs |
| --- | ---: | ---: | ---: | ---: |
| Spatial00 train | 80: 79/1 | 54: 53/1; 505 | 26: 26/0; 237 | 20 / 20 |
| LIBERO10:00 train | 28: 27/1 | 17: 16/1; 507 | 11: 11/0; 302 | 7 / 7 |
| Spatial00 validation | 68: 68/0 | 54: 54/0; 505 | 14: 14/0; 112 | 17 / 10 |
| LIBERO10:00 validation | 36: 32/4 | 16: 12/4; 507 | 20: 20/0; 540 | 9 / 9 |

- Source1: **742 train +617 validation labels per objective**; one trial each.
- Source2: **1,551 train +1,664 validation labels per objective**; two trials
  each, preserving the original Spatial arrays exactly. Of training rows,
  1,012/1,551 (65.2%) are Gaussian warmup; 539 are post-update behavior.
- All3,215 selected raw chunks were inspected. Features, actual executed
  noise, rewards, done boundaries and teacher clocks matched AD arrays;
  AD/DPT histories matched in every field except target labels. Assembled
  arrays matched the original whole histories, with no overlapping-prefix
  multiplication. Dataset, raw-trace, actor and recovery hashes matched.
- AD targets equal actual executed noise. Gaussian-warmup targets intentionally
  exceed[-1,1]; all505/507 warmup rows have an out-of-box coordinate. This is
  the recorded source recipe, not a corrupted clip. DPT labels are finite and
  bounded; publishing code uses the final actor's deterministic `eval_actions`.
  Labels were not re-inferred here. Train DPT mean row norms:0.562 Spatial,
  0.939 LIBERO10. These are candidate labels, not verified expert targets.

The only training failures were Spatial case7/episode1 and LIBERO10 case2/
episode1. Their next same-case successes also had teacher clock0. There are
**zero post-update training failures/recoveries**. All71 post-update episodes
across training and validation succeeded. LIBERO10 validation's apparent
12/16→20/20 rise compares states40..43 against44..48; it is not a matched
teacher-improvement experiment. Crossing-onset training cases were already
successful. Near-ceiling success and changing cases prevent a causal gain claim.

## Versions, K4 cases and observable learning context

Each physical case has four complete same-state/seed episodes, but **the SAC
learner/replay and stored AD/DPT trial persist across cases**. The stored
training trial lengths are `[742]` versus `[809,742]`; no recurrent trial reset
occurs at K4 case boundaries. All train states are disjoint from source-local
validation states40..49. A repeated seed case is not counted as a new nominal
state; Spatial validation has17 cases but10 actual nominal vectors.

Spatial training contains27 actual behavior versions (0 plus26 learned), ending
at clock4,760; its final saved actor/DPT teacher is update4,920. LIBERO10 has12
versions (0 plus11 learned), ending6,060; final actor6,620. The final update
therefore has **no subsequent physical behavior in these source histories**.
Validation likewise ends at2,240/10,860 before final2,400/11,400. File existence
and successful earlier versions do not qualify the final actors as experts.

Saved500-update contracts use batch8, random windows≤128 tokens, recurrent
memory300. Reconstructing all4,000 windows from the saved sampler seed matched
the final sampler state in all eight AD/DPT full/static recoveries.

| Observed post-update supervision in full-history windows | Source1 | Source2 |
| --- | ---: | ---: |
| Sampled post-update label positions | 64,587 | 76,470 |
| Latest completed episode's terminal feedback visible | 93.7% | 87.8% |
| Entire immediately preceding episode visible | 80.7% | 64.5% |
| Exact one-support query positions with their full support visible | 80.9% | 63.0% |
| Entire replay-dependent prior learning run visible | 0% | 0% |

Windows begin with fresh recurrent state; the first token still carries the
actual previous noise/reward/done. Earlier context is omitted. Teacher updates
sample persistent replay and inherit earlier optimizer state, so these windows
do not expose every transition responsible for the current actor. This makes
the task bounded-context approximation to learning, not exact reconstruction
of SAC; it does not make AD invalid by definition. Static controls deliberately
erase feedback. Full windows cross physical cases in3,109/4,000 Source1 versus
2,536/4,000 Source2 samples. Source2 samples1,991 Spatial and2,009 LIBERO10
windows, so the fixed-compute comparison also reduces Spatial repetition.

## Costs and scaling interpretation

Selected Source1 costs7,892 train +6,538 validation =**14,430 actions**.
Selected Source2 costs16,167 train +17,252 validation =**33,419 unique actions**;
Source1 is reused within that total, not collected again.

The original **unselected** Spatial training qualification is a different
60-episode/606-chunk run costing**6,406 actions**. Its arrays are in neither
training bank. Including that spent source qualification gives**39,825 unique
source-generation actions** for the inspected campaign portion; the shared
selected validation run is counted once. Target robot experiments and other
campaign expenses are separate.

Continue interpreting the bounded robot comparison against this measured
teacher/data quality. Scaling these histories alone does not establish
failure-driven adaptation; neither a positive nor negative robot result should
be presented as a test of an already-qualified improving/expert teacher.
