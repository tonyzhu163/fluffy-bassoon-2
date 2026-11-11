# Frozen feedback diagnosis

The three existing attempts checkpoints beneficially depend on feedback in this
synthetic task. Changing the history source explains only a small part of their
shortfall against exact rule enumeration. All six original evaluation curves
(three seeds × two splits) reproduced exactly before the fresh comparisons.

## Fresh held-out panels

Query accuracy at six attempts, mean across three fits. Each fit uses 8,192
sessions per split. Within a history source, all interventions retain identical
scenes, choices and query targets. Demo-only uses the same checkpoint.

| History source | Real outcomes | Outcomes masked | Outcomes shuffled | Demo only | Full-prior reference, same history |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learner's own greedy choices | 57.38% | 46.58% | 43.53% | 44.93% | 86.38% |
| Original training actor | 59.15% | 47.16% | 43.49% | 44.93% | 84.49% |
| Independent random choices | 54.65% | 43.94% | 38.83% | 44.93% | 72.26% |

The predeclared primary panel is held-out random choices: its choices cannot
indirectly reveal earlier rewards. Real outcomes exceed shuffled outcomes by
15.82 percentage points on average and masked outcomes by 10.71 points. Both
effects have the same positive sign in every seed. With three seeds, these are
replications across three fits, not a population-level confidence interval over
training runs.

| Seed | Real − masked, pp (paired 95% interval) | Real − shuffled, pp (paired 95% interval) |
| --- | ---: | ---: |
| 0 | 12.33 [11.15, 13.51] | 20.00 [18.74, 21.25] |
| 1 | 9.52 [8.39, 10.65] | 12.65 [11.47, 13.82] |
| 2 | 10.28 [9.17, 11.39] | 14.82 [13.65, 15.99] |

Intervals use the standard error of paired per-session correctness differences
within each fit. Raw gains/losses, all prefix lengths and negative log likelihoods
are retained in `eval01/seed-*/result.json`.

## What explains the remaining gap?

Using the training actor raises held-out accuracy only 1.77 points relative to
own histories. Even on those training-style histories, the model remains 25.34
points below the full-prior reference. Training rules show the same problem:
60.19% versus 85.21% on training-actor histories. Thus a history-policy mismatch
exists, but does not account for most of the observed deficit.

Restricting training-rule evaluation to two-attribute rules gives 62.72% with
training-actor histories, compared with 59.15% for the exclusively two-attribute
held-out rules. This 3.57-point generalization gap is much smaller than the
within-training-rule reference gap (22.46 points on that subset). The current
results favor investigating fitting, supervision and representation before
making held-out rule memorization the main diagnosis.

Giving the enumeration reader the evaluator's exact split prior raises its
held-out reference to 95.32% on own histories and 93.93% on training-actor
histories. That privileged prior is not a fair learned baseline; it also does
not explain away the original full-prior shortfall.

## Next step and limits

A bounded toy comparison of longer optimization versus denser supervision is
now justified. Use fresh development panels for that choice and a separate
confirmation panel. These diagnostic panels have now been inspected and should
not become an unacknowledged confirmation set. A substantial LIBERO-Gen campaign
remains premature; no robot run was launched.

These interventions establish beneficial input dependence for frozen query
predictions. They do not measure alternate closed-loop trajectories, learned
information-seeking exploration, or robotic adaptation. Masking/shuffling may
produce inputs outside the training support; shuffled outcomes can create
inconsistent histories. Full-prior enumeration remains a reference rather than
the split-specific optimum, and the split-aware reader uses privileged prior
knowledge. See the [locked protocol](PLAN.md).

## Execution

- Baige job `job-guy3jxur9ic0`, three distinct RTX 4090 GPUs.
- Evaluation commit `6974ae82ec9125da38771691a87a9828583ff2e0`; checkpoint training
  commit `50508c86ed24902b500dad6ae314a597df2b9d59`.
- All probes passed; all three workers returned zero; all original train and
  held-out curves replayed exactly; masked encoding equalled blind encoding;
  all zero-attempt interventions matched; shuffled reward counts were preserved
  at every attempt index.
- Scheduler succeeded November 11, 2026 at 14:35:06 Shanghai, duration 2m08s.
  This follow-up used 0.107 scheduler GPU-hours (0.148 submission-to-end bound).
  Cumulative exploration plus follow-up: 7.530 scheduler GPU-hours, conservative
  bound 7.768. The follow-up's allocation cap was 2.25 GPU-hours.
- The 18 NPZ trace files remain under
  `/mowenjie/fluffy-bassoon-2/outputs/learning-histories-feedback-20261111-r1/eval01/`.
  Their checksums are in `TRACE-SHA256.json`; source checkpoint checksums are in
  each result. Small metrics and receipts are mirrored here.
