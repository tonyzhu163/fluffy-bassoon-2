# RoboTwin behavior coverage and source-quality reporting
Current continuation: [launch verification](LAUNCH.md) records fresh r2collection. The prior32-episode table below covers r4only; do not treat it as a live count or as evidence about later policy versions.


2026-10-30 Asia/Shanghai. Documentation only; no allocation, new robot action,
data exclusion or admission gate is introduced. The original1,024D arm,
<=20%clip-rate qualification rule and pending two-hour request are unchanged.

## What currently exists

All32source episodes from the bounded r4 run occurred **before any optimizer
update**. The later saved-data PPO diagnostics updated independent forks but
executed no subsequent robot actions. They do not add learning histories or
post-SGD robot behavior to this source corpus.

| r4 source controller | Episodes | Decisions | Executed controls | Actual behavior | Post-SGD robot decisions |
| --- | ---: | ---: | ---: | --- | ---: |
| Persistent PPO |12|140|4,273|initial, zero-update policy|0|
| SAC |12|139|4,274|bounded-uniform warmup|0|
| Trained-reset PPO |8|95|2,928|initial, zero-update policy|0|
| **Total** |**32**|**374**|**11,475**|all before SGD|**0**|

The1,551camera-qualification controls are separate and receive no source
learning credit. All13,026prior physical controls remain charged once under
the shared1M benchmark cap. Raw r4 RGB was not saved; it cannot be reconstructed
from features/hashes. No improving teacher, learned recovery or meta-learning
performance is established by these artifacts.

## Reports required from future authorized collection

Report the following by controller, task, partition, stage and interface. These
are descriptive audit fields, not filters for keeping successes or excluding
cases. Preserve all outcomes and the declared six-source design.

- Count episodes/chunks/actions in explicit warmup, policy behavior before its
  first SGD, and policy behavior after SGD. Use the **pre-action** optimizer
  clock and actual behavior checkpoint/hash. An update after an episode does
  not turn that episode into post-update behavior. In SAC, the first actor
  action after uniform warmup may still use update0 weights, so non-warmup is
  not sufficient evidence of trained behavior. Episodes crossing update onset
  should retain their per-decision clocks rather than receive an ambiguous tag.
- Tabulate success/failure and same-case subsequent success separately before
  and after SGD. A success following a warmup failure before any optimizer
  step is not evidence of learned recovery. Do not infer causal improvement
  from changing reset cases, random-action draws or an unpaired success rate.
- Identify every checkpoint/actor that actually generated behavior, including
  the final saved policy. Separate source-train, source-validation and target
  execution coverage. A final actor used only for label inference or whose
  update occurs after the final source rollout has no subsequent source behavior.
  Future frozen source/target evaluations may execute it, but must not be
  relabeled as source-training data. File existence alone does not qualify an
  actor as an expert, improving teacher or successful robot policy.
- Keep requested seeds, available effective reset identity, initial observation
  hashes and policy RNG receipts distinct from counts of independent physical
  cases. Report paired cases and RNG conventions when comparing source stages.
  The fixed source-validation seed grid does not by itself establish identical
  stochastic action draws across checkpoints. No extra baseline episodes are
  added by this note; any new paired intervention must be declared and charged.
- Audit raw executed noise/action/reward/done/duration against exported labels,
  policy versions and full recovery. Count unique trajectories/decisions before
  windowing; overlapping windows are training presentations, not new data. Keep
  selected source-bank cost, unselected qualification cost and target-evaluation
  cost separately visible while retaining all physical spending in the1M cap.

For direct RL²/PPO, distinguish within-trial memory adaptation from outer SGD
between rollout batches. SAC's replay may influence a teacher beyond a bounded
visible history; do not claim that a context window exactly reconstructs that
learning process. These observations do not invalidate data or methods by
construction, and they do not create a new collection or approval requirement.

The future collector already records behavior versions, per-decision update
metadata, phase/task/trial identity and raw transitions. Quality summaries must
be derived from those actual executed records after collection, not estimated
from planned update counts. A source checkpoint's post-update physical behavior
coverage remains unknown until it actually acts.

## Related LIBERO evidence

The separately owned selected-bank audit reported intact saved data but no
post-update training failures/recoveries;65.2%of its larger training bank was
warmup, and final actor updates had no subsequent source-history rollout. That
is a warning against labeling an update clock as demonstrated improvement,
not a reason to exclude tasks or claim RoboTwin shares LIBERO's outcomes.
See the other owner's `outputs/libero-source-bank-quality-20261030-r1/REPORT.md`.
The benchmarks' full1M collection requests remain distinct from their small
current pilot banks; RoboTwin still has six intended source tasks, with only
three visited in the existing bounded r4 source run.
