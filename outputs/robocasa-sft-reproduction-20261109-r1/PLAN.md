# Official Human300 π0.5 baseline reproduction

**Closed at the atomic-first project boundary.** The complete atomic result
is independently verified. The composite continuation was intentionally
closed after the user clarified the current scope; no new training or
composite sweep is pending. See [closeout](CLOSEOUT.md). The original
prospective plan below is retained as provenance.

Authorized by the user on 2026-11-09: try reproducing the baseline, investigate
RoboCasa 1.0.1 changes, and use 1.0.0 if required. Owner: this chat
`01a1be92-5e00-7df0-b90e-9719078be031`. Evaluation only, zero training.

## Reference and source audit

- Original checkpoint: `robocasa/robocasa365_checkpoints@c484448aba1a9b60a04c9b0ca117241518ea69f3`,
  `pi05_pretrain_human300/multitask_learning/75000`, preserved JAX parameters
  and bundled native normalization. Use native OpenPI first, independently of
  the existing converted RLinf inference path.
- Published submission: Atomic-Seen 39.6%, Composite-Seen 7.1%,
  Composite-Unseen 1.2% (weighted overall 16.912%); RoboCasa version 1.0.0,
  OpenPI code reference `ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`.
  [Submission](https://github.com/robocasa-benchmark/leaderboard/blob/main/submissions/standard/pi05_2026-04-02.json).
- That OpenPI evaluator defaults to replan every **5 controls**, 224-pixel
  resize-with-padding, all three cameras, seed 7, 50 trials per task,
  `split=pretrain`, first-success termination, native action conversion.
  Its config is `pi05_pretrain_human300`, prediction horizon 50, 10 ODE steps.
  Its CLI has a `task_set`/`task_soup` typo; use an explicit task list rather
  than silently changing the scientific settings.
- The linked code multiplies the 1.0.0 registry horizon by 1.5. RoboCasa
  `29f7ce8` moves that multiplier into the registry; OpenPI `b175bb9` removes
  it from the evaluator. Never multiply a 1.0.1 registry by 1.5 again.
  Preserve both old-registry and extended-horizon success times so the
  original metadata ambiguity is visible.

## Stages and decisions

1. Static source audit: exact version-commit diff, complete task/horizon
   equality, assets, action conversion, controller, reset and success logic.
   Separate the narrow 1.0.1 release from subsequent commits still labeled
   1.0.1. Record all source pins and mechanical compatibility edits.
2. On allocated Linux resources, qualify imports and the actual GPU. Load
   native original parameters, verify unchanged model identity, then run real
   pretrain-split episodes. Record inference time and controls/s before scaling.
3. Run the official 18 Atomic-Seen tasks × 50 trials with native H5 replanning.
   This is an aggregate baseline reproduction, not another task-selection
   screen. Report every task, both original and extended horizons, reset errors,
   actual executed controls and incomplete cases. Do not drop failures.
4. Complete the remaining 16+16 composite tasks × 50 trials when qualified
   throughput fits the remaining budget. Compare the three reported aggregates
   and their uncertainty with the published references; no paired comparison is
   possible without original per-case data. A ±5 percentage-point aggregate
   band is a prospective practical screen, not a formal equivalence claim.
5. If the native 1.0.1 result disagrees materially, use a fresh detached 1.0.0
   source on the same cases, policy and effective horizons, preserving both
   results. Do not downgrade shared environments or existing runs in place.
   If the source audit proves matching runtime code and horizons, investigate
   policy/precision/assets before attributing any discrepancy to the version.

## Resources, retention and monitoring

Baige scheduler allocations only for model computation. Discover current
queue capacity and actual GPU type before launch. Run the repository GPU
probe inside each allocation, bind rendering to its GPU, keep retry disabled,
and charge startup, failed attempts and release. This task has a cumulative
**50 GPU-hour maximum** without another budget request; start with a bounded
qualification and choose the full sweep allocation from measured throughput.
No PPO, new SFT, reward-model scoring or downstream policy migration.

Use the protected existing JAX checkpoint and native asset cache read-only.
Check storage, quota and a write probe; retain compact per-episode records,
source/hardware receipts, scalar summaries and selected diagnostic videos.
Never modify an active run or move shared caches. Deploy this repository's
verified commit through Git into a fresh detached source checkout. Report
job ID, progress, last check and next action, and inspect running work at
least every 20 minutes through completion or a concrete blocker.
