# Intermediate native checkpoint evaluation

2026-11-04. Owner chat `01a1a077`. User requested intermediate evaluations
and asked whether to select the best checkpoint. **Qualification running on Baige Pro6K as `job-1nfvvw2b93t5`**
(one GPU,45-minute cap,automatic retry off; source`2affc246`). No checkpoint
selected and no full sweep submitted. [Admission](QUALIFICATION-R1-ADMISSION.json),
[submission](QUALIFICATION-R1-SUBMITTED.json), [allocated GPU probe](QUALIFICATION-R1-GPU-PROBE.json)
and [zero-error initial likelihood replay](QUALIFICATION-R1-INITIAL-PARITY.json).

The completed [10M source result](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md)
is already on main. Its update-130 results and failed improvement gates stay
as recorded. This follow-up measures the learning curve and reports a separate
validation-selected policy. It does not restart PPO or admit a teacher for
student fitting.

## Evaluation and selection

- All five ID tasks; retained policies 0, 1, 2, 10, 20, ..., 130.
- Fresh **64-case DEV** per task, identical case order and eight-lane batches
  for every policy. Primary metric: native first success in 600 controls.
  Report success at end and first-success timing too.
- Select the largest DEV success count separately per task, breaking exact
  ties in favour of the earlier checkpoint. Lock model hashes and the selection
  receipt before TEST. Include policy zero in selection so a worse adapted
  policy is not forced into the result.
- Fresh **128-case TEST** per task for policy zero, the DEV winner and 130;
  evaluate duplicate policies once. Report paired wins/losses, exact McNemar,
  paired confidence intervals and both endpoint and selected-policy results.
  Show the full DEV curve, its sample size and the selection procedure.
  Selecting the maximum among many noisy estimates is not itself improvement.
- Never select on the original 64-case before/after panel or on OOD tasks.
  These results cannot retroactively change the original teacher gates.

Case seeds: tasks in campaign order OpenDrawer, Toaster, Drawer-to-Counter,
Coffee, Stove have ordinal 1..5. DEV starts at `8,000,000 + 10,000*ordinal`;
TEST at `9,000,000 + 10,000*ordinal`. Each bank contains the next eight seeds.
Qualification uses only Drawer-to-Counter seeds 10,030,000..10,030,007,
disjoint from both panels and all source cases.

Execution reuses clean source runtime `0988b5f9e9bf31ac3db673cc2860959611b9ae12`
and RLinf `8a1233c3fae6f5905adfda7684c455842b0ecfb8`, with its native
environment wrapper, three cameras, state/action transforms, horizon 600,
H50 commands, ten-step evaluation ODE and inference precision. Eight EVAL
lanes on one allocated GPU serialize logical banks; fixed batch membership
and common sampler RNG are required. Bitwise deterministic kernels are not.
No actor, optimizer, backward pass, TRAIN collection or source-file mutation.
Strict full-model loads and saved initial-policy likelihood parity precede
native episodes. Archived weights are restored to private PFS copies with
SHA256 verification; source checkpoints/corpora are read only. PFS floor200GiB.

## Bounded qualification before sizing the sweep

One GPU, at most **45 minutes / 0.75 GPU-hours**, automatic retry off. Evaluate
Drawer-to-Counter at policy0, policy130, then policy0 again, on the same eight
qualification cases (24 episodes / 14,400 policy controls). Record native
reset controls separately; preserve unresolved intents on failure. Keep the
original wrapper's initial physical/public snapshots, full chunk records and
episode success receipts. Require all canonical physical/public checks, exact
lane/case identities, twelve H50 chunks per episode, finite inference and
unchanged shared sampler RNG states. Report repeat action/success differences;
do not claim bitwise equality. Diagnose any discrepancy before a sweep.

The only runtime extension is eval-only worker orchestration and checkpoint/
case-bank switching, recorded in this folder. It must pass allocated Linux
import/config checks and a real simulator qualification before promotion.

The proposed full panel is at most **7,040 episodes / 4,224,000 controls**
(5*16*64 DEV plus at most5*3*128 TEST). The qualification is a separate
throughput/paired-reset check, not an initial tranche of an unapproved sweep.
Use its measured time, model-restore overhead and all-attempt charge to give
one concrete full-sweep budget. If the proposed budget is8GPU-hours or more,
obtain the approval required by AGENTS.md before submitting that sweep.

## Optimization diagnosis

First establish whether any source actually peaks and then falls on DEV.
The original before/final64-case panels alone do not establish a collapse.
Read both actor ranks' KL/clip/value metrics, respecting their native
chunk-level reductions, and join them to the learning curve. Finite losses,
zero unchanged-policy likelihood gaps and frozen-VLM checks already passed;
that qualification does not establish useful optimization. Large logged
gradient norms are before clipping; entropy-loss zero is a configured zero
coefficient. Neither alone demonstrates a bug.

The separate [per-dimension diagnostic](../robocasa-flow-sde-dimension-drift-20261104-r1/PLAN.md)
belongs to chat `01a1a4a2`; do not take its job or modify its files. Conditional
saved-transition likelihood drift is not full-action KL or robot performance.
Any hyperparameter ablation is a subsequent scientific decision after these
measurements; no training recipe change is authorized here.
