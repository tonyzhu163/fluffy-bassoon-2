# Intermediate native checkpoint evaluation

2026-11-04. Owner chat `01a1a077`. User requested intermediate evaluations
and asked whether to select the best checkpoint. **Qualification complete;
full sweep was approved at22GPU-hours including qualification and all attempts;
**stopped at the human's request on Nov4,15:41 Shanghai.**
All80 DEV panels completed; TEST stopped at209/240 policy banks.
Allfive allocations are terminal and the monitor is deleted. See [stop closeout](USER-STOP.md).**
See [DEV selections and TEST-start audit](DEV-SELECTIONS-R1.md). Baige Pro6K
`job-1nfvvw2b93t5` succeeded and released its one GPU at12:02:28 Shanghai;
source`2affc246`. Selected updates are30/1/80/20/30 in campaign order.
These are DEV selections; independent improvement is not yet established.
The sweep is stopped; see [terminal jobs and cost](EXECUTION.json).
[Qualification result and limitations](RESULT.md),
[audit](QUALIFICATION-R1-AUDIT.json), [adopted sweep manifest](SWEEP-PREPARED-R2.json), [budget approval](BUDGET-APPROVAL.json),
[admission](QUALIFICATION-R1-ADMISSION.json),
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
  Report success at end and first-success timing too. The manifest fixes an
  human-requested back-to-front checkpoint order130,120,…,10,2,1,0 before
  seeing DEV. The earlier randomized-order proposal remains in the R1 manifest.
- Select the largest DEV success count separately per task, breaking exact
  ties in favour of the earlier checkpoint. Lock model hashes and the selection
  receipt before TEST. Include policy zero in selection so a worse adapted
  policy is not forced into the result.
- Fresh **128-case TEST** per task for policy zero, the DEV winner and 130;
  evaluate duplicate policies once. Report paired wins/losses, exact McNemar,
  paired confidence intervals and both endpoint and selected-policy results.
  Show the full DEV curve, its sample size and the selection procedure.
  Selecting the maximum among many noisy estimates is not itself improvement.
  Counterbalance policy order within each eight-case TEST bank by cycling all
  unique-policy permutations; with two models alternate AB/BA. Apply Holm
  correction across the five primary selected-versus-initial comparisons;
  report paired95% case-bootstrap intervals as descriptive, not simultaneous.
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

Qualification measured70.63–78.35 controls/s including native resets. At its
slowest rate, nominal full-panel simulation/inference is16.61GPU-hours before
startup, weight restoration and counterbalanced TEST reloads. Proposed ceiling:
**22GPU-hours across this follow-up, including qualification and any retries**,
at most five one-GPU jobs in parallel, maximum250minutes per job, with the
cumulative ledger taking precedence. The first qualification is conservatively
charged349GPU-s /0.096944GPUh, from scheduler creation to success; the process
clock records304GPU-s. The human approved this cap and then specified back-to-front checkpoint order.
The estimate is a forecast; cumulative charged GPU time remains the hard limit.

Qualification confirmed exact initial physical/public states and all three
images, identical first H50 actions, fixedB8 calls and shared sampler RNG. The
first observed repeat difference was three RGB components, each one intensity
level, in the right camera after50controls; later actions/trajectories differed
and identical-policy success changed6/8→8/8. Keep this rendering/trajectory
variability explicit; neither exact trajectory reproduction nor bitwise GPU
kernels is required. No qualification success rate is used for selection or
as a performance claim. The larger fresh panels and separate, counterbalanced
TEST protect the selection readout from this small-panel noise.

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

## Closeout

After all five result files and exit-zero clocks exist, run `CLOSEOUT.py` on
Linux CPU with this run as `--root`, its live `EXECUTION.json` as `--execution`
and a fresh `results-r1` directory as `--output`. It rejoins every bank and
episode, rechecks selection and paired statistics, applies Holm across the
five primary tests, exports the80-point DEV table, and hashes all retained
raw records. Deploy the postprocessor through Git to a fresh tools/readout
location; keep the active `f30a2a4f` GPU/staging checkout unchanged. Confirm
scheduler release and charge every attempt before declaring final closeout.
