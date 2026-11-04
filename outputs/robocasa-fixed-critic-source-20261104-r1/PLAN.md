# Two replacement-task PPO restarts, conditional on three existing teachers

2026-11-04. The human narrowed the initial five-task restart proposal:
restart only Counter-to-Cabinet and Faucet first, provided the existing
OpenDrawer, Toaster and Coffee sources each independently demonstrate
statistically significant learning. Reuse those three qualified sources
rather than rerunning them automatically. The two fresh runs use the same
PPO recipe with the fixed critic and evaluation at every saved checkpoint.
The other-three evidence gate, executable integration, qualified code pin
and a new compute reservation remain to be completed before launch. No job
has been launched from this run identity.

## Source tasks

| Task | Role in the revised source set |
| --- | --- |
| OpenDrawer | Retain existing source only if its independent learning gate passes; no restart now |
| CloseToasterOvenDoor | Retain existing source only if its independent learning gate passes; no restart now |
| CoffeeSetupMug | Retain existing source only if its independent learning gate passes; no restart now |
| PickPlaceCounterToCabinet | Fresh fixed-critic PPO; replaces PickPlaceDrawerToCounter |
| TurnOnSinkFaucet | Fresh fixed-critic PPO; replaces TurnOffStove |

Counter-to-Cabinet and Drawer-to-Counter are both atomic pick-and-place
tasks. The former moves an object from the counter into an already-open
cabinet; the latter moves an object from an already-open drawer onto the
counter. This preserves the coarse skill category, without assuming equal
difficulty or the same PPO learning behavior. Native definitions are in the
[pinned RoboCasa source](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/atomic/kitchen_pick_place.py).

Counter-to-Cabinet's two previous 600-control banks scored 24/50 and 28/50
([audited panels](../robocasa-joint-h600-preparation-20261102-r4/RESULT.md));
Faucet's replacement qualification scored 18/50
([result](../robocasa-buttons-replacement-qualification-20261104-r1/RESULT.md)).
These motivate selection; each restarted run measures its own policy-zero
baseline on the qualified runtime. Counter-to-Cabinet had older steering-PPO
exposure and is explicitly a source task here.

## Verify the existing three before admitting this two-run route

Use the existing owner's
[checkpoint-selection sweep](../robocasa-flow-sde-checkpoint-selection-20261104-r1/PLAN.md)
and its audited closeout. For each of OpenDrawer, CloseToasterOvenDoor and
CoffeeSetupMug, require a DEV-selected checkpoint locked before fresh TEST,
paired TEST wins greater than losses, and exact two-sided McNemar
Holm-corrected p<.05. Keep the sweep's original five-comparison correction;
do not retrospectively reduce its family to these three. A DEV peak, higher
TRAIN rate, or significance on a different endpoint is not this test.

Retain the full curve and the chronological policy-version/corpus audit so
the admitted source represents an actual learning process. Preserve the
entire historical corpus; bind the selected teacher and any admitted history
prefix explicitly, with original update order and checkpoint hashes. No
monotonic increase at every adjacent checkpoint is assumed.

All three must pass individually to use this three-old-plus-two-new route.
If one fails or its evidence is incomplete, report that task separately and
revisit its source decision; do not launch its restart or silently expand
back to five. The existing 512-case policy0-versus130 tests are a separate
preregistered endpoint analysis, not a substitute selected-checkpoint test
or a reason to try multiple tests until one passes.

At this revision, the saved sweep is still in DEV and no selected TEST
result is verified. See [evidence check](EXISTING-THREE-CHECK.json).
Existing evaluations continue under their owners and budgets. The old three
used the historical critic implementation; the new two use the correction.
Record that source-generator distinction in the combined corpus rather than
claiming all five were collected with the fixed critic.

## Fresh training on Counter-to-Cabinet and Faucet only

Use the same Human-300 SFT motor, transforms, three cameras and native RLinf
Flow-SDE PPO process as the
[previous source campaign](../robocasa-flow-sde-10m-per-task-20261103-r1/PLAN.md).
Each of these two tasks starts from that original SFT model, with a fresh value head,
optimizer, sampler and RNG initialization. Do not resume an old trained PPO
checkpoint or mix old histories into the restarted corpus.

Keep the scientific recipe: frozen VLM, trainable action expert and value
head; simulator first-success reward; horizon600/H50; ten flow steps;
Flow-SDE noise0.3; 128 episodes per iteration; global minibatch256;
four PPO epochs; actor/value learning rates5e-6/1e-4; gamma0.99/lambda0.95;
KL penalty and KL stopping off, with KL still logged; 130 iterations and
9,984,000 TRAIN controls per task. Preserve the qualified two-actor-rank
advantage normalization and eight-GPU-per-task execution unless a separately
qualified execution change preserves these quantities.

The critic correction makes rollout values use the same selected denoising
index as the value prediction recomputed for PPO. Require unchanged-weight
value and likelihood parity on the actual allocated runtime, with action,
sampling and RNG behavior preserved. The existing
[CPU qualification](../robocasa-flow-sde-critic-parity-20261104-r1/CORE-QUALIFICATION.json)
and [candidate check](../robocasa-flow-sde-critic-parity-20261104-r1/QUALIFIED-FIX.json)
are supporting evidence; the production replay's independent audit was still
pending in its latest saved
[execution record](../robocasa-flow-sde-critic-parity-20261104-r1/EXECUTION.json).
This restart tests the corrected process; it does not establish that the
critic defect caused the historical failures.

## Evaluate during collection

- Save and evaluate policy0, then every five PPO iterations through130,
  following the prospective cadence in docs/FLORA-PPO.md. If qualification
  or resume also retains steps1/2, evaluate those checkpoints too.
- Use one fixed, disjoint DEV64 case bank per task at every boundary,
  horizon600 and the qualified ten-step evaluation ODE. Pair actual initial
  physical/public scenes and sampler RNG across checkpoints; retain image
  diagnostics and first-success/end-success outcomes separately.
- Bind each completed evaluation to its exact model hash and publish the
  curve as training advances. A checkpoint is not treated as evaluated
  merely because its weights were saved. Keep evaluation RNG isolated from
  the training sequence and verify full-state resume behavior.
- Select by DEV first-success count, ties to the earliest checkpoint including
  policy0. Lock the model hash before a separate fresh paired TEST128 against
  policy0; report exact McNemar and Holm correction across the two new tasks.
  This is a separate prospective family from the existing five-task sweep.
  TEST cannot drive training, checkpoint selection or task replacement.

With steps0/1/2 and5,10,...,130, there are29 DEV panels per task:
1,856 episodes /1,113,600 controls. TRAIN plus those panels is11,097,600
controls; TEST128 for initial and selected adds at most153,600, for
11,251,200 policy controls per task. At128 reserved reset controls per episode,
the corresponding nominal policy-plus-reset reservation is13,651,456 per
task, before retries. The historical12,210,000 ceiling and before/after-only
arithmetic must not be copied unchanged. The two-run nominal totals are
22,502,400 policy controls and27,302,912 policy-plus-reset reserved controls.

## Implementation and launch preparation

Before launch, add Faucet to the append-only task/seed registry; create a
separate resolved config; derive the full evaluation schedule and accounting
from that config; integrate checkpoint-bound evaluation and curve receipts
into the runner; qualify the fixed critic, periodic evaluation and full-state
resume on allocated Linux; then pin the verified main commit in a new
execution record. Keep old configs, contracts and active worktrees immutable.

Measure Q1 time/storage and reserve training, all checkpoint evaluations,
TEST, staging and release overhead together. The predecessor cost373.934
GPU-hours across all attempts; that is context, not a new allowance. The two
full restarts exceed the repository's8GPU-hour approval threshold and
require an explicit new compute budget before submission. Existing critic,
checkpoint-sweep and endpoint-test budgets belong to those separate runs.

Retain the full chronological action-supervised corpus, raw RGB, policy
versions, resumable checkpoints and numerical joins. Require independently
verified teacher improvement before fitting students. The proposed15-task
meta-learning evaluation (five Atomic-Seen, five Composite-Seen and five
Composite-Unseen) is downstream; its exact memberships remain to be fixed
and all must exclude these five source tasks.
