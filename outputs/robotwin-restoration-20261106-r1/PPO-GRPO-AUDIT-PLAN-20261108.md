# RoboTwin learner restoration and effectiveness gate

The current result is **no verified effective PPO or GRPO**. Continue source-task
learning controls before student/ICRL experiments. The user approved a cumulative
100 GPU-hour cap; historical costs remain in `LATEST-BUDGET.json`.

## Verified progress

Both corrected two-collection controls and independent full-state/numerical
readbacks passed. Full paired DEV readout also passed, including actual noise
and physical traces: PPO6/64→10/64 (six gains,two losses,p=.2891),
GRPO6/64→11/64 (five gains,no losses,p=.0625). Neither meets the source gate.
The [precommitted selection](MOTUS-SOURCE-SELECTION.json) chooses GRPO for
disjoint confirmation before any confirmation outcome. Preserve checkpoint2;
student work remains paused while source effectiveness is unresolved.

## Evidence that changes the next action

- Actual eight-rank GAE, episode routing, masks and reward coverage pass independent
  CPU checks. Positive reward is present; absence of successful episodes is not
  the explanation. Raw group8 GRPO arithmetic also passes, with 73/128 tied groups.
- The unchanged-policy likelihood experiment finds actorB32 exact on all 5,958
  valid decisions. B256 changes joint ratios to 0.520–1.883. 11.279% lie outside
  the clipping interval before SGD. The actor recomputation replaced the original
  reference and hid that discrepancy. Retire B256 learning controls.
- The old continuation completed its second update but failed the strict optimizer
  layout guard before checkpoint2 and DEV2. A read-only DCP probe did not reproduce
  that failure. The corrected B32 actual-data update subsequently completed all15
  native Adam steps/rank and save/restore. Independently read every owned Adam
  tensor descriptor and cold checkpoint file before launching new training.
- The published RLinf result uses task-specific SFT on `adjust_bottle`; our Motus
  joint initializer, phone task and H32 cadence differ. One fresh cycle with a
  16-case endpoint panel does not establish convergence or effectiveness.

## Ordered work

1. Close the B32 qualification: independently check all eight original behavior
   errors are zero, all24 post-update/save/restore layouts, all owned moments and
   group options, all checkpoint hashes, scheduler release and cumulative cost.
   Keep production optimizer validation strict unless an actual failing layout
   proves a narrower representation exception necessary.
2. Launch `MOTUSBEHAVIORPPO01-CONTRACT.json` after Linux CPU admission. Start fresh
   from Motus SFT. Restore RLinf PPO group1: 1,024 episodes/collection over its
   1,000 phone TRAIN scenes, four256-env rounds. Micro32/accum8/world8 preserves
   global2048/five epochs. Require exact original behavior likelihoods at every
   collection before updating. Two fresh collections produce checkpoints0/1/2.
3. Inspect advancing scene, collection and update receipts; reject stale-policy
   data or likelihood mismatches. Independently cold-read checkpoints and restore
   the selected full boundary in a separate process. Record source success,
   optimizer counts, KL, clipping, advantages, critic fit, memory and timing.
4. Run a separately named fresh B32 GRPO group8 control with the same Motus motor,
   native binary reward and optimizer/global batch. Preserve tied groups and
   native standardization. PPOgroup1 versus GRPOgroup8 compares complete recipes;
   it cannot isolate the advantage estimator at identical scene diversity.
   It may run concurrently in a separate exclusive allocation after counting
   both complete startup/cleanup reservations against the100h cap.
5. Evaluate SFT and the preselected iteration2 checkpoint on 64 fixed source DEV
   scenes. Verify actual per-case/chunk initial Gaussian draws, native batch2,
   ordering, render settings, horizon and controller match across policies.
   Preserve trace hashes, paired successes, gains/losses and exact paired test.
   Use disjoint64 confirmation scenes once after DEV selection; leave22 unused.
   The training-loop16-scene panels are progress diagnostics. Do not select a
   method or claim gain from those counts alone.
6. A source effectiveness claim requires at least10 percentage points paired
   improvement and a two-sided exact paired test p<.05 on confirmation. One
   training seed supports only a preliminary source result. Repeat training
   seeds before claiming stability; no held-out task becomes TRAIN.
7. If execution passes but gain is absent, report a valid negative control. Extend
   fresh-data iterations within the remaining cap before a broad hyperparameter
   sweep. Measure critic fit and policy0 Flow-SDE/ODE coverage to choose a single
   next ablation. Additional budgets above100h require approval. Student work
   remains paused until an effective source learner is established.

## Engineering conditions and spending

Retain shared scene offload/recycling, batched simulator control, skipped
intermediate rendering, frozen encoder eval mode, bounded asynchronous evidence
writes, local NVMe caches/staging, BOS cold archives, allocated GPU/render binding,
full sampler/RNG/Adam checkpoints, and combined resource-reserve monitoring.
Native likelihood partition is now a numerical condition. Raising memory usage
by changing it is inadmissible without exact joint-likelihood qualification;
measure useful end-to-end throughput instead of filling VRAM for its own sake.

Both corrected two-collection training controls now passed independent cold checkpoint
and numerical readback. The first full DEV readouts completed64 SFT cases each,
then failed the optimizer representation guard before evaluating checkpoint2.
Actual saved failing layouts proved only empty unowned entries were added by
native RLinf offload/onload; every captured owned Adam descriptor and group stayed
exact. Twelve Linux CPU core tests and the actual native functions qualify the
narrow exception. All8 GPU guards and complete checkpoint2 readouts remain required.

Selected checkpoint2 GRPO confirmation is complete and independently verified:
SFT7/64→10/64, eight gains/five losses, +4.6875points, paired p=.5810547.
The predeclared effectiveness gate failed. Do not pool DEV and confirmation
or reselect a checkpoint on confirmation. Student work remains paused.

The final checkpoint1 existing-DEV diagnostic and independent cross-run audit
are complete. GRPO's larger DEV curve is6/64→10/64→11/64; checkpoint2 versus1
has three gains/two losses and p=1.0. The small progress-panel regression was
not supported by this larger comparison. Checkpoint2 confirmation stays negative.

Closed usage is356198GPU-seconds (98.943889h);3802GPU-seconds remain in the
100h cumulative cap. No GPU jobs are active or reserved. Further complete
training needs additional approval. The prepared `MOTUS-LONGER-CURVE-PROPOSAL.md`
requests155h cumulative for eight fresh GRPO collections through checkpoint10,
fixed DEV and new confirmation, with Linux qualification before admission.
The budget request is pending; it is not launch authorization.
Each new job has a fresh output identity and immutable reviewed main commit.
Check advancing progress after launch and monitor at most20 minutes apart;
inspect the exact owned job and preserve its latest complete checkpoint before
any intervention. Record live progress only in our row of `outputs/coordination.md`.

Literature: `docs/literature/robotwin-ppo-review.md`; formulas and search/source
identities: `RECIPE-REVIEW.json`; numerical measurements:
`MOTUSJOINTAUDIT01-COMPLETE.json` and `MOTUSBEHAVIORUPDATE01-COMPLETE.json`.
