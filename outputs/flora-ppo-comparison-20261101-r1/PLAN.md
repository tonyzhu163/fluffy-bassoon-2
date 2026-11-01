# Quick test: RLinf PPO on LIBERO90 task 44, simulator reward vs Flora reward

Draft, 2026-11-01; initializer correction published on main `2a1eebae`.
The human authorizes launch when ready, readable logs and early stopping, and
normalized partial credit for Flora. **No comparison job has launched.** The
separate frozen-pi0.5 DSRL allocation is stopped and released; its recovery and
ledger remain separate, and other jobs stay paused.
This is a single-seed pilot to see whether the result is worth pursuing. It is
not a definitive comparison.

## Question

On LIBERO90 task 44 (`KITCHEN_SCENE9_turn_on_the_stove`, "turn on the
stove"), does PPO trained only on
[`nvidia/Flora-Rewarder`](https://huggingface.co/nvidia/Flora-Rewarder)
graded outcome labels improve true success faster than PPO trained on the
simulator's sparse success reward? The simulator check scores every evaluation.
Flora is used only as the training reward in arm F. This compares binary
simulator feedback with graded learned feedback; it does not isolate label
accuracy from the value of partial credit.

## Setup

- **Policy:** the official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`
  release, already fine-tuned on LIBERO40. Use the same immutable starting
  weights and normalization for both arms; do not train another LIBERO40 stage.
  The staged source receipt is
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`
  and its JAX parameter signature is
  `c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9`.
  [Release provenance](../pi05-libero40-provenance-20261029-r1/PROVENANCE.json).
- **Human correction (2026-11-01):** use OpenPI40 to test adaptation outside
  the release's LIBERO fine-tuning task set. The earlier RLinf130 choice includes
  task44 and would answer a seen-task fine-tuning question. It is not an allowed
  fallback for this comparison. LIBERO40 already includes the operation "turn
  on the stove"; task44 is a different benchmark task/scene instance, not proof
  of a novel skill or absence from foundation pretraining. Task44 has also been
  selected through prior development screening.
- **RLinf loading prerequisite:** convert/load this exact source through the
  official path, bind its normalization, tokenizer, camera/state/action mapping,
  and verify fixed-input/fixed-noise action parity before the first update.
  Record actual native step0 success under the comparison's sampler and
  full-horizon final-state criterion. Historical OpenPI40 native6/20 is a
  reference from another evaluation protocol, not this new baseline.
- **Recipe:** RLinf's pi0.5 LIBERO PPO config, restricted to task 44, with
  the published PPO hyperparameters. To get more updates within a small
  budget, reduce the env count to 32 envs × 4 rollout epochs. That is about
  51k actions per iteration at a 400-step horizon.
- **Episodes in both arms:** success termination is disabled in training, so
  every episode runs the full horizon. Otherwise episode length would leak
  the simulator's success signal into arm F. Each episode gets one terminal
  reward on its last valid chunk, before GAE, as specified below. No process
  rewards, extra reward terms or changes to the official PPO losses are added.

| Arm | Who decides the terminal reward |
| --- | --- |
| **S** | 1 if the simulator says the stove is on in the final state, otherwise 0. |
| **F** | `(expected_score - 1) / 4`, using Flora's expected outcome score in [1, 5]. The trainer never reads the simulator label. |

In arm F, the simulator label is still logged to a side file for later
analysis, and the trainer never reads that file. The human's partial-credit
choice supersedes the earlier binary-threshold design: there is no training
cutoff, fitted score transform or per-batch min/max scaling. Preserve the raw
score, probabilities and expected score. Reject missing, non-finite or
out-of-range scores before any update; retain the pending batch for recovery.
The normalized grade is not a calibrated probability of task success.

## Steps

1. **Flora smoke test.** Download the pinned revision to Baige NVMe and run
   one task-44 episode through it: up to 64 agentview frames, upright, with
   the instruction. Check that the outcome score parses.
2. **Baseline check, 64 episodes, no training.** Roll out the SFT
   policy at the full horizon and label every episode with both the
   simulator and Flora. This gives:
   - the starting success rate
   - Flora's score distributions for simulator successes and failures, and AUROC
   - the exact scorer/parser and fixed [1, 5] to [0, 1] mapping

   **Stop** if the starting rate is under 5% or over 70%, since PPO then has
   little to work with. **Also stop** if Flora barely tracks the simulator
   (AUROC < 0.7), indicating inadequate outcome separation on this bank.
   Either way, report what was found. Do not tune the score mapping from these
   outcomes. Use training states 0..39 for this check; a smoke episode counts
   toward 64 only if its policy, seeds and video recipe are identical.
3. **Wire up and test on CPU.** Add a relabel hook after rollout and before
   GAE, using Flora served on its own GPU. Add the full-horizon training env
   setting. Write tests that:
   - arm S, with a stub rewarder returning simulator labels, reproduces the
     plain config
   - arm F's trainer cannot read the simulator label
   Finish these focused checks and exact-source loading/parity qualification
   before admitting training; run tests and model computation on allocated Linux.
4. **Run.** One seed each for S and F, from the same commit, seed and
   training states, for up to 10 iterations, subject to the stop and budget rules.
   After every iteration, evaluate 50 episodes using the simulator check:
   states 40..49 with five predeclared action draws each. These are ten reserved
   development placements, not 50 independent states or a blind test. Use the
   same evaluation schedule and full-horizon final-state criterion in both arms.
5. **Report.** Write `RESULT.md` with:
   - true-success curves against physical actions, elapsed time and GPU-hours;
     disclose repeated placements when reporting uncertainty
   - Flora score versus simulator success during training in arm F; a rising
     Flora score with flat/falling true success flags possible reward exploitation
   - actions and GPU time per arm, with Flora's compute charged to F, exact stop
     reasons, shared-budget comparisons and time/actions to a true-success target

## Visibility and stopping

Use well-formatted text logs, not a dashboard or external tracking service.
Write a timestamped line on phase changes, episode/label/update/evaluation
completion and at least every 60 seconds during long stages. Each line includes
the run, arm, phase, iteration, progress, elapsed time and remaining budget.
Evaluation lines include true successes out of 50, current/best success and
plateau progress. Phase summaries include rollout/label/update/evaluation time,
physical actions (training, evaluation, settle and failures), aggregate GPU-hours,
GPU memory, PPO KL/clip fraction/value loss and available official diagnostics.
Log Flora score distributions and its relationship to simulator truth through
the isolated monitoring sidecar. Report stale or failed stages as such.

Keep append-only scalar metrics and an atomic latest-status record, including
the last heartbeat, complete checkpoint path and stop reason. These support
readable logs and later reporting without a separate UI.

Freeze the same per-arm rule before collecting outcomes. Let `p0` be the native
baseline success and `p_i` the true evaluation success after PPO iteration `i`;
let `B_i = max(p0, p_1, ..., p_i)`.

- **Plateau:** after at least six iterations, stop if
  `B_i - B_(i-3) < 0.04` (less than four percentage points of best-success gain
  across the latest four evaluations). If `B_i - p0 < 0.10`, report low/flat
  performance as **futility**, not successful convergence.
- **Ceiling:** stop after three consecutive evaluations at or above 96% true
  success (at least 48/50). This rule can stop before iteration six.
- **Hard limits:** stop before any next phase would exceed the admitted action,
  GPU-time or per-job wall-clock limit, regardless of learning progress.

These are pilot stopping heuristics, not statistical proof of convergence.
The controller uses simulator evaluation success, never Flora score or training
return, for convergence decisions. Stop at a closed rollout/update/evaluation
boundary, save full model/critic/optimizer/sampler/RNG and iteration state, emit
the reason and checkpoint reference, then release the allocation. Reserve time
for this closure. A stopped arm stays frozen; its peer may continue under the
same rule and existing cap. Compare shared-budget prefixes, and report an arm
that never reaches a target as censored; do not claim convergence from a crash.

## Budget

- One 32-env x 4-round training iteration uses an estimated 53,120 physical
  actions at H400 plus the official LIBERO default of 15 settle actions per
  episode. Its 50 evaluation episodes add 20,750, for 73,870 all-in actions per
  iteration, or 738,700 for ten iterations.
  Count actual physical execution, including padding/overshoot, rather than
  relying on this estimate; never exclude evaluation or settle actions.
- Keep the 1M-per-learner action ceiling. The earlier 20-iteration extension
  would exceed it. If S has not learned by iteration ten and neither stop rule
  has fired, admit only matched extensions that fit all remaining limits (at most
  13 iterations under the estimate above), never extend a stopped arm. Do not
  judge F against an S that has failed to learn.
- The shared 64-episode baseline plus at most one separate smoke episode uses
  an estimated 26,975 physical actions. Give this common preparation its own phase
  bucket, charge it once and disclose its attribution alongside each arm.
- **Human resource override (2026-11-01):** "no these two jobs i authorize
  however many, you can only ask up to 8 gpus per job". The former ten aggregate
  GPU-hour ceiling is superseded for these two comparison arms and their
  necessary qualification/scoring. No replacement aggregate GPU-hour ceiling
  was supplied; do not limit a run to the earlier 67/35-minute budget examples.
  Request at most eight GPUs in any scheduler job, counting every GPU in that
  allocation. A separate rewarder allocation must independently satisfy this
  per-job limit. Choose useful execution parallelism from actual capacity and
  throughput, preserving the saved batch, losses, optimizer schedule and cases.
- Continue logging all GPU allocation time for startup, qualification,
  baseline, Flora, training, evaluation, failures and checkpoint closure. The
  standing rule that each individual job is under eight hours still applies;
  reserve closure time and preserve full recovery. The existing scientific
  action limits and convergence/futility rules remain in force. This resource
  authorization does not resume other paused work or add scientific arms.

## Reading the result

With one seed, only a large effect is readable. Possible outcomes:

- F tracks S, and both clearly rise above the starting rate: worth a
  multi-seed follow-up.
- F stays flat while S rises: Flora's labels don't carry enough signal for
  this task.
- F's score rises while true success does not: investigate reward exploitation
  or misaligned partial credit before claiming a learning improvement.
- S itself doesn't improve: the test says nothing about Flora.
