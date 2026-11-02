# Flora-Rewarder PPO

**Current first step (2026-11-01).** The paper stays about in-context RL for
frozen VLAs ([PAPER.md](PAPER.md)). Before more ICRL work, get an ordinary RL
loop working where the reward comes from a learned video rewarder instead of
the simulator. That loop is the reward source later in-context learners need
on tasks without a success checker.

Live plan and status: [run plan](../outputs/flora-ppo-comparison-20261101-r1/PLAN.md),
[coordination](../outputs/coordination.md).

## Rewarder

- [`nvidia/Flora-Rewarder`](https://huggingface.co/nvidia/Flora-Rewarder)
  ([project page](https://research.nvidia.com/labs/gear/flora)): a 27B video
  reward model. Outputs an episode score 1–5 and per-step
  normal/suboptimal/failed labels. Non-commercial license; download with the
  user's own Hugging Face authentication to project NVMe.
- Too slow for dense per-step reward. Use one terminal outcome label per
  episode. Its reported failed-step recall is only 53%, and its training
  simulators are not published, so measure its agreement with the simulator
  before trusting it on a task.

For RoboCasa, new reward recordings contain synchronized left, wrist and right
RGB videos, in that collage order. Use the same official terminal-success
prompt and 1–5 score across tasks, supplying only the native task instruction;
completed and active protocols use no task-specific rubrics. A new
[qualitative completion-criteria candidate and replay](../outputs/robocasa-flora-rubric-replay-20261102-r1/RUBRIC.md)
is drafted from native success predicates for separate validation. It uses
observable end states rather than asking the video model to measure simulator
distances or percentages. A rubric change requires a new prompt/calibration/cache
identity; the original qualification does not carry over. The policy already consumes these three
views. Save each view from reset through the final physical step, verify all
video hashes, and bind the camera order in the reward protocol. Archived
left-only results retain their original protocol and cannot qualify a new
three-view run. Camera changes require a new scoring/calibration identity,
with simulator success kept outside the reward prompt.

## Benchmarks, in order

1. **LIBERO** is a test bed only, not a reported benchmark: pi0.5 leaves too
   little headroom. Use it to get the loop working, starting from RLinf's
   published 40-task few-shot pi0.5 SFT checkpoint, and to measure Flora
   against simulator success.
   If a Flora plumbing check is needed there, LIBERO90 task 59 has headroom
   on the frozen official OpenPI `pi05_libero` (C565) motor: native 3/16,
   untrained tied noise 9/16
   ([screen](../outputs/libero90-pi05-headroom-screen-20261101-r1/RESULTS.md)).
2. **RoboCasa365** is the reported benchmark, using the RLinf-loadable
   RoboCasa365 pi0.5 checkpoint, starting with atomic tasks from its 50-task
   evaluation subset; composite tasks come later. It is NVIDIA's GR00T
   benchmark, closer in appearance to the rewarder's likely training data and
   far from saturated. Always report true simulator success.
   **Horizon (human, 2026-11-02):** the five PPO tasks run at 600 controls
   (12 H50 chunks) in training, Flora scoring and ID evaluation, cutting
   stale end frames from Flora's 64-frame sample. Native pi0.5's longest
   first success was 538 controls over 126 episodes, and one
   PickPlaceDrawerToCounter end success was lost at 600
   ([timing](../outputs/robocasa-pi05-success-timing-20261102-r3/RESULT.md)).
   OOD tasks keep their native horizons, because Flora never scores them and
   the residual has no clock input; success at control 600 is a secondary
   number from the same records. 600-step results need their own Flora
   calibration and headroom check, and never pool with 750-step results. The600-control
   factory and four-lane colocated memory guard now pass qualification
   ([audit](../outputs/robocasa-joint-lane-qualification-20261102-r5/PARENT-REVIEW.json));
   fresh50/task calibration and the joint before-panels are still pending.
3. **RoboTwin** follows, for the comparison with VLRLA, the direct precedent
   in [PAPER.md](PAPER.md). It needs its own Flora agreement check: two-arm
   success is harder to judge from one camera.
4. **Real robots** come after a simulated sample-efficiency gain, with Flora
   labeling logged episodes offline and scoring live episodes online.

## Gates on each benchmark

Flora optimization is parked until the simulator-reward positive control
passes and the false-full-score problem is corrected and independently
qualified. The retained TRAIN audit found full scores on58/183 true failures;
AUROC alone does not qualify rewards for optimization. Measure false-positive
full scores and lost true positives on held-out cases before adopting a fix.
More source-history collection waits for measured simulator-reward learning.

Earlier LIBERO and RoboTwin runs failed because source learners never
improved ([FINDINGS.md](FINDINGS.md#lessons-carried-forward)). Pass each gate
before the next:

0. **Gate 0: Flora agreement and headroom, per task.** Score native rollouts with
   Flora and require AUROC ≥ 0.7 against simulator success. Keep tasks whose
   native success is neither 0 nor ceiling, and screen untrained tied noise
   as well as native sampling.
1. **Gate 1: reward positive control.** Ordinary PPO or DSRL must improve true success with
   simulator reward and with Flora reward. If only the simulator arm
   improves, the reward is the problem, not the method. The RLinf action-expert
   scale-up covers only this gate's simulator-reward half. Its result cannot
   establish improvement by a frozen-motor controller.
2. **Gate 2: matched frozen-motor learner.** Qualify ordinary interface PPO
   on the interface selected by gate 0, with sampling noise inside the policy,
   first-success returns and chunk discounting. Use the same deterministic
   steering map for the baseline, history source and meta-learner.
3. **Gate 3: collect histories** with Flora reward only, from learners that
   improve. Plan source histories across OpenDrawer, DrawerToCounter,
   CoffeeSetupMug and TurnOffStove; hold out whole source streams for validation
   and use separate held-out-task folds to test task generalization.
4. **Gate 4: steering interface and meta-learner.** The meta-learner is a standard
   plain causal decoder (AD/DPT/Vintix); architecture is not a tested axis
   ([PAPER.md](PAPER.md)). The steering interface is a bounded action residual
   (alternatives in [PAPER.md](PAPER.md)); the ordinary RL
   baseline and the history-generating learner use the same interface as the
   meta-learner ([review](literature/vla-steering-review.md)). The
   sample-efficiency claim is query success per robot episode against that
   ordinary Flora-reward learner at equal interactions, with reset and
   reward-blind controls. Full-weight PPO is a reward check, not that
   baseline.

Prospective comparisons share the case/noise/attempt schedule in
`paired_evaluation.py`: 200 canonical cases per task by default, paired motor
streams for every model and intervention, and both first-success and cutoff
success. Include same-weight memory reset and reward-blind controls; cluster
repeated draws by case. Historical 50-case development panels remain separate.
New decoder fits use a 128-token training window and at least 128-token model
capacity; see [fit profile](../configs/native_ppo/gate4_history.json). Capacity
alone does not show that long contexts were trained.

Flora first success needs a separately qualified causal prefix/chunk rule:
report confirmation precision, recall and signed timing against native first
success, including premature detections and never-success episodes. A terminal
score cannot supply those timestamps. A hold over chunk scores is distinct
from a hold over consecutive frames. [Readiness work](../outputs/gate-readiness-20261102-r1/PLAN.md).

## RoboCasa residual PPO rollout execution

Fresh residual-PPO training and its common-before profile default to
`target_kl=0.02`, unless explicitly configured otherwise. After each optimizer
epoch, evaluate the current policy on the entire collected chunk batch and
stop actor and critic optimization if the mean sampled KL from the behavior
policy exceeds that threshold. Keep the completed epoch; this is early
stopping, not rollback or a guarantee that final KL stays below the threshold.
An initial check also rejects further optimization of an already over-threshold
policy on a pending batch. Historical resumes preserve their saved setting;
`target_kl=null` retains fixed-epoch optimization. The checkpoint contract
includes any enabled threshold and rejects changing it during recovery.

Each `ppo-epoch-III-EE.json` records full-batch KL, clip fraction, actor and
critic losses, entropy, explained variance, pre-clipping gradient norms,
optimizer-step counts and elapsed time. Epoch rows also stream into the
progress log. The final update record includes requested/completed epochs,
actual optimizer steps and sample passes, stop reason, initial/final policy
diagnostics, and raw advantage/return statistics. Explained variance is null
for constant returns. Existing top-level loss/KL values remain averages of
pre-step optimization minibatches; use `final_metrics` for the final policy's
KL and `epoch_metrics` for its evolution.

Future matched joint runs use eight native environments per GPU and one shared
frozen policy copy. The scientific batch stays at 80 complete episodes, 16 per
training task; execution batching does not increase its size or change the
configured optimizer. Two-GPU nodes request 64 CPUs and 256 GiB RAM, matching
the measured capacity configuration. Each worker continuously refills its eight lanes from the original case bank,
starting the next case as soon as a lane finishes. The selected cutoff for
new matched comparisons is **600 physical policy controls (12 H50 chunks)**,
based on the [partial native first-success panel](../outputs/robocasa-pi05-success-timing-20261102-r3/RESULT.md).
Both reward arms and their before/after evaluations must use the same cutoff
and batching. This is a prospective protocol decision: implementation and
Flora calibration must bind the600-step clips and terminal labels before a new
comparison is admitted. Existing750-step calibrations and pinned runs retain
their original contracts. First success and success at the cutoff are distinct;
the timing panel contains one DrawerToCounter terminal success lost at600.

Motor and actor draws are assigned in the original serial case/chunk order.
Worker records return in that order before labels or GAE, and a failed lane
leaves the entire bank incomplete. Checkpoints bind the lane count, original
case IDs and full recovery state. Legacy profiles without the new field retain
one lane; current run worktrees and historical recoveries remain unchanged.
Batched BF16 inference is not bit-identical to historical singleton inference.

The colocated Flora startup check requires at least 27,408 MiB free after
scorer warmup: the measured eight-lane native peak plus an 8 GiB reserve.
Standalone rollout capacity does not qualify simultaneous scorer/rollout
memory use; a job fails before dispatch if that reservation is unavailable.
See the [execution configuration](../configs/native_ppo/robocasa_residual_execution.json)
and [allocated Linux core checks](../outputs/robocasa-rollout-batch-integration-20261102-r1/QUALIFICATION.json).

## First test: LIBERO90 task 44, simulator reward vs Flora reward

A single-seed pilot to decide whether the idea is worth pursuing; the
[run plan](../outputs/flora-ppo-comparison-20261101-r1/PLAN.md) is authoritative.
All other jobs are paused for it.

- **Policy:** the published PyTorch
  [`RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc4`](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/tree/45ccfcc4e28634f1576ebf78cab0fbe2fd82432d),
  trained with 40 demonstrations across the four ten-task LIBERO suites
  ([paper, section 5.2.1](https://arxiv.org/html/2510.25889v1#S5.SS2.SSS1)).
  Both arms use the same weights and bundled normalization through RLinf's
  native loader, with strict backbone loading and a fresh shared baseline.
  No additional SFT or JAX conversion is required. This is a different
  initializer from historical OpenPI40 `c565…`; its success rate is measured
  anew. RLinf130 is excluded because its SFT includes task 44.
- **Recipe:** RLinf's published pi0.5 LIBERO PPO config restricted to task 44
  (`KITCHEN_SCENE9_turn_on_the_stove`), 32 envs × 4 rollout epochs at H400
  (~53k actions per iteration). Training disables success termination so
  episode length cannot leak success. One terminal reward per episode, with no
  extra reward terms.
- **Arms:** S gets 1 if the simulator says the stove is on in the final state,
  otherwise 0. F gets Flora's graded `(expected_score − 1) / 4` partial credit;
  there is no threshold and no tuned mapping. F's trainer can never read the
  simulator label, which is logged to a sidecar for analysis.
- **Steps:** (1) Flora smoke test on one episode. (2) 64 baseline episodes
  labeled by both, giving the start rate, score distributions and AUROC; stop
  if the start rate is outside 5–70% or AUROC < 0.7. (3) A relabel hook before
  GAE plus CPU tests: S with a stub rewarder reproduces the plain config, and F
  cannot see the simulator label. (4) Up to 10 iterations per arm, with 50
  simulator evaluation episodes (states 40–49 × 5 draws) after each. (5)
  `RESULT.md`: true-success curves against actions and GPU time, Flora score vs
  truth (rising score with flat success means reward exploitation), and costs
  with Flora's compute charged to F.
- **Stopping:** plateau after ≥6 iterations when best success gains < 4 points
  over the last four evaluations (futility if < 10 points above baseline);
  ceiling at three evaluations ≥ 48/50; hard action, time and wall-clock limits.
  Decisions use simulator success only. Stop at a closed boundary with full
  recovery.
- **Budget:** ~76k all-in actions per iteration with 4/8 workers, including
  six unscored evaluation padding episodes, under 1M per arm. GPU-hours
  are uncapped for these two arms by human authorization, but each scheduler
  job may request at most 8 GPUs and must run under 8 hours. Logs are plain
  text heartbeats, with no dashboard.

Task 44 has a historical simulator-reward control: ordinary DSRL took the
separate OpenPI40 native pi0.5 initializer from about 5/20 to 20/20 there
([FINDINGS.md](FINDINGS.md#libero)). This is not the starting baseline or a PPO
result for the ready PyTorch checkpoint used in this comparison.
