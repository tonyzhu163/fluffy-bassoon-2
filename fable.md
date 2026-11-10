# ICML plan: constructed histories for in-context adaptation of a frozen VLA

Revised November 10, 2026. This plan replaces the PPO-then-distil pipeline as the main line and responds to the [VLRLA reviews](https://openreview.net/forum?id=6ZrDl6noHg). It builds on the [simulation experiment plan](outputs/vlrla-feedback-plan-20261109-r1/PLAN.md). No compute reservation or launch is attached. ICML 2027 has not published a call for papers as of November 10, 2026; aggregators project January 16 (abstract) and January 22 (paper) from the 2026 cycle.

## Why the pipeline changes

- **PPO histories followed by AD/DPT/DIT/IQL is VLRLA's recipe.** Reviewer Hv9z called it "essentially identical to Algorithm Distillation". More learners on the same histories do not change that.
- **The teachers barely improve.** Three of five RoboCasa PPO teachers gained 7–9 points on 512 paired cases. On RoboTwin, with corrected wrist cameras, Sidney SFT/PPO40/GRPO40 scored 75/77/78 clean and 87/88/90 randomized, with p ≥ 0.45 ([result](outputs/robotwin-sidney-native-endpoints-20261109-r1/RESULT.md)).
- **No adaptation signal.** In the [Stage 1 AD readout](outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/STAGE1-RESULT.md), 1 of 10 full-versus-reset contrasts is nominally positive, unadjusted, one fit per interface.
- **DIT and IC-IQL were numerically AD.** The largest advantage weight in the four fits was 1.02–1.11 ([validation record](outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/monitor/VALIDATION-FINAL08.json)).
- **A learning-curve history does not contain the lesson.** A later episode is better because the teacher's weights changed, and each episode starts from a new scene. Nothing in episode k predicts the right action in episode k+1.

## Contribution

A way to build training histories in which the earlier episode carries the information the later one needs, and evidence that a frozen VLA trained on them uses its own previous attempt.

1. **Persistent hidden variable.** Something the current observation does not reveal stays fixed across a trial's episodes and is drawn independently of scene and attempt index.
2. **Branching.** At the query decision, restore the same simulator state and execute several alternative actions. Record each outcome.
3. **Measured label.** The query target is the alternative that succeeded. No trained teacher is involved.

Two trials with the same query observation and opposite latents then need different actions, and only the support history distinguishes them.

## Dataset protocol

One trial is built in seven steps, identical on every benchmark.

1. **Draw scene and latent independently.** Sample a task, a scene seed and a latent value. The latent must not correlate with scene or attempt index.
2. **Run the support episode.** The frozen VLA acts normally with the latent applied. Save public observations, delivered actions, reward and boundaries. Keep failures.
3. **Reset to a fresh query scene.** Same task and latent, new starting pose.
4. **Run to the decision point and save the simulator state.** The decision point is where the correction applies, such as the approach before first gripper closure.
5. **Branch.** Restore that state K times and execute a fixed set of alternatives, including "no correction", with the same motor noise. Record success and a continuous measure such as signed miss distance.
6. **Write the record.** Support history, query observation, every alternative with its outcome, and the index of the best. The latent value, object poses and unchosen-branch outcomes go in an evaluator-only file the student never reads.
7. **Split by whole trial and by scene.** Hold out a latent region, whole scenes and at least one task before fitting or normalisation.

Before a large collection, verify on development scenes that returning to the decision point and repeating the same action reproduces the same outcome. Otherwise branch outcomes are noise. On RoboCasa this already holds for reset-by-seed plus exact command replay: the [bowl fork script](outputs/robocasa-compact-bowls-20261109-r1/fork_grasp.py) asserts every public state equal to the original trace before branching.

| | RoboCasa365 | RoboTwin | LIBERO plus/pro |
| --- | --- | --- | --- |
| Role | Primary: build and train here | Transfer test and comparison on VLRLA's tasks | Third embodiment; untouched transfer test |
| Candidate latent | End-effector offset on handle tasks | Per-arm offset, or which arm is miscalibrated | Released S18 offsets |
| Return to decision point | Verified: seed reset plus command replay matches every public state | SAPIEN; replay determinism not yet qualified here | MuJoCo; not yet checked here |
| Known obstacle | No handle acquisition under offset on two pilot tapes | Phone stand is at 75–87% SFT; choose lower-success tasks | S18 perturbs reported positions, not the physical command, with one direction for all 40 tasks |

S18's single direction gives no pairs with opposite correct answers, so it cannot be training data without a custom variant. If SAPIEN restore is not deterministic, RoboTwin supplies support and query episodes for evaluation only. The [pinned S18 code](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/simulator-source/libero/libero/envs/nonphysical_perturbations.py#L3213-L3247) was re-read on November 10 and adds offsets only to observation keys containing `eef_pos` or `joint_pos`. The single direction across 40 tasks rests on the [Nov 9 static check](outputs/vlrla-feedback-plan-20261109-r1/PLAN.md) and was not re-counted.

**First build.** One task and one latent on RoboCasa, about 100 trials with five alternatives each. Measure restore determinism, how often any alternative succeeds, and controls per trial. This is the same run as E1 and prices the full corpus.

## Claim

A frozen VLA trained on constructed histories corrects its next attempt from one support episode, and the correction depends on which support episode it saw. On native benchmark tasks, where nothing persists across resets, history does not help.

## What the reviewers asked for

| Objection | Response |
| --- | --- |
| Identical to AD; limited novelty | The contribution is the data construction and the finding, not the bridge |
| Unclear what a failure tells a pretrained VLA | The persistent latent makes the failure informative by design; own versus donor history tests it |
| Zeroed history is out of distribution | Separately trained memoryless bridge; donor histories stay in-distribution |
| Gains may be imitation of later-in-sequence behaviour | No learning-curve ordering exists in constructed data; attempt-index control retained |
| The two-plate result replays a constructed rule | Continuous latent, held-out latent region, held-out task, second latent family |
| No advantage over GRPO; one task carries the average | Success versus delivered actions at equal budget; wins/ties/losses; leave-one-task-out |
| Feedback and repeated attempts are a cost | All supports, resets and labels charged; branching cost reported as training cost |

## Experiments

### E0. Prerequisites in code

- **Done, untested on Linux:** `native_pilot.wrist_near` sets the wrist near plane when RoboTwin builds its cameras, in training and evaluation, and every reset checks it. The Sidney config sets 1 cm. Pinned RoboTwin hard-codes 10 cm. Qualify it on Baige with the unit test and one real episode before any PPO/GRPO baseline.
- **Open:** `validate_recipe` on `main` admits only the two-iteration `adjust_bottle` pilot. The Sidney PPO/GRPO arms ran from run-pinned source. Generalise the check when the E5 baseline is contracted, with its Linux tests.
- **Open:** a query-label path for the student interface, chosen after E1 picks the actuator.

### E1. Control authority (go/no-go)

The Nov 9 pilots are small but unfavourable. Frozen π0.5 released to the right regardless of the requested bowl, while a scripted carry reached the requested side in 6 of 8 continuations ([compact bowls](outputs/robocasa-compact-bowls-20261109-r1/RESULT.md)). The drawer offset changed the lateral miss but produced no handle acquisition on the two tapes examined ([hidden-goal pilot](outputs/robocasa-meta-feedback-20261109-r1/RESULT.md)).

For one task and one latent, branch a fixed set of alternatives from saved query states on development scenes. Compare three actuators: KV or velocity steering of the frozen expert, a bounded residual on the decoded action, and a choice among scripted corrections.

**Go** with the weakest actuator for which some alternative succeeds on a clear majority of states where the uncorrected action fails. **No-go** if none does; the paper then waits on a different latent or task.

**Latent-prediction diagnostic.** On the same development trials, fit a small supervised head that predicts the imposed offset from the support-history tokens, and a second from the current query observation alone. Hold out whole scenes. This separates the two failure modes:

- History head fails: the information is not in the tokens. Change what enters context before fitting any policy.
- History head succeeds but no alternative succeeds: the limit is control authority.
- Current-observation head succeeds: the latent is visible without history, so the setting cannot test adaptation.

The head uses the true latent as its label, so it is a diagnostic and an upper bound, not a method arm.

### E2. Constructed histories, primary

- **Latent.** A continuous 2D lateral end-effector offset, fixed through support and query. The query uses a fresh starting pose.
- **Corpus.** One real support episode, then the query state with all branched alternatives and their outcomes. Whole trials stay together across the train/validation split.
- **Main learner.** DPT-style: predict the measured best alternative at the query, given the support history. Three independent fits.
- **Label-source arms on the same pairs.**
  - Behaviour cloning of the action actually taken.
  - Outcome-weighted imitation, with weights from measured branch outcomes and no learned critic.
- **External baseline.** AD on PPO histories (the VLRLA recipe), from the existing RoboCasa corpus.
- **Diagnostic.** The latent-prediction head from E1, refit on each learner's history representation, reported beside its success rate.
- **Query conditions.** Own history; donor with a nearby latent; donor with a distant latent; episode reset; trained memoryless; trained reward-blind; native SFT retries.
- **Primary endpoint.** Paired difference in native query success after one support, own history minus distant-latent donor, pooled across tasks.
- **Mechanism readout.** Signed commanded displacement at first grasp against the imposed offset.

### E3. Does it generalise?

1. A held-out region of offset direction and magnitude.
2. A held-out task with a different handle.
3. A second latent family, chosen after E1.

Items 1 and 2 are required for a main-track claim.

### E4. Native tasks, secondary

Own versus donor versus reset on RoboCasa atomic tasks with the same trained controls. Expected result: no own-over-donor effect. Fix the task selection rule before reading outcomes.

### E5. Deployment cost

Success versus delivered physical actions at 0, 1, 2, 4 and 8 supports for native retries, per-task PPO/GRPO from the same initialiser, the memoryless bridge and the candidate. PPO/GRPO appears here only, as the equal-budget baseline on two or three tasks. On RoboTwin it must be retrained with corrected wrist cameras first; phone stand has little headroom at 75–87% SFT, so pick tasks by SFT success band.

## Evidence thresholds

- **Primary.** Pooled own-minus-donor gain of at least 5 pp, paired 95% lower bound above zero, same sign in all three fits. Per-task intervals at 200 cases are about ±6 pp and are secondary.
- **Label source.** The measured-label learner beats behaviour cloning on the same pairs.
- **Against the existing recipe.** It beats AD on PPO histories on the persistent-latent tasks.
- **Reward attribution.** Claimed only if the trained reward-blind bridge loses the gain.

## Schedule and gates

| Date | Milestone | Gate |
| --- | --- | --- |
| Nov 17 | E0 done; E1 read | No actuator with authority ends this route |
| Nov 28 | E2 corpus built; first fits | |
| Dec 8 | E2 development readout, three fits | No own-over-donor effect means no main-track submission |
| Dec 22 | E3 items 1 and 2; E4 | |
| Jan 9 | E2 confirmation on fresh cases; E5 | |
| Jan 16 / Jan 22 | Abstract / paper | Projected dates; confirm when the call is posted |

Dates are proposals, not derived from measured throughput.

## Out of scope

New PPO teacher campaigns as a data source, the four-learner comparison, DIT and IC-IQL with learned critics, Flora and physical-robot trials, RoboCasa composites, and any new VLA fine-tuning.

## Risks

- **No control authority.** E1 tests it first. A stronger actuator weakens the "frozen expert" framing and must be reported as such.
- **The latent is visible in the current observation.** Then history adds nothing. The current-observation head in E1 checks this before any large fit.
- **Constructed-rule objection.** E3 decides whether the result is more than one synthetic perturbation.
- **Simulator-only labels.** Branching is unavailable on a real robot. The claim is limited to training in simulation and adapting from ordinary episodes at deployment.
- **Compute.** E2 exceeds 50 GPU-hours. Price it from E1 throughput and get approval before launch.
