# ICML plan: constructed histories for in-context adaptation of a frozen VLA

Revised November 10, 2026. This plan replaces the PPO-then-distil pipeline as the main line and responds to the [VLRLA reviews](https://openreview.net/forum?id=6ZrDl6noHg). It builds on the [simulation experiment plan](outputs/vlrla-feedback-plan-20261109-r1/PLAN.md). No compute reservation or launch is attached. The ICML 2027 deadline is assumed to be late January 2027 and has not been checked.

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

- Move the 1 cm wrist near plane into the RoboTwin training environment. It currently exists only in the [evaluation script](outputs/robotwin-sidney-native-sft-20261109-r1/run.py).
- Let the checked-in runner accept the Sidney phone-stand recipe; `validate_recipe` admits only the two-iteration `adjust_bottle` pilot.
- Add a query-label path for the student interface used in E2.

### E1. Control authority (go/no-go)

The Nov 9 pilots are small but unfavourable. Frozen π0.5 released to the right regardless of the requested bowl, while a scripted carry reached the requested side in 6 of 8 continuations ([compact bowls](outputs/robocasa-compact-bowls-20261109-r1/RESULT.md)). The drawer offset changed the lateral miss but produced no handle acquisition on the two tapes examined ([hidden-goal pilot](outputs/robocasa-meta-feedback-20261109-r1/RESULT.md)).

For one task and one latent, branch a fixed set of alternatives from saved query states on development scenes. Compare three actuators: KV or velocity steering of the frozen expert, a bounded residual on the decoded action, and a choice among scripted corrections.

**Go** with the weakest actuator for which some alternative succeeds on a clear majority of states where the uncorrected action fails. **No-go** if none does; the paper then waits on a different latent or task.

### E2. Constructed histories, primary

- **Latent.** A continuous 2D lateral end-effector offset, fixed through support and query. The query uses a fresh starting pose.
- **Corpus.** One real support episode, then the query state with all branched alternatives and their outcomes. Whole trials stay together across the train/validation split.
- **Main learner.** DPT-style: predict the measured best alternative at the query, given the support history. Three independent fits.
- **Label-source arms on the same pairs.**
  - Behaviour cloning of the action actually taken.
  - Outcome-weighted imitation, with weights from measured branch outcomes and no learned critic.
- **External baseline.** AD on PPO histories (the VLRLA recipe), from the existing RoboCasa corpus.
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
| Late Jan | Submission | Deadline to be confirmed |

Dates are proposals, not derived from measured throughput.

## Out of scope

New PPO teacher campaigns as a data source, the four-learner comparison, DIT and IC-IQL with learned critics, Flora and physical-robot trials, RoboCasa composites, and any new VLA fine-tuning.

## Risks

- **No control authority.** E1 tests it first. A stronger actuator weakens the "frozen expert" framing and must be reported as such.
- **The latent is visible in the current observation.** Then history adds nothing. Check with a current-inputs-only predictor before any large fit.
- **Constructed-rule objection.** E3 decides whether the result is more than one synthetic perturbation.
- **Simulator-only labels.** Branching is unavailable on a real robot. The claim is limited to training in simulation and adapting from ordinary episodes at deployment.
- **Compute.** E2 exceeds 50 GPU-hours. Price it from E1 throughput and get approval before launch.
