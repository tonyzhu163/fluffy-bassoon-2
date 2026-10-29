# In-context noise steering of a frozen pi0.5 on LIBERO-Plus/PRO and Meta-World

Status, 2026-10-29: **planned; no code or runs yet.** CPU work (code, tests,
cached meta-training) needs no ask under [AGENTS.md](../../AGENTS.md). Stage L0
and every later GPU collection need user approval of the stated budget.
This replaces ManiSkill as the benchmark for
[fallback idea 1](../coordination.md#1-in-context-strategy-selection-over-a-frozen-vla);
the [noise-mode plan](../noise-mode-icrl-20261029-r1/PLAN.md) supplies the arm
definition, the gates and `noise_screen.py`, which carry over unchanged.

## Question and claim

On a LIBERO task, scene and perturbation it never trained on, does a controller
with frozen weights raise pi0.5's success within a few attempts by choosing the
action expert's initial noise from its own history of attempts and rewards?

Target claim: the controller's success rises across the K attempts of a held-out
trial, the rise disappears when memory is reset or rewards are removed, it beats
a generic bandit and a static multitask DSRL actor at the same interaction
budget, and it reaches in K attempts a success that per-task DSRL needs far more
episodes and gradient updates to reach.

## When are we done

Better held-out success on LIBERO-Long/Plus/PRO is necessary but not
sufficient. A static steering actor trained on the same tasks could produce it
by ordinary generalization, which is DSRL's result, not meta-learning. The
claim is complete when all of T1–T5 below pass on held-out trials; T6 adds the
efficiency claim and T7 the architecture claim. If only static generalization
improves (T1 flat, T2 fails), report a multitask-DSRL result and stop calling
it in-context RL.

## Order

Meta-World runs first because its hidden goal makes a static shortcut
impossible and its kill test is cheap; LIBERO M0 runs alongside it because it
is the longest step and depends on no Meta-World result (concurrency approved
by the user, 2026-10-29; budgets fixed at launch).

1. Meta-World goal-hidden motor and mode-spread check.
2. In parallel, LIBERO M0.
3. Meta-World cached tests, T1–T5.
4. LIBERO L0, then L1 and L2; L3 only if warranted.

If the mode-spread check fails, fix the noise interface before further LIBERO
spending.

## Motor and interface

| Item | Choice |
| --- | --- |
| Motor | Our own SFT of the pretrained pi0.5 base on the demonstrations of the 78 training tasks only (Stage M0), then frozen. Not an RLinf LIBERO checkpoint: those trained on all 130 tasks, including the held-out ones |
| Inference | RLinf `e1a82eb` `pi05_libero` conventions, as in `src/fluffy_bassoon/rl/libero_goal.py` on branch `libero-goal-headroom`: 256 px views, 8-D state, 5 Euler steps, horizon 10 with 5 executed, 15 settle steps |
| Steering action | The initial flow noise `[10, 32]` of a chunk. Per-attempt arms (one fixed tensor for every chunk of an attempt, M = 16, as in the noise-mode plan) first; per-chunk noise (DSRL actor) later |
| Feedback | Binary success at attempt end and an attempt-boundary flag. No shaped reward |

Neither the motor nor the controller sees validation or held-out tasks in
training, so "held out" applies to the whole system. Verify, and record,
whether the pi0.5 base pretraining mixture contains any LIBERO data.

## Splits and perturbations

- Tasks: the audited split of 78 training, 10 validation and the official 10
  held-out (LIBERO-10) tasks; the two LIBERO-90 tasks whose goals duplicate
  held-out tasks stay excluded. Link the audit artifact here before L0.
- Perturbations: redo the audit at the level of task × perturbation. Every
  LIBERO-Plus or LIBERO-PRO variant of a held-out or excluded task is held out.
- Training uses LIBERO-Plus perturbation types that pass L0. Evaluation uses
  held-out tasks under those types, then LIBERO-PRO and any held-out
  perturbation types as a transfer test.
- A **trial** is one task, one scene and one perturbation held fixed for K = 8
  attempts (K = 16 for extrapolation). Each attempt resets to the same scene.
  Memory resets only between trials. Weights are frozen at evaluation.

## Stage M0: train the motor on the 78 training tasks (GPU, approval needed)

Fine-tune the pretrained pi0.5 base on the LIBERO demonstrations of the 78
training tasks only, following the openpi/RLinf LIBERO SFT recipe (inference
conventions as above). Select the checkpoint on the 10 validation tasks, never
on held-out tasks. Then freeze it and record its hash.

Qualify it before any steering work:

- Training tasks: success comparable to published LIBERO SFT, or the steering
  results only reflect a weak motor.
- Held-out tasks, zero-shot with native noise: expected to be low. That is the
  headroom, but it must not be zero everywhere. Noise can only choose among
  behaviors the motor already has, so L0 must find held-out tasks where some
  arms succeed. LIBERO-10 tasks reuse LIBERO-90 scenes and chain familiar
  subtasks, which makes that plausible; it still has to be measured.
- If held-out success is zero for every arm, the motor lacks the support and
  no noise controller can help. Then hold out fewer or easier tasks, or train
  on more data, and record the choice.

## Stage L0: headroom per perturbation type (GPU, approval needed)

No training. With the M0 motor, for about 6 training and all 10 held-out
tasks, clean and under each LIBERO-Plus perturbation type: native noise
against the 16 arms,
with 2 repeats per (scene, arm) on paired scenes. Reuse G1–G4 from the
[noise-mode gates](../noise-mode-icrl-20261029-r1/PLAN.md#gates-fixed-before-any-learner-training)
with perturbation type × scene as the condition. Also record determinism of
repeats and whether arms change executed actions.

- Keep a perturbation type only if G1 (oracle arm beats native) and G3 (arm
  rankings are consistent within a scene) pass. Perception failures, where no
  arm helps, drop out here.
- G2 (per-scene oracle beats the best static arm) is the in-context RL gate.
  If G2 fails for every type, a static selector solves LIBERO and the
  meta-learning question moves to Meta-World only.
- Budget to fix with the user before launch: on the order of 6,000 episodes,
  one GPU on thu2 with the LIBERO environment already built for the
  LIBERO-Goal check.

## Stage L1: references (GPU, approval needed)

1. **Per-task DSRL** following its published LIBERO procedure, on about 5
   training and 2 held-out tasks under the kept perturbations. Record success
   against environment episodes. This is the gradient-learning curve for T6
   and confirms that steering can improve these tasks at all.
2. **Static multitask DSRL**: one observation- and instruction-conditioned
   noise actor trained on all training tasks and kept perturbations, evaluated
   zero-shot on held-out trials. This is the baseline that separates
   meta-learning from generalization.

## Stage L2: cached per-attempt meta-learning

Collect success for every (task, scene, perturbation, arm, repeat) once with
the frozen motor (GPU, approval needed). Then train and evaluate on the cache
alone (CPU or one GPU, no simulator):

- RL² selector over arms, trained on training-task trials.
- DPT-style selector: predict the scene's best arm from the history.
- Thompson sampling and UCB with the same K.

This is the cheapest complete test of T1–T5, and it is the result to obtain
before any live per-chunk work.

## Stage L3: live per-chunk steering (GPU, approval needed)

Only after L2 passes T1–T5. Same interface as DSRL, history-conditioned:

- DSRL → RL²: initialize from the static multitask actor, then trial-level
  PPO with memory carried across attempts.
- Straight RL² from scratch, at the same budget.
- AD on genuine per-task DSRL learning histories from a subset of 20–30
  training tasks.
- DPT and Vintix-style noise distillation from teacher noise with annealed
  corruption. Part of the annealing must depend on reward (corruption drops
  only after success, stalls or rises after failure), and some trials must
  never improve, so the learner cannot read improvement off the attempt index.

Mixing for annealing is variance-preserving,
`eps = sqrt(1 - a) * eps_teacher + sqrt(a) * eps_random`. Before building a
corpus, measure success against `a` on a few tasks; if success jumps only near
`a = 0`, the histories carry no gradual improvement to imitate.

## Tests on held-out trials

| Test | Pass |
| --- | --- |
| T1 within-trial gain | Success at attempt K minus attempt 1 is positive, paired by scene |
| T2 memory | The same weights with memory reset every attempt lose the T1 gain |
| T3 feedback | Shuffled or zeroed rewards, and a history from a different task, lose the gain |
| T4 beyond search | Beats Thompson sampling and UCB over the same arms at equal K |
| T5 beyond generalization | Beats static multitask DSRL (zero-shot) on the same trials |
| T6 efficiency | Matches, within K attempts, the success per-task DSRL reaches only after many more episodes |
| T7 architecture | CTM, GRU and Transformer-XL memory at matched parameters and budget |

Statistics: paired scenes, trial-level bootstrap intervals, at least three
training seeds per learner, and criteria fixed before L2 training. Report
success against attempt index for every method and control.

## Meta-World companion

Meta-World supplies a hidden variable the instruction cannot reveal.

- Fine-tune pi0.5 on scripted-expert demonstrations of ML1 tasks where the goal
  position varies and is neither visible nor named. The policy then becomes
  multimodal over goals, and the initial noise selects the goal.
- First check, before any controller: success for each (goal, arm) with the
  cached arms. If outcomes do not spread across goals as the noise varies, the
  motor has collapsed to one mode and noise cannot steer it; stop there.
- Then run the same L2 cached tests and T1–T5 on ML1, and ML10 if it
  qualifies. Compare against the existing state-based ML10 protocol in
  [metaworld-ml10-matched-baige-20261028-r1](../metaworld-ml10-matched-baige-20261028-r1/RESULTS.md),
  whose task pools were goal-visible.
- Check first whether LeRobot's Meta-World support provides a starting
  checkpoint or loader.

## Relationship to existing work

- The ManiSkill noise-mode Stage 0 becomes optional. Its per-env noise
  override, outcome cache and gate analysis port to LIBERO.
- The unmerged `libero-goal-headroom` branch (instruction steering on
  LIBERO-Goal, thu2) already qualifies the LIBERO environment, checkpoint
  loading and inference conventions; reuse them.
- The pasted pilot of 78 per-task DSRL teachers on clean LIBERO-90 is replaced
  by L0 then L1 on perturbations with headroom.

## Risks

- **Support:** the 78-task motor may fail held-out tasks for every noise, so
  there is nothing to select. M0 and L0 measure this before any controller.
- **Headroom on training tasks:** the motor may already succeed on most
  training trials, leaving the controller little to learn from. Perturbations
  and LIBERO-PRO add failures there.
- **Visible perturbations:** if every kept type is visible, static multitask
  DSRL may tie (T5 fails). Prefer types where one observation cannot tell
  which noise is right.
- **RL² optimization** with images over multi-attempt trials is the most likely
  failure. Per-attempt arms in L2 avoid it; L3 is optional for the main claim.
- **Cost:** pi0.5 inference dominates collection. Cache wherever the decision
  is per attempt.

## Run identity

Planning ID `libero-noise-icrl-20261029-r1`. Each GPU stage gets its own run ID
`libero-noise-icrl-<stage>-<cluster>-<date>-r<N>` and its own worktree pinned to
a reviewed commit on `main`.
