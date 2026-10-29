# In-context noise steering of a frozen pi0.5 on LIBERO-Plus/PRO and Meta-World

Status, 2026-10-29: **Meta-World and LIBERO-Plus/PRO released-motor screens coded and
CPU-tested; launches delegated to coordination; no GPU runs yet.** Released RLinf motors first; our own motors later (see Motor). CPU work (code, tests,
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

Meta-World runs first because a hidden goal makes a static shortcut impossible
and its check is cheap; LIBERO L0 runs alongside it (concurrency approved by the
user, 2026-10-29; budgets fixed at launch). Both use released motors, so no
training precedes the first result.

1. Meta-World screen with the released motor: goal marker visible (headroom)
   and hidden (mode spread), `scripts/diagnostics/metaworld_pi05_noise_cache.py`.
2. In parallel, LIBERO L0 with the released LIBERO motor on LIBERO-Plus.
3. Meta-World cached tests, T1–T5.
4. LIBERO L1 and L2; L3 only if warranted.
5. Later: our own motors (Stage M0 for LIBERO; a goal-hidden Meta-World
   fine-tune if the hidden-goal mode spread fails).

If the hidden-goal mode spread fails, fine-tune the Meta-World motor on
goal-hidden demonstrations before concluding that noise cannot steer it.

## Motor and interface

| Item | Choice |
| --- | --- |
| Motor, first | Released RLinf pi0.5 LIBERO SFT, frozen; pick the few-shot `RLinf-Pi05-LIBERO-SFT` or `RLinf-Pi05-LIBERO-130-fullshot-SFT` in L0 by headroom |
| Motor, later | Our own SFT of the pi0.5 base on the 78 training tasks only (Stage M0); draft code on branch `motor-sft-draft` |
| Inference | RLinf `e1a82eb` `pi05_libero` conventions, as in `src/fluffy_bassoon/rl/libero_goal.py` on branch `libero-goal-headroom`: 256 px views, 8-D state, 5 Euler steps, horizon 10 with 5 executed, 15 settle steps |
| Steering action | The initial flow noise `[10, 32]` of a chunk. Per-attempt arms (one fixed tensor for every chunk of an attempt, M = 16, as in the noise-mode plan) first; per-chunk noise (DSRL actor) later |
| Feedback | Binary success at attempt end and an attempt-boundary flag. No shaped reward |

The released motors trained on all 130 LIBERO tasks, including LIBERO-10, so
with them the held-out tasks are new to the controller only. Out-of-distribution
conditions for the motor come from LIBERO-Plus and LIBERO-PRO: the controller
meta-trains on training tasks (clean and kept LIBERO-Plus types) and is tested
on held-out tasks under held-out perturbation types and on LIBERO-PRO. With our
own M0 motor, "held out" applies to the whole system; verify whether the pi0.5
base pretraining mixture contains LIBERO data.

## Splits and perturbations

- Tasks: the audited split of 78 training, 10 validation and the official 10
  held-out (LIBERO-10) tasks; the two LIBERO-90 tasks whose goals duplicate
  held-out tasks stay excluded. Frozen in
  [split.json](../libero-pi05-corpus-20261029-r1/split.json), loaded and
  hash-checked by `src/fluffy_bassoon/libero_split.py`; initial states 0–39 of
  training tasks for training trials, 40–44 and 45–49 reserved.
- Perturbations: redo the audit at the level of task × perturbation. Every
  LIBERO-Plus or LIBERO-PRO variant of a held-out or excluded task is held out.
- Training uses LIBERO-Plus perturbation types that pass L0. Evaluation uses
  held-out tasks under those types, then LIBERO-PRO and any held-out
  perturbation types as a transfer test.
- A **trial** is one task, one scene and one perturbation held fixed for K = 8
  attempts (K = 16 for extrapolation). Each attempt resets to the same scene.
  Memory resets only between trials. Weights are frozen at evaluation.

## Stage M0 (later): train the motor on the 78 training tasks (GPU, approval needed)

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

No training. With the released motor (later also the M0 motor), for about 6 training and all 10 held-out
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

### LIBERO-Plus/PRO contents and the L0 screen

Both packages perturb LIBERO-10, Spatial, Object and Goal; their LIBERO-90 suites
are the originals. So the 78 training tasks have **no** Plus/PRO variants, and every
Plus/PRO variant of LIBERO-10 belongs to a held-out task. Per held-out task:

- LIBERO-Plus (`rlinf-liberoplus`, 2,519 LIBERO-10 variants): camera 419, robot
  initial state 393, sensor noise 449, language 383, light 274, background 289,
  layout 173, level 139; at least 4 of each type for each of the 10 tasks.
- LIBERO-PRO (`rlinf-liberopro`): language, object, position (swap), task and
  distractor variants, one per task per directory (60 distractor files).

Training conditions for the controller therefore cannot be LIBERO-Plus types on
training tasks. Options, to decide after L0: train on Spatial/Object/Goal
variants (outside the frozen split, so audit their goals against LIBERO-10
first), or apply the name-encoded Plus perturbations (camera, robot initial
state, sensor noise) to LIBERO-90 training tasks, since the Plus env parses
them from any task name.

The L0 screen (`scripts/diagnostics/libero_variants_noise_cache.py`,
`src/fluffy_bassoon/rl/libero_variants.py`) uses the released
`RLinf-Pi05-LIBERO-130-fullshot-SFT` (SHA256 `4d9089c9…6529d6683`, as qualified
on branch `libero-goal-headroom`). Per package it picks 2 variants of every
perturbation type, spread over held-out tasks, plus 2 clean base tasks, and
runs 4 initial states × (native + 8 arms): about 650 Plus and 430 PRO episodes
of at most 520 steps. Per type, `gates.json` reports native and best-arm
success and G1–G4 with variants as conditions.

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=<allocated> python scripts/diagnostics/libero_variants_noise_cache.py \
  --kinds plus,pro --model-path <RLinf-Pi05-LIBERO-130-fullshot-SFT> \
  --norm-stats <physical-intelligence/libero/norm_stats.json> \
  --expected-model-sha256 4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683 \
  --per-category 2 --scenes 4 --arms 8 --num-envs 16 --time-budget-seconds 21600 --output <run>/l0
```

Needs RLinf `e1a82eb` with `rlinf-libero==0.1.3`, `rlinf-liberoplus>=0.1.3`,
`rlinf-liberopro` and their assets (`liberoplus-download-assets`,
`liberopro-download-assets`, as RLinf's installer does). CPU tests:
`tests/test_libero_variants.py`; its real-package test must pass in that env.

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

- Motor: released `RLinf/RLinf-Pi05-MetaWorld-SFT` (revision `60b2d03`),
  few-shot SFT on `lerobot/metaworld_mt50`; RLinf reports 43.8% MT50 success.
  It sees one 480 px camera with the goal marker visible, a 4-D hand state and
  RLinf's task prompt; `src/fluffy_bassoon/rl/metaworld_pi05.py` reproduces
  RLinf's evaluation conventions.
- Conditions: four discrete goals per task (reach-v3, push-v3), at least 0.1
  apart, so a success identifies the goal. Each scene fixes the object.
- Screen, no training: success for each (goal, scene, arm, repeat) with the
  marker **visible** (headroom on a seen task) and **hidden** (site alpha 0;
  the goal is then absent from image, state and prompt; a CPU check confirms
  no pixel changes with the goal). Gates G1–G4 with goals as conditions, plus
  goal coverage: every goal reached by some arm and arms preferring more than
  one goal.
- If hidden-goal coverage fails, the released motor has collapsed to one mode
  without the marker. Then fine-tune it briefly on goal-hidden scripted
  demonstrations (draft on branch `motor-sft-draft`) and rerun the screen.
- Then run the same L2 cached tests and T1–T5 on ML1, and ML10 if it
  qualifies. Compare against the existing state-based ML10 protocol in
  [metaworld-ml10-matched-baige-20261028-r1](../metaworld-ml10-matched-baige-20261028-r1/RESULTS.md),
  whose task pools were goal-visible.

### Running the Meta-World screen (GPU, approval needed)

Needs an environment with RLinf `e1a82eb` (its vendored openpi) plus
`metaworld==3.1.1`, the RLinf snapshot as `--rlinf-root` (its
`toolkits/standalone_eval_scripts/openpi` loader) and the downloaded checkpoint.
Inside an allocated single-GPU step (`FLUFFY_GPU_EXECUTION=slurm` or `baige`):

```bash
python scripts/diagnostics/metaworld_pi05_noise_cache.py --rlinf-root <RLinf-e1a82eb> \
  --model-path <RLinf-Pi05-MetaWorld-SFT> --tasks reach-v3,push-v3 --visibility visible,hidden \
  --scenes 4 --arms 16 --repeats 2 --time-budget-seconds 10800 --output <run>/s0
```

That is 2 tasks × 2 visibilities × 4 goals × 4 scenes × 17 arms × 2 repeats =
2,176 episodes of at most 160 steps; throughput is unmeasured, so run reach-v3
alone first and size the rest from it. Each `<task>-<visibility>/gates.json`
holds G1–G4, determinism, mode effect and goal coverage; `summary.json`
collects them. CPU tests: `tests/test_metaworld_pi05.py`, `tests/test_libero_split.py`.

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
