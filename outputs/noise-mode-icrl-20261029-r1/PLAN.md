# In-context mode selection in a frozen pi0.5's noise space

Status, 2026-10-29: **Stage 0 code on `main` at `352edc4`; CPU suite passed in
thu2 job 49206. Baige S0a `job-gjkse1sb5gyj` completed 68 episodes but failed
repeat determinism (4/32 trajectories identical; three success flips). S0b/S0c
were held.** See [audited results](../noise-mode-icrl-20261029-s0-r1/RESULTS.md).
CPU stages (code, tests, cached meta-training) need no ask under
[AGENTS.md](../../AGENTS.md). The Stage 0 and Stage 2 GPU collections and the
Stage 4 live evaluation need user approval of the stated budgets before launch.
This makes [fallback idea 1](../coordination.md#1-in-context-strategy-selection-over-a-frozen-vla)
concrete.

## Question and claim

A frozen flow VLA often fails because it chose a bad mode, not because it lacks
the skill. Under a hidden condition held fixed across the K attempts of a
trial, does a history-conditioned selector over a small set of fixed noise
latents find a working mode faster than a generic bandit, using reward alone
and with frozen weights?

Target claim, if the gates pass: a meta-learned selector uses arm-success
structure shared across conditions. On held-out conditions it beats
per-condition Thompson sampling at K = 8 and closes most of the gap between
the best static arm and the per-condition oracle. The memory-wiped and
reward-blind lesions remove that gain.

This differs from the rotation PoC
([maniskill-pi05-icrl.md](../../docs/maniskill-pi05-icrl.md)) in what varies.
There, a synthetic perturbation corrupts the motor. Here, the motor is
unperturbed, the condition is a natural deployment variation, and the
learner's action is a choice the motor already supports.

## Arms

An arm is a fixed initial-noise tensor `eps_k ~ N(0, I)` of the action
expert's shape, `[action_horizon, model_action_dim]`. Each `eps_k` is drawn
once from a recorded seed and shared across all scenes and conditions. It is
used as the initial noise for **every chunk** of the attempt, so one arm is
one coherent behavioral mode over 80 steps. M = 16 arms. The default policy
is native sampling, with fresh noise every chunk.

Why a per-env override is required: `run_batch` seeds one CUDA generator per
batch, so an env's noise currently depends on its slot and on the batch
composition ([maniskill_pi05.py](../../src/fluffy_bassoon/rl/maniskill_pi05.py),
`run_batch`). An arm must mean the same tensor regardless of batch layout.
The eval sampler is the four-step Euler ODE (contract field `sampler` in the
rotation cache), so the initial noise is the only stochastic input. Stage 0
checks this.

Deferred variants, after A1 works: a structured basis (`eps = z + alpha*d_j`),
per-chunk latent choice (DSRL-like, needs live RL), and arms on the
SFT motor.

## Hidden conditions

The condition is fixed across the K attempts of a trial. Each attempt resets
to a fresh pose and overlay from the condition's pool. The learner never sees
images, proprioception or the condition. Two candidates are screened:

| | Condition | Visible to motor | Sim change | Main confound |
| --- | --- | --- | --- | --- |
| C2 (primary) | Hidden physical variant of the object: friction × mass scale on a small grid | No | ManiSkill3 per-object physical material and mass, applied after reset | Variants must change which arm works, not only lower success for all arms |
| C1 (secondary) | Object identity. Train on 0–15, hold out 16–24 | Yes | None | An observation-conditioned DSRL policy can learn object → latent without in-context RL. It is reported only with that baseline |

C2 is primary because the separation from DSRL requires a condition that
the observation does not reveal (amended 2026-10-29). C1 is screened in
the same allocation because it needs no simulator change.

## Stage 0: interface and headroom screen (GPU, approval needed)

Code (built 2026-10-29; CPU suite passed in thu2 job 49206):

1. **Noise override.** At RLinf `e1a82eb`, `Pi0RL.predict_action_batch(...,
   mode="eval", noise=...)` passes the tensor to `sample_actions`, a
   deterministic four-step Euler ODE whose initial noise is its only
   stochastic input. No RLinf patch is needed.
   `ms.run_batch(..., initial_noise, fresh_noise, issued)` passes a fixed
   `[envs, 8, 32]` tensor at every chunk. Rows flagged `fresh_noise` instead
   draw new N(0, I) noise from the seeded batch generator at each chunk;
   this is native arm -1, matching RLinf's sampling distribution but not its
   exact draws. `ms.arm_noise(seeds)` draws each arm on CPU from its own
   seed. Without `initial_noise`, the call to RLinf is unchanged.
2. **Physics variants (C2).** `ms.object_physics(PutOnPlateInScene25,
   friction_scale, density_scale)` wraps `_build_actor_helper`, so only
   source objects (`model_db_carrot`) change; plates and the arena keep
   RLinf's values. RLinf builds materials once at env construction, so each
   variant gets its own env, and the patch stays active for that env's
   lifetime.
3. **Collection.** `scripts/diagnostics/maniskill_pi05_noise_cache.py` takes
   the same contract, clean-source and frozen-motor checks as the rotation
   cache, plus `--variants friction:density,...`, `--arms`, `--repeats` and
   `--time-budget-seconds` (stops cleanly between batches). It writes per-row
   outcomes with the arm seed, noise SHA256, slot and a SHA256 of the issued
   actions, and saves the issued chunks under `trajectories/`.
4. **Gates.** `python -m fluffy_bassoon.meta_rl.noise_screen --cache-dir ...
   --condition variant|object --train ... --evaluate ...` computes G1–G4, a
   determinism report and a first-chunk mode-effect distance from saved
   artifacts only. Tests: `tests/test_maniskill_noise.py`.

Checks in the allocation:

- **Determinism.** Same scene and arm, run twice in different slots: complete
  issued trajectories match. If they do not, outcomes are Bernoulli and the
  cache needs repeats.
- **Mode effect.** First-chunk distance between arms on the same scene,
  against native-versus-native repeats. If the arms barely change actions,
  stop. The four-step ODE may be insensitive to the initial noise.
- **Physics.** Success or grasp rates differ between nominal and the most
  extreme variant on the same scenes. If they are identical, the patch did
  not take effect.

Screen: primary motor, 16 arms plus native. The three runs below share one
GPU budget of at most 2 hours, in priority order, each with
`--time-budget-seconds`. Stop and report if the first run's throughput
projects past the budget.

| Run | Variants (friction:density) | Objects × scenes | Repeats | Rows | Envs |
| --- | --- | --- | --- | ---: | ---: |
| S0a determinism smoke | 1:1 | 0,1 × 1 | 2 | 68 | 34 |
| S0b C2 hidden physics | 1:1, 0.3:1, 1:4, 0.3:4 | 0–7 × 2 | 1 | 1,088 | 34 |
| S0c C1 object identity | 1:1 | 0–7 and 16–19 × 4 | 1 | 816 | 48 |

Gate commands: S0b with `--condition variant --train 0,1,2,3 --evaluate
0,1,2,3` (leave one variant out); S0c with `--condition object --train
0:0,...,0:7 --evaluate 0:16,...,0:19`. With four variants, C2's G2 and G4
rest on only three reference conditions. A pass justifies a larger variant
grid in Stage 2, not a claim.

### Gates (fixed before any learner training)

These gates qualify selectors over this fixed random bank. They do not test
DSRL's learned continuous, observation-conditioned per-chunk noise actor.
Failure does not rule out that richer interface or establish that the frozen
motor has no steerable support outside the tested bank.

Per condition c, write `p(c, k)` for the arm-k success rate over that
condition's poses. Estimate it on disjoint pose halves: select on one half
and score on the other, so the oracle is not inflated by taking a max.

| Gate | Definition | Pass |
| --- | --- | --- |
| G1 headroom | Per-condition oracle arm minus native, scored on the held-out half | ≥ 15 pp on held-out conditions |
| G2 heterogeneity | Per-condition oracle minus the best single static arm, chosen on training conditions | ≥ 10 pp. Otherwise this bank does not establish enough benefit from condition-specific selection |
| G3 consistency | Split-half rank correlation of `p(c, ·)` within a condition | Spearman ≥ 0.5 averaged over conditions. Otherwise the arms are noise, and reward from one pose says nothing about the next |
| G4 structure | Training-condition arm profiles predict held-out profiles better than a uniform prior (log score of a posterior over training profiles) | Positive gain. This is what a meta-learner can exploit beyond Thompson |

A training-object ceiling is a known risk: the step-150 motor reports 89%
rollout success. If G1 passes only on held-out objects, train on a split of
the held-out-like objects with the lowest native success and report that
selection.

**Kill:** if G1–G3 fail on both C1 and C2, record the negative headroom result.
It is a negative result for this fixed-bank route, not a rejection of learned
noise steering or a general test of whether VLA failures are mode failures.
Do not train cached selectors on an unqualified bank.

## Stage 1: freeze the protocol

From the screen, freeze M, the condition set, the train/held-out split, the
pose pools, K (default 8) and the repeat count. Write them into a contract
file before any cache collection for training.

## Stage 2: outcome cache (GPU, approval needed)

Collect (condition, pose, arm, repeat) for all training and held-out
conditions. Sizing for C1 with deterministic outcomes: 16 training objects ×
12 poses + 9 held-out objects × 12 poses = 300 scenes × 17 = 5,100 episodes.
Recompute the time from the Stage 0 throughput and ask for the GPU budget
then. Use separate files for training and held-out conditions, with SHA256
recorded in each contract.

## Stage 3: cached meta-training (CPU)

`meta_rl/maniskill_noise.py`, following
[maniskill_rotation.py](../../src/fluffy_bassoon/meta_rl/maniskill_rotation.py):

- `NoiseArmCache`: validates a complete (scene, arm) table and drops graded
  outcomes.
- `CachedNoiseTrial(_TrialEnv)`: samples a condition, then K poses without
  replacement. The action is `Discrete(M)`, the reward is `success_once`, and
  the learner input is the previous arm, previous reward, attempt end and
  clock. A `reward_blind` flag is included.
- Full-trial PPO (gamma = GAE lambda = 1) on GRU and Transformer-XL first,
  following the coordination order: get adaptation with a strong simple
  learner, then add the CTM. Use three seeds per core and matched parameter
  caps. Add the CTM at T1 only after GRU/XL show adaptation.

Reference strategies, all scored on the same paired held-out trials:

| Strategy | Role |
| --- | --- |
| Native | Default motor |
| Best static arm (A) | Trained static selector, chosen on training conditions |
| Random arm | Floor |
| Per-condition Thompson / UCB over M arms (B) | Generic bandit, no transfer |
| Posterior over training profiles (Bayes teacher) | Structured upper reference, and the optional AD warm-start teacher |
| DSRL static (C) | Observation-conditioned noise policy trained with RL on all training conditions, weights frozen at test. Separates zero-shot generalization from in-context RL. Its one-arm-per-attempt form (a map from a frozen first-frame pi0.5 prefix embedding to an arm) runs on the cache; the per-chunk form needs live RL |
| DSRL per-condition fine-tuning (D) | The static DSRL policy updated by RL on each held-out condition with the same K = 8 episodes. The weight-update adaptation baseline, at matched interaction budget; also report it at larger budgets as a reference |
| Per-condition oracle | Ceiling |

Position against [DSRL](https://diffusion-steering.github.io/): the action
space is the same; the adaptation mechanism differs. DSRL adapts by RL weight
updates per task and conditions on the observation. This learner is
meta-trained once and adapts from reward within K attempts with frozen
weights. The claim must be about exploration efficiency under conditions
the observation does not reveal, not about latent steering itself.

A best-of-N verifier baseline (RoboMonkey-style) is not available. The
checkpoint's value head reads only the VLM prefix (`value_after_vlm`), so it
cannot rank noise arms. It needs an action-conditioned critic, which is a
later baseline.

Learner lesions: memory wiped every attempt and reward-blind, both
greedy and sampled.

## Stage 4: live check (GPU, approval needed)

Run the frozen best-seed learner in the closed loop on held-out conditions,
along with native, the best static arm and Thompson. Use 64 paired trials per
condition at K = 8 if the budget allows, otherwise at least 128 trials in
total. Live results must fall within the cache estimate's 95% CI. Otherwise
the cache was a shortcut and the cached results are not reported.

## Success criteria (held-out conditions, cache, then live)

1. Normal minus memory-wiped ≥ 5 pp mean success over K, with a paired
   bootstrap 95% CI excluding 0, for at least one core across three seeds.
2. Normal > per-condition Thompson at K = 8. The CI on the difference
   excludes 0.
3. (Normal − best static) / (oracle − best static) ≥ 0.5.
4. Normal > DSRL static (C) and > DSRL per-condition fine-tuning (D) at
   K = 8. Otherwise the result is zero-shot selection or ordinary RL, not
   in-context RL.
5. The reward-blind lesion is no better than the best static arm.

Report success by attempt index, cumulative regret against the oracle, and
the costs: cache episodes, meta-training updates, CPU and GPU hours.
Architecture comparisons are descriptive unless the parameter, state and
compute matching is reported.

## Risks

- **Insensitive noise.** The four-step ODE may collapse distinct initial noise
  to similar actions, so G1 fails. Mitigation: larger-norm structured
  latents (a deferred variant), and report the sensitivity curve.
- **Ceiling.** A strong motor leaves no headroom on training objects (see the
  G1 note).
- **No shared structure.** If G4 fails, the best possible learner equals
  Thompson. Then the paper becomes the headroom finding plus a bandit result,
  not meta-learning.
- **Cache realism.** Deterministic outcomes give an exact table but
  understate deployment noise. Stage 4 checks this.

## Run identity

Run ID prefix `noise-mode-icrl-20261029`. Use new suffixes per stage (for
example `-s0-r1`), each pinned to a reviewed `main` commit in a detached
worktree. Record cluster, job, commit and receipt paths in
[coordination](../coordination.md) and verified results in
[EVALUATION.md](../../src/fluffy_bassoon/meta_rl/EVALUATION.md).
