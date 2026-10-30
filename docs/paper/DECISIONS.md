# Paper decisions and claim boundaries

Durable research choices; no run history. Paper outline:
[DRAFT.md](DRAFT.md). Verified scores belong in the method's `EVALUATION.md`;
raw evidence and dated plans belong in `outputs/`; live state belongs in
`outputs/coordination.md`.

## Agreed design

- Title: **Learning to Steer: In-Context Reinforcement Learning for Frozen
  VLAs**. A working title, not evidence that ICRL already works.
- Main question: high robotic success with few deployment interactions and
  competitive final performance. Meta-RL/ICRL are means to that outcome.
- Primary motor: official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`,
  config `pi05_libero`, native horizon ten. Documented release fine-tuning:
  LIBERO-Spatial/Object/Goal/10, 40 tasks / 1,693 demonstrations. This is not
  an exhaustive foundation-pretraining audit; no new VLA SFT is required.
- Prefer the already staged official release, receipt
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
  Public HF `lerobot/pi05_libero_base`, revision
  `5fcf06fc42b6717e22ab885fd381a096574b12c7`, is an alternative candidate;
  no binary equivalence to the official JAX release is asserted.
  [Provenance](../../outputs/pi05-libero40-provenance-20261029-r1/PROVENANCE.json).
- **Meta-training source only LIBERO40; targets LIBERO90, Plus and Pro.**
  The [source/target protocol](../../outputs/libero40-meta90-20261029-r1/PROTOCOL.md)
  governs this arm. The RLinf130 motor, its source data and earlier data-arm
  comparisons do not carry into it.
- Steering: 32 bounded noise coordinates broadcast over the frozen motor
  horizon. Keep independent native noise and tied-noise controls separate.
- One frozen motor/tokenizer/normalization/feature receipt across all tasks.
  Requalify the OpenPI40 backend on the actual allocation. RLinf4104 features
  and learned controllers are incompatible receipts, not reusable coordinates.
- Planned source route: actual advancing DSRL histories with paired AD and
  DPT-style final-query labels. Historical executed actions and feedback stay
  actual. Source improvement and final-actor expertise require independent
  evaluation; update counts alone prove neither.
- Fixed-teacher CND, direct RL²/PPO and supervised warm-start plus RL remain
  separate candidate conditions, with their own source cost and contracts.
- Primary one-shot: **one complete support episode → a separate query
  episode**, fixed motor/controller weights. Both episodes' interactions
  count. Report query success separately from support success.
- A trial fixes goal/instruction and actual scene/perturbation case; memory
  persists from support to query. New trial → empty context. Different cases
  make different trials. Case/split metadata is never a controller input.
  See [META-TASK.md](../frozen-vla-libero/META-TASK.md).
- Causal Transformer is the current controller choice. A strong static
  Transformer establishes policy transfer, not implicit deployment-time ICRL.

## Source and target boundaries

- Use all 40 documented release tasks as source and the full LIBERO90, Plus
  and Pro suites as targets. No semantic-goal filtering or prior-development
  exclusions. Declare disjoint source training/local-validation cases; whole
  cases/trials stay together before window extraction.
- Many cases per task means actual object/robot placements and conditions,
  not just many RNG seeds. Same saved initial state plus a new seed need not
  change placement. Record actual states and stochastic observations.
- Thirty-two cases per task is a candidate data scale, not a known sufficient
  amount or approved collection budget. Measure source-data scale and teacher
  learning length separately from deployment-shot efficiency.
- No LIBERO90/Plus/Pro teacher collection, replay, labels, meta-gradients or
  normalization in this source40 arm. Older 90:57/90:47 and Plus 2387/1438
  histories remain excluded even when their goals alias a source40 goal.
- Claim full-suite transfer, not that every LIBERO90 task is semantically unseen.
  Source-goal overlap and source-familiar Plus/Pro goals may remain. Suite
  names alone do not establish new goals or strict distribution separation.
- Prior 90:1/2/47/57 inspection, Plus 667/290/2387/1438 and Pro development
  variants do not exclude them from full-suite evaluation. Disclose prior
  inspection. Existing semantic audits are archival only; stop further
  admission auditing and do not gate execution on those artifacts.
- No LIBERO90/Plus/Pro histories enter source training, even though their
  full evaluations include previously inspected tasks and variants.
- Separate zero-support and 1/2/4/8-support query endpoints. Earlier K total
  attempt conventions cannot be relabeled as support-shot counts. Official
  first-attempt scores are another reporting protocol; disclose settings,
  coverage and prior inspection. No blind excluded score is required.
- No final-target feedback for curriculum, method/checkpoint selection or
  corpus generation. Previously inspected target cases remain disclosed
  within the full-suite results.
- Prior [78/10/10 split](../../outputs/libero-pi05-corpus-20261029-r1/split.json)
  with exclusions 90:38/77 and
  [clean+Plus source arm](../../outputs/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md)
  remain immutable historical protocols. LIBERO10 is source in the new arm;
  the old source-state 0–39/validation 40–44/test 45–49 rule is not inherited.
- Historical clean-only→Plus/Pro claims concern unseen benchmark shifts for
  the attachment; historical clean+Plus→Plus claims concern within-Plus
  held-out cases. Pro was absent from both source arms. These distinctions
  explain earlier evidence; they are not new source40 data arms.

## Necessary comparisons and accounting

- Native, tied noise, ordinary official DSRL, simple declared noise search;
  cold and source-pretrained DSRL distinguished and interactions charged.
- Independently train static and episode-reset controllers with identical
  source trials, labels, initialization and optimization budgets. Resetting
  the full-trained model only at inference is a separate intervention.
- Static resets each decision and lacks past feedback. Episode reset removes
  cross-episode history while preserving within-episode memory. Neither
  control alone proves a mechanism; compare full-history query outcomes.
- Official PixelSAC uses RGB/state; history/static use qualified OpenPI40
  prefix features. Same raw source data does not isolate representation or
  architecture. Feature-matched SAC addresses a separate representation
  question; the matched static student isolates context use.
- Source-matched replay-only SAC pretraining is an additional control, not
  the online source-teacher recipe. It adds optimizer cost, not new rollouts.
- Reward/history interventions and clock/no-history controls probe feedback
  use. Reward shuffling removes numeric-reward information, not necessarily
  visual feedback; no single intervention is a universal ICRL proof/gate.
- MLP is not intrinsic to DSRL. Architecture attribution requires matching
  optimizer, critics, objective, features, data and budget.
- Report support success, separate query success, primitive interactions,
  task/case strata and trial/task-clustered uncertainty. At-least-once success
  can increase through repeated stochastic attempts without adaptation.
- Count unique source interactions once across paired objectives; separate
  source training/validation, teacher/controller compute and deployment
  support/query costs. Include competence checks and direct meta-RL rollouts.
  Report deployment efficiency and amortized total cost; pretraining is not free.

## Hypotheses and open choices

- Source40 motor competence may already leave little useful headroom. First
  establish diverse native failures that alternate steering can recover.
  Neither many easy successes nor many irrecoverable failures suffice.
- Source teachers need not finish in 4/8 episodes. Published DSRL uses batch 256,
  replay warmup exceeding 500 query chunks, UTD20; a fresh K4 run cannot reach
  that warmup. Declare and validate any shortened schedule rather than call
  earlier warmup 0/batch 32 smoke histories the published recipe.
- Source teacher competence, source scale, optimization schedule and resource
  budget remain open; no giant corpus launch is authorized by a paper update.
  If genuine case diversity still lacks useful headroom, revise task/benchmark
  conditions before large collection. Artificial hidden physics is not assumed.
- A goal-revealing instruction need not reveal the best steering coordinate.
  Test whether feedback supplies value beyond observation-only control.
- Scheduled CND can learn a clock shortcut; reward-dependent corruption is
  only an unadopted ablation. Square-root mixing does not preserve a Gaussian
  law for arbitrary bounded/selected teacher coordinates.
- DPT is not observation-only BC by definition. Context dependence is
  empirical; final-candidate labels are not called expert without evaluation.
  Matching a simple bandit weakens a complex-controller contribution without
  negating the definition of meta-learning.
- Central hypothesis: reusable history-to-steering training improves separate
  query success with few supports, preserves high final performance and earns
  its full source cost. **Not supported by the first source40/target90 waves:** supervised and direct PPO controllers are below their contemporaneous native controls; full shifted-suite and multi-shot comparisons remain incomplete.
- Freeze training seeds, stopping rules, target cases, success margins and
  uncertainty procedure before final comparison. No outcome-selected final
  failures or synthetic hidden condition introduced to force a memory gain.
- [VLRLA](https://arxiv.org/html/2610.19517v1) already studies fixed-parameter
  VLA ICRL and a frozen-expert bridge; DSRL supplies latent-noise steering.
  Broad first/novelty claims and venue acceptance claims remain unsupported.
- Historical one-family RLinf130 smoke: clean 24/24, Plus source 21/24 and native
  Plus/Pro development 8/8 each. Ceiling/non-monotonic results do not establish
  an improving expert, broad task generalization or useful memory. Earlier
  OpenPI40 native 8/8 + tied 8/8 checks are runtime/headroom evidence only.
  These scores stay in the [evaluation ledger](../frozen-vla-libero/EVALUATION.md),
  outside the new experiment's result slots.
- Source40 runtime qualification is complete on one task: train59/60 and
  validation68/68 success, 1,223 actual chunks and4,620 source SAC updates.
  Paired AD/DPT exports and full recoveries are verified. This is execution
  evidence; it does not establish an improving teacher or meta-controller.

- User selected an initial1M executed-action allowance per benchmark, shared
  across SAC, fresh PPO, validation, settling and retries. All40 LIBERO source
  training goals and six RoboTwin source goals remain. Source-local LIBERO
  validation uses eight fixed goals on disjoint states. This original shared cap applied to the initial pilot. Later human steering
  accepted roughly1M source actions per learner, without increasing the32-hour
  LIBERO compute pool; existing runs retain their original contracts. Neither
  quantity is a proven sufficient meta-learning dataset.

- Execution order: after minimal runtime and saved-data qualification, overlap
  collection, fitting and frozen query evaluation on immutable completed
  batches. Keep ordinary target-online DSRL and fresh PPO baselines moving
  while offline students train. Unexpected outcomes get a written diagnosis
  and a bounded discriminating test before repeating the same recipe at scale.
