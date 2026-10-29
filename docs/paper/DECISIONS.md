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
- Primary backbone: frozen RLinf π₀.₅ LIBERO130 fullshot SFT release,
  revision `6222623f635769bfc73c9472e29fab9b7fd8e027`; native horizon ten.
  Original-130 base-task exposure disclosed. Older OpenPI/π₀ teachers and
  feature datasets cannot be silently migrated to this motor.
- Evaluation suites: **LIBERO-Plus and LIBERO-Pro**, using the same held-out
  cases for two approved data arms: clean original LIBERO and clean plus a
  predeclared Plus source partition. The v1 original-task holdout remains an
  auxiliary attachment-only transfer test; its frozen artifact is unchanged.
- Steering interface: 32 bounded initial-noise coordinates broadcast over
  the frozen motor horizon; native full-noise and tied-noise references.
- Shared public representation: one frozen motor receipt, tokenizer,
  normalization assets and backend-specific canonical feature coordinates across tasks.
  No per-task drifting source features in a pooled student corpus.
- Current source route: actual fresh-DSRL learning trials, paired AD and
  DPT-style final-candidate query labels. Final actors are not assumed experts.
  Fixed-teacher continuous noise distillation remains a separate candidate
  route requiring its own competence evidence. Actual historical noise and
  feedback are never replaced by query labels.
- Separate methods: AD from real advancing source policies; direct RL²/PPO;
  any supervised warm-start + RL condition named and charged separately.
- Deployment: frozen weights, empty initial context, own observations,
  actions and actual rewards; memory persists across attempts in a trial.
- Causal Transformer is the current controller choice. Pretraining a static
  Transformer is policy transfer; architecture alone does not establish ICRL.
- Meta-task = goal/instruction plus scene/perturbation condition, repeated
  for K attempts. Outer training learns a history-to-steering rule; deployment
  changes context, not its weights. Defining this trial does not prove useful
  adaptation. See [META-TASK.md](../frozen-vla-libero/META-TASK.md).

## Frozen data boundary

- [Manifest](../../outputs/libero-pi05-corpus-20261029-r1/split.json):
  78 training / 10 validation / 10 official test tasks + two excluded
  LIBERO-90 auxiliary test tasks; deterministic semantic-family split.
- Exclude 90:38 and 90:77 due held-out full-goal equivalence. Keep official
  LIBERO-10 exactly ten. Shared scenes and atomic skills permitted;
  unseen full tasks/goals are the primary transfer claim.
- Fifty initial states verified per canonical task. Training-task source
  bank 0–39; auxiliary familiar-task validation 40–44; test 45–49.
- Entire multi-episode trials inseparable. Task split precedes scene split
  and training windows. Local `validation` scenes of a training task do not
  become primary held-out-task validation.
- No final-test task in source teacher training/collection, student
  gradients, normalization, curriculum or checkpoint selection. Validation
  teachers/data remain isolated; test-time online DSRL is evaluation only.
- Plus/Pro are primary external tests, with base-family lineage. Training-base
  perturbations measure robustness; held-out-base perturbations additionally
  measure task transfer. No final-test feedback for method selection.
- Distribution claim by source arm: clean-only → Plus/Pro concerns benchmark
  shifts unseen by the attachment. With Plus source training, held-out Plus
  cases are within-Plus generalization; stronger OOD claims require declared
  factor/parameter/base-family exclusions. **No Pro pretraining in either arm**.
- Plus training cases2387/1438 are disjoint from already used development
  cases667/290. Same clean validation and final Plus/Pro cases in both arms;
  additional Plus collection cost reported. The [r3 declaration](../../outputs/libero-pi05-corpus-20261029-r3/PLUS-SOURCE-PLAN.md)
  versions this approval; earlier protocols stay immutable.
- The current Plus source categories are lighting/sensor noise; current
  development categories are camera/robot initialization. Describe that
  narrow category split explicitly; sharing the Plus name does not make
  every held-out parameter/family in distribution or make the whole suite unseen.
- Tiny development subset fixed before outcomes: original object task 1,
  cream cheese → basket, matching training family 90:47; two official Plus
  cases and two Pro cases. This family is reserved from blind final tests.
  Original task 90:57 has no official Plus perturbation counterpart.
- Keep first-attempt official-style scores separate from K1/2/4/8 adaptation;
  record suite settings and coverage. Full official aggregates that include
  development cases require an overlap disclosure and a blind excluded score.
- [Evaluation protocol v2](../../outputs/libero-pi05-corpus-20261029-r2/EVALUATION-PROTOCOL.md)
  records exact development cases/source pins and final-case exclusions.
  It does not adopt a full 130-teacher collection or rewrite the v1 split.
- Released VLA pretraining may include benchmark tasks. Claim only holdout
  from attachment/source-teacher training unless independently established.

## Necessary comparisons

- Native, tied noise, ordinary official DSRL; pretrained and online steering
  distinguished; source and deployment interaction costs reported.
- Feature-matched SAC; independently trained static and episode-reset
  controllers; full-model inference reset as a separate intervention.
- Source-matched official PixelSAC uses RGB/state; history/static use the
  shared RLinf4104 features. Same raw data does not isolate architecture or
  representation. The matched static student is the current history control;
  a feature-matched SAC would answer a separate representation question.
- Reward/history interventions and clock/no-history controls before claiming
  feedback-driven cross-attempt adaptation.
- MLP is not intrinsic to DSRL. Architecture attribution requires matched
  optimizer/critic/objective/input/data/budget, not PPO-versus-SAC results.
- Episode reset preserves within-episode memory; reset every decision is
  the static comparator. Neither result alone proves memory benefits.
- \(K=1,2,4,8\); success versus primitive interactions and final success;
  trial/task-clustered uncertainty; training seeds and stopping rules
  specified before final evaluation.
- Full meta-pretraining/source cost, shared source accounting and amortized
  deployment cost. Validation imitation loss is a diagnostic, not robot proof.

## Hypotheses and unresolved choices

- Before a large teacher corpus: finish the bounded strong-backbone pilot;
  predeclare training/dev steering headroom and repeatability measurements;
  then test a small corpus against static/reset and simple bandit/search
  controls. Large collection remains deferred until these checks justify it.
- A goal-revealing instruction need not reveal the best noise coordinate.
  Artificial hidden physics is not necessary and is outside the agreed scope.
- Scheduled CND can teach a clock shortcut. Reward-dependent corruption is
  only an ablation candidate; no adoption without evidence. Square-root
  mixtures do not preserve a Gaussian law when teacher coordinates are
  bounded, selected or otherwise non-Gaussian.
- DPT is not BC by definition; contextual versus observation-only behavior
  is empirical. Matching a simple bandit weakens a complex-controller
  contribution without negating the meta-learning definition.
- Reward shuffling probes numeric-reward dependence, not all feedback use;
  visual outcomes may remain informative. No single reward/history/bandit
  intervention is a universal ICRL gate or sufficient proof by itself.
- Same initial-state index + different RNG seed does not imply a new object
  placement. Verify actual resets and distinguish distinct states/repetitions.
- Hypothesis: pretraining across useful histories yields a reusable process
  that outperforms ordinary DSRL at low deployment budgets without losing
  final success. **Untested on the planned held-out task bank.**
- Hypothesis: paired DPT and CND teach different useful behavior despite
  identical historical contexts. **Outcome TBD.**
- Hypothesis: reward-conditioned cross-attempt memory adds value over matched
  static/reset policies. **Outcome TBD.**
- Pretraining data scale, controller capacity, context length, optimizer
  schedule, source competence distribution and deployment headroom: measure
  on training/validation tasks; final values still open.
- Ordinary DSRL held-out initialization, resource/seed plan, success margins,
  final uncertainty procedure and stronger-suite protocols: prespecify before
  full comparison. No test-selected task difficulty or synthetic hidden
  condition introduced merely to force a memory advantage.
- [VLRLA](https://arxiv.org/html/2610.19517v1) already studies fixed-parameter
  VLA ICRL and a frozen-expert bridge. DSRL already supplies noise steering.
  Broad novelty claims excluded; contribution depends on a verified method,
  controlled robot-performance evidence and a complete related-work audit.
- Runtime checks, native ceiling probes and non-VLA engineering pilots are
  not evidence for the central adaptation claim. Do not promote their
  success counts into meta-learning result tables. Verified quantitative
  evidence: [LIBERO evaluation ledger](../frozen-vla-libero/EVALUATION.md).
- Current corpus/probes cover one cream-cheese-to-basket goal family only.
  Clean source success ceilings and non-monotonic Plus source episodes provide
  no demonstrated improving teacher; the native development probes are also
  at their success ceiling. Broad new-goal/meta-task generalization and useful
  context dependence remain unproven. Final candidates are not called experts
  without independent evaluation.
