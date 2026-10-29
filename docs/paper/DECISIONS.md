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
- Primary backbone: released frozen π₀.₅ on standard LIBERO; native action
  horizon ten. π₀ teachers/checkpoints are incompatible with π₀.₅ source
  training and cannot be silently migrated.
- Steering interface: 32 bounded initial-noise coordinates broadcast over
  the frozen motor horizon; native full-noise and tied-noise references.
- Shared public representation: one frozen motor receipt, tokenizer,
  normalization assets and canonical feature-v2 coordinates across tasks.
  No per-task drifting source features in a pooled student corpus.
- First source-data route: competent fixed teacher + continuous noise
  distillation. Paired DPT changes only query targets on the same actual
  histories. Source noise and feedback remain the values actually executed.
- Separate methods: AD from real advancing source policies; direct RL²/PPO;
  any supervised warm-start + RL condition named and charged separately.
- Deployment: frozen weights, empty initial context, own observations,
  actions and actual rewards; memory persists across attempts in a trial.
- Causal Transformer is the current controller choice. Pretraining a static
  Transformer is policy transfer; architecture alone does not establish ICRL.

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
- Plus/Pro locked for later tests, with base-family lineage. Training-base
  perturbations measure robustness; held-out-base perturbations additionally
  measure task transfer. No final-test feedback for method selection.
- Released VLA pretraining may include benchmark tasks. Claim only holdout
  from attachment/source-teacher training unless independently established.

## Necessary comparisons

- Native, tied noise, ordinary official DSRL; pretrained and online steering
  distinguished; source and deployment interaction costs reported.
- Feature-matched SAC; independently trained static and episode-reset
  controllers; full-model inference reset as a separate intervention.
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
  success counts into meta-learning result tables.
