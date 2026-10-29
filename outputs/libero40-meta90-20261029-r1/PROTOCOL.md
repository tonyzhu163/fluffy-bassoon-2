# LIBERO40 source, LIBERO90 and shifted-target meta-learning

Status: user-selected full-suite design, 2026-10-29. No semantic filtering. Previous source data,
splits and runs stay immutable. Corpus scale and resource budget are not yet
a launch contract.

## Frozen backbone

- Official OpenPI `gs://openpi-assets/checkpoints/pi05_libero`.
- Release README identifies `pi05_libero` training config; that config uses
  `physical-intelligence/libero`: 40 tasks / 1,693 demonstrations.
- Suites: Spatial, Object, Goal, LIBERO10, ten tasks each. This describes
  release fine-tuning; it is not an exhaustive foundation-pretraining audit.
- Already staged on Baige; receipt SHA256
  `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.
- HF candidate `lerobot/pi05_libero_base`, revision
  `5fcf06fc42b6717e22ab885fd381a096574b12c7`; public, ungated.
  No binary-equivalence claim to the JAX release. Prefer the already staged,
  qualified original release for the next qualification.
- Freeze motor during source collection, controller training and deployment.
  Requalify the existing OpenPI runtime on the actual allocation. RLinf130
  data, controller weights and feature contracts do not carry into this arm.

## Meta-training cases

- Source goals come only from the 40-task suites. No LIBERO90 teacher data
  enters this arm, including the older task57 teacher or recent task47 data.
- Many genuine cases per semantic task. Track actual object/robot initial
  state and perturbation identity, not only a numeric RNG seed.
- Different cases supply different trials. Within a controlled trial,
  preserve the case across retry episodes and reset controller memory only
  at trial boundaries. Record stochastic observation variation explicitly.
- Keep source training and local validation cases disjoint; all histories
  from a case remain together. Freeze exact state/case lists after audit.
- Thirty-two cases per task is a candidate scale, not a claimed sufficient
  amount or an approved collection budget. Test several data scales rather
  than asserting that a particular number guarantees meta-learning.
- A source SAC learning history can require far more than 4/8 episodes.
  Deployment shot budget and source-teacher training length are different.
- Published DSRL's released pi0 script uses batch256 and replay warmup
  exceeding500 query chunks, then UTD20. A fresh K4 source run cannot
  reproduce that schedule. Qualify source improvement and final actor
  competence on independent source-development cases before calling labels
  expert or scaling AD/DPT histories.
- AD preserves executed chronological actions/outcomes. DPT-style query
  labels may use a qualified final actor; actual contexts are never rewritten.
  Train a separate static control on identical trials/features/labels.

## Target roles

- Meta-training: all40 original tasks in Spatial/Object/Goal/LIBERO10.
- Evaluation: the complete90-task LIBERO90 suite, plus LIBERO-Plus and
  LIBERO-Pro. No goal-family exclusions or prior-development task exclusions.
- User explicitly requested no further semantic auditing. Earlier audit files
  are archival evidence, not an admission or launch gate for this protocol.
- Report the full-suite result as source40-to-target90 transfer. Do not assert
  that every target task is semantically novel. Plus/Pro source-familiar goals
  test condition transfer rather than automatically establishing new-goal transfer.
- No target90/Plus/Pro trajectories enter source pretraining. Dataset/seed
  roles and source/deployment costs remain explicit.
- Existing78/10/10 and earlier clean/augmented data protocols stay immutable.
  They do not control the new source40 arm; LIBERO10 is now source data.

## Shots, controls and decision

- Zero-shot: empty history, first attempt.
- One-shot: one support attempt with actual observations/actions/rewards,
  followed by a separate query episode. Motor/controller parameters stay fixed.
- Also evaluate2/4/8 support attempts; support interactions count toward
  deployment cost. Query success is not pooled with support success.
- Match target cases and support budgets across full history, trained static,
  episode-reset lesions, simple search, cold DSRL and source-pretrained
  DSRL controls. Distinguish pixel-SAC features from VLA-prefix features.
- Source-matched replay-only SAC pretraining is an additional control, not
  a reproduction of the online source-data-generation recipe.
- First establish native failures that alternative steering can recover.
  Many easy successes or many irrecoverable failures do not establish a
  useful meta-learning population. No fixed failure percentage is sufficient.
- Official pi05 is strong on its40 training tasks; switching checkpoints
  does not itself guarantee informative source failures. If diverse cases
  still lack useful steering headroom, revise benchmark/task conditions
  before expensive corpus collection. No artificial hidden physics is assumed.
- Report unique source interactions, teacher compute, controller compute,
  deployment support/query interactions and success separately.

Provenance snapshots: `../pi05-libero40-provenance-20261029-r1/`.
Primary scale audit: `../../docs/frozen-vla-libero/PRETRAINING-SCALE.md`.
