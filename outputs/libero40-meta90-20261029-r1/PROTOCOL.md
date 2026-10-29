# LIBERO40 source, LIBERO90 and shifted-target meta-learning

Status: user-selected design, 2026-10-29; goal audit complete. New protocol; previous source data,
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

| Target | What it tests | Required exclusions |
| --- | --- | --- |
| LIBERO90 | New task families for the controller; also absent from the documented 40-task fine-tuning set | Audit full goals/bindings against all40 source tasks; separate exact semantic duplicates; reserve inspected development tasks |
| LIBERO-Plus | New perturbations of source-familiar tasks unless the goal is separately held out | Keep target cases/perturbation families out of source collection and selection |
| LIBERO-Pro | New benchmark conditions; task novelty depends on the base goal | No Pro source histories; separate development and blind cases |

- Previously inspected LIBERO90 IDs1,2,47,57 are development, not blind tests.
  Exact aliases may require reserving more cases/families after the audit.
- Do not claim all90 tasks are semantically unseen simply because their
  suite is named90. The [full goal audit](goals/AUDIT.md) finds 22 typed overlaps and 3 additional
  ambiguous fixture-family overlaps. Conservatively reserving all 25 plus
  the inspected development families leaves 63 candidate task IDs. These are
  candidates, not 63 proven distinct goals or a selected final test set.
- Plus/Pro source-familiar goals support condition adaptation, not an
  automatic new-goal claim. Benchmark absence alone is not a mathematical
  guarantee of distribution separation.
- Existing 78/10/10 protocol does not govern this new source40 arm. In
  particular, LIBERO10 is now source data; preserve the old split as history.

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
