# RoboTwin primary candidate; LIBERO transfer preparation retained

Date: 2026-10-29, Asia/Shanghai. User authorized implementing the LIBERO
transfer direction and preparing RoboTwin for the VLRLA setup. Subsequent
steering in this chat raised the lack of failures and asked to move to RoboTwin.
Recommendation: prioritize a RoboTwin recoverability screen; retain LIBERO as
a plumbing check and the separate audited transfer contract. This work does
not modify active runs or their splits, or stop jobs owned by other chats.

Reviewed main: `4d4f46f4cdcd8cfd7c7056e95bf85d0a90e0ed59`.
Changes are in a new detached worktree. Main's uncommitted coordination
changes belong to ongoing work and must be preserved.

## Preparation in this run

1. Freeze a separate 40-source/90-target manifest from pinned official
   LIBERO. Audit transitive full-task overlaps and label overlap controls.
2. Preserve the existing 78-source campaign and the active clean/augmented
   pilot jobs. They are not LIBERO-40 data and cannot be relabeled as such.
3. Prepare an isolated RoboTwin environment on Baige using existing pinned
   sources and cached weights. Run CPU imports and task/split checks only.
4. Record the twelve paper tasks, source-history PPO settings, frozen bridge
   interface and randomized cross-episode protocol. Retain the older GRPO
   smoke as an integration result.

## First new GPU qualification, once preparation is verified

Proposed envelope: one Baige GPU, at most two GPU-hours total, with at least
128 GiB host RAM for RoboTwin. First qualify a real RoboTwin episode and
rendering with the cached adjust_bottle motor. A source-task recoverability
screen follows only with a verified suitable common motor. LIBERO collection
and large AD/DPT training are not the next stage of this chat.
Select concrete task/state/arm counts from measured startup and episode time
before running; every episode, startup failure and retry counts toward the cap.
Do not spend this budget on full multi-task training or the paper's eight-GPU
collector. The new envelope is a proposal pending the user's budget approval.

The small existing LIBERO corpus was at ceiling (clean 24/24 successful
episodes); the augmented corpus 21/24 and Plus/Pro development native 8/8
each also do not establish useful improving histories. A few failures alone
do not validate a source algorithm. Require an independently measured
ordinary steering improvement before broad teacher collection or meta-training.

LIBERO-90 and final Plus/PRO data cannot influence recipe selection. Plus/PRO
variants of source40 bases establish robustness only. A combined unseen-task
and shift claim requires a declared auxiliary base-family holdout.

## Current prerequisites for the larger experiment

- The parallel VLA chat has active technical pilots. Reuse reviewed, tested
  components after they are integrated; avoid launching duplicate pilots.
- Neither our new 40-source controller training nor a matched six-source-task
  RoboTwin motor has been qualified. A budget estimate follows measured
  throughput, rather than assuming the old Can or one-goal smoke scales.
- The archived RoboTwin virtual environment is absent. Shared runtime pins,
  sources, assets and adjust_bottle weights remain; MPLib must be overlaid at
  0.2.1 without changing the shared 0.1.1 package.

Contracts: [LIBERO](../../docs/libero-transfer.md),
[RoboTwin](../../docs/robotwin-vlrla.md).
Verified preparation: [results and receipts](RESULTS.md). The static LIBERO
audit produced 40 source tasks, 68 strict targets and 22 overlap controls;
three Linux regressions passed. The isolated RoboTwin CPU environment passes
all twelve task imports; GPU rendering and a common multi-task motor remain
unqualified. No new GPU jobs were launched by this preparation.
