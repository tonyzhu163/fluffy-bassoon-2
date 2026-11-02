# Flora prefix scoring for a first-success reward

**Status: DRAFT, not approved, not launched.** Claude Code session
(tonyzhu163) proposes; this session has no Baige route, so it **needs a Baige
owner** and **human approval of the GPU budget** below. No new simulator
controls, no policy inference, no SGD.

## Why

Gate 1's Flora arm must match the RLinf simulator run
([plan](../robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/PLAN.md)): +1 at
first success, episode terminates, horizon 400, eight H50 chunks. Flora would
then be asked "done yet?" after every chunk, not once per finished episode.
The [offline rule analysis](../flora-first-success-offline-20261102-r1/RESULT.md)
found a calibrated rule (P5 ≥ 0.545) with 13% false successes on whole
episodes. It could not test prefixes: Flora has only ever scored whole
750-control videos. Each extra check is another chance to fire falsely, and
an early fire on a partly opened drawer rewards the wrong behaviour.

## Question

At the RLinf chunk boundaries, which first-success rule over Flora's prefix
labels (threshold on P5, and K consecutive positive checks) matches the
simulator's first success closely enough to use as the gate-1 Flora reward?
If none does, what is the fallback?

## Inputs (existing files only)

- **Videos.** The 384 TRAIN/OpenDrawer Flora-PPO episodes and the 50
  CALIBRATION/OpenDrawer native episodes from the
  [rubric replay manifest](../robocasa-flora-rubric-replay-20261102-r1/MANIFEST.json),
  three views, one frame per control plus the initial frame, 751 frames each.
  No PPO termination, so the first 401 frames of each video are a valid
  400-control episode.
- **Per-control truth.** Each monitor record written by
  `robocasa_residual.py` holds `success_steps` and `first_success_step`
  (one-based controls after reset). **First verify** that all 434 records
  contain these fields. This is the same check as factorial Stage A1
  ([plan](../robocasa-flora-rubric-factorial-20261102-r1/PLAN.md)); run them
  together. Truth is never loaded by the scoring workers.
- **Scorer.** The original official prompt, model, engine settings and
  three-view collage of the replay (`TerminalWorker`, temperature 0, batch 8).
  The rejected candidate rubric is not used.

## Method

1. **Prefixes.** For each video and each boundary c ∈ {50, 100, …, 400},
   score frames 0..c only. Implement this as a frame limit on the collage
   (`num_frames = c + 1` passed to `terminal_frame_indices` and
   `video_metadata`), with no re-encoding. 434 × 8 = **3,472 labels**.
2. **Qualification before the run** (allocated Linux, one GPU):
   - With the limit set to the full 751 frames, re-score 16 videos and
     reproduce their saved labels exactly (grade and probabilities to 4 dp).
   - Print the sampled frame indices for c = 50 and c = 400. If fewer than 64
     distinct frames exist, record how the official sampler repeats frames;
     do not change it.
   - Check that a prefix label reads only frames ≤ c (index assertion).
3. **Rules.** Fire at the first boundary k where P5 ≥ τ on K consecutive
   checks, for τ on a fixed grid {0.45, 0.50, …, 0.90} and K ∈ {1, 2}.
   With K = 2 the reward lands one chunk late; record that delay.
4. **Truth per episode at horizon 400.** Success means
   `first_success_step ≤ 400`; the simulator chunk is ⌈first_success_step / 50⌉.
5. **Selection without leakage.** Split the 48 TRAIN cases 24/24 by case
   index (fixed before scoring: even/odd). Choose (τ, K) on the even half;
   report the odd half and CALIBRATION unchanged.

## Readout

Per rule, on held-out cases:

- **Episode false-success rate:** Flora fires by 400 while the simulator
  never succeeds by 400.
- **Recall:** Flora fires by 400 when the simulator does.
- **Early fires:** in simulator successes, Flora fires before the simulator's
  first-success chunk. These pay reward for unfinished work.
- **Timing error:** Flora chunk minus simulator chunk, as a histogram.
- **Per-check false-fire rate by boundary,** to show whether short prefixes
  or long ones cause the false successes.
- **Comparison with whole-episode scoring:** the full-video labels at
  horizon 750 with the same τ.

## Decision rules (fixed before scoring)

- **Adopt** the chosen rule as the gate-1 Flora first-success reward if, on
  held-out cases: false-success rate ≤ 15%, recall ≥ 0.80, early fires ≤ 10%
  of simulator successes.
- **Fallback A:** if early fires are the only failure, use K = 2, or skip
  the earliest boundaries. Fix the earliest scored boundary from the
  simulator's `first_success_step` distribution in the even-half cases (for
  example, the boundary before which under 5% of first successes occur),
  read during the Stage A1 check, before any prefix label is opened. The
  [timing panel](../robocasa-pi05-success-timing-20261102-r3/RESULT.md)
  reports only the maximum (279 of 26 OpenDrawer episodes), not early times.
  Report this as a changed reward, not as matching RLinf.
- **Fallback B:** if false successes stay above 15% at every rule, do not
  use Flora for termination. Score the fixed 400-control episode once and
  put the binary reward at the final chunk with γ = 0.99. Say plainly that
  this differs from the simulator arm's first-success reward, so a Flora/sim
  gap then mixes reward quality with reward timing.
- In every case, the residual ~10% whole-episode false-success floor
  remains until factorial Stage A explains the confident false positives.

## Budget

The replay cost 6,218 charged GPU-seconds for 1,268 full-video labels
(≈ 4.9 GPU-s per label including model load). Prefix labels decode fewer
frames but still sample up to 64, so assume the same rate: 3,472 labels ≈
**17,000 GPU-s**. Proposed ceiling **24,000 GPU-s** (6.7 GPU-hours), at most
four GPUs, one allocation of at most two hours, qualification and failed
attempts charged, no automatic retry. If the ceiling stops the run, keep the
completed batches and report which boundaries are complete; do not resume
under a new budget without approval.

## Preservation

Read-only access to the replay's videos and monitor records; write only to
this run's output directory. Keep every prefix label, the frame indices used
for each label, receipts and the qualification labels. Pin the reviewed
`main` commit on a fresh worktree; do not touch the RLinf, interface-screen
or AD/DPT worktrees.
