# Flora rubric factorial replay and drawer ground truth

**Status: DRAFT, not approved, not launched.** This plan replaces the untested
revision 2 in the [first replay](../robocasa-flora-rubric-replay-20261102-r1/RUBRIC.md#next-iteration-draft).
Revision 2 drops the OpenDrawer "fully open" sentence, so it cannot test the
audit's hypothesis that the 58 false Score-5 failures come from the prompt
never stating the native ≥0.95 opening bar. Revision 1 changed shared and
task-specific text together, so the first replay cannot say which part helped
or hurt. Revision 1, its frozen text and its results stay unchanged.

## Questions

1. **Ground truth.** Are the false Score-5 failures near misses (drawer
   visibly almost fully open), wrong-target openings, or something else? If
   the difference is not visible, no wording will fix it.
2. **Which text does what.** Separate the effect of the shared final-state text
   from the task completion text, on the same videos and engines.
3. **Reward mapping.** Does a different map from the five grade probabilities
   to reward separate successes from failures better for PPO, independent of
   wording?

## Stage A: drawer ground truth (CPU, no scorer)

**A1 (existing records only).** For all 183 TRAIN failures, read the saved
monitor fields `simulator_success_once`, `success_steps`, `first_success_step`
and `native_ep_meta` (current `robocasa_residual.py` writes them; first verify
the two OpenDrawer Flora-PPO runs' records contain them). Compare the 58 original false Score-5 failures with the
other 125 failures: how many reached the native bar at some step and then lost
it, and how many target the left vs right drawer. This needs no approval.

**A2 (deterministic action replay, needs approval).** If each TRAIN episode's
executed physical actions are retained (`pending-NNN.pt` rollouts or command
journals), replay them open-loop from the recorded `native_seed` on allocated
Linux CPUs, with no policy or scorer. Admit an episode only if the
initial-observation hash, the per-frame video hashes and the final native
success all match the original. For each admitted episode, record the final
normalized opening of the target drawer and of every other drawer. This
produces simulator controls but zero policy inference and zero SGD; about
384 × 600 controls at most.

Classify each failure from A2 before reading Stage B's results:

| Class | Rule |
| --- | --- |
| Near miss | target opening in [0.80, 0.95) |
| Partial | target opening in [0.30, 0.80) |
| Wrong target | target < 0.30 and some other drawer ≥ 0.80 |
| Undone | `simulator_success_once` true and final target < 0.95 |
| Other | anything else |

If A2 is infeasible (actions not retained or replay not bit-identical),
record why, use A1 alone, and have a person label the final frames of a
fixed random 20 of the 58 without seeing scores.

## Stage B: factorial scorer replay

Same 634 videos as the first replay, verified against its
[manifest](../robocasa-flora-rubric-replay-20261102-r1/MANIFEST.json) by
SHA256. Four arms, every video, all in the same persistent engine and
inference batch per video:

| Arm | Task text passed to the official terminal helper |
| --- | --- |
| O | native instruction (original control) |
| S | native instruction + `Completion criteria:` + `shared_criteria` |
| T | native instruction + `Completion criteria:` + current task's `reward_training_task_criteria` |
| R1 | native instruction + both, exactly as revision 1 |

S, T and R1 take their text byte for byte from the frozen
[revision 1 JSON](../robocasa-flora-rubric-replay-20261102-r1/candidate.json)
(SHA256 `815137b5…a220586`). No new wording is written. On OpenDrawer videos,
T is exactly the "fully open" plus target-grounding sentence, which is the
direct test of the audit hypothesis. Rerunning O and R1 alongside S and T
keeps batch composition comparable; the first replay changed 17 of 634
original grades through batch composition alone.

Everything else is held fixed as in the first replay: official code
`129a2ada`, model receipt, camera bytes and order, 64-frame sampler with first
and final frames, temperature 0, seed 0, output parsing, five-grade
probability extraction. Scorer jobs never read truth, old grades, Stage A
results or simulator state. 2,536 labels.

## Readout (CPU, after both stages)

For every arm, compute three reward maps from the saved probabilities. The
production map stays the reference; the others are a declared ablation:

- `E`: `(expected grade − 1) / 4` (production)
- `P5`: probability of grade 5
- `G5`: 1 if the greedy grade is 5, else 0

Report, per cohort (TRAIN/OpenDrawer and the five calibration banks):

- **Primary:** TRAIN within-case pairwise accuracy (48 cases × 8 draws, every
  case has both outcomes; 655 success–failure pairs). This is the ranking PPO
  advantages use. Baselines from the first replay: O/E 0.937, R1/E 0.918,
  O/P5 0.928, R1/P5 0.904.
- AUROC; mean reward on successes and on failures and their gap.
- Score-5 false positives, true positives lost, precision and recall; paired
  correction and regression counts.
- Each Stage A class: count and mean reward per arm and map, and how many
  leave Score 5.

Statistics: paired bootstrap over the 48 TRAIN cases (10,000 resamples, seed
0) for differences against O/E; exact McNemar tests for Score-5 changes;
Holm correction across the three arm comparisons. Calibration banks have 50
videos each and are reported as checks, not tests.

## Decision rules (fixed before the replay)

- **An arm becomes the wording candidate** only if, under map E, its TRAIN
  within-case accuracy is not lower than O's (bootstrap 95% interval lower
  bound ≥ −0.01), no calibration AUROC falls by more than 0.02, TRAIN true
  Score-5 losses are ≤ 10 of 201, and TRAIN false Score-5 grades fall with a
  Holm-adjusted p < 0.05. If several pass, choose the highest within-case
  accuracy; if none pass, keep O.
- **Stop wording work** if Stage A puts most of the 58 in Near miss and T
  does not move them, or if no arm passes. Keep the original prompt and record
  the false-success rate as a known reward noise floor in
  [FINDINGS](../../docs/FINDINGS.md).
- **Wrong-target or Undone classes dominate** → the next wording, if any,
  targets that class; it gets a new run ID and a fresh test.
- **Reward map.** If P5 or G5 raises the TRAIN success–failure gap with
  within-case accuracy no more than 0.01 below E, propose it as a separate
  PPO-reward experiment. This replay does not change the production map.
- **Independent check.** Any chosen wording or map is frozen and then scored,
  together with O/E, on fresh videos that played no part in development: the
  250 fresh 600-control calibration episodes from the joint preparation, once
  they exist. Nothing qualifies as a production reward from these 634 videos.

## Resources and approvals

- Stage A1: CPU read of existing records; no approval.
- Stage A2: allocated Linux CPUs only, ≤ 384 episodes, zero policy inference,
  zero SGD. **Needs human approval** because it produces simulator controls.
- Stage B: Baige, 2 nodes × 2 GPUs, four TP1 engines, batch 8, memory
  utilization 0.78 (as qualified in the first replay). The first replay spent
  6,218 GPU-seconds on 1,268 labels including cold starts; 2,536 labels need
  about 12,400. **Requested ceiling 18,000 GPU-seconds** (4 GPUs × 75 min),
  16 GiB output, no automatic retry, partial rows and charges kept on failure.
  **Needs human approval of the budget.**
- This proposing session has no Baige route. The run owner implements
  `REPLAY.py` as a four-arm copy of the first replay's scorer and summarizer,
  with this run's identity, runs CPU numerical checks on Linux, records the
  source commit, and adds their own row to
  [coordination](../coordination.md) when submitted.

## Out of scope

New prompt wording, new robot episodes for policy evaluation, PPO training,
any change to the production prompt, map or label caches, and the five OOD
task descriptions in the earlier draft.
