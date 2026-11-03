# Why the two zero-steering baselines differ

Read-only audit completed November 3, 2026. The velocity and all-layer KV
`test_before` panels evaluate the same frozen pi0.5 checkpoint with steering
explicitly set to zero. Their differences are execution reproducibility
problems, not an initial advantage of either learned interface. The earlier
chat explanation that their evaluation seeds differ was incorrect: the learner
initialization seeds differ, but the evaluation case and motor-noise schedules
are shared.

## Direct evidence

The CPU audit read both eight-worker, 500-episode panels on Baige, verifying
2,000 initial/first-chunk artifact hashes. It ran no model, simulator or
optimizer and dispatched no new physical controls. The exact source is
`8b9e1bbb3a73d55eda9bd9acdb30a85fee79d151`, with RLinf runtime
`8a1233c3fae6f5905adfda7684c455842b0ecfb8`. Both panels bind initializer hash
`661bf172a2ae9dfca75216697eeaa543408bcbdfa53c98132759b93e0e8993a5`.

| Saved comparison | Matching pairs / 500 |
| --- | ---: |
| Case descriptors and constructor seeds | 500 |
| Explicit native Gaussian seeds and actual tensors | 500 |
| Both executed steering vectors equal zero | 500 |
| Initial physical scene/state fingerprints | 481 |
| Initial public robot-state arrays | 489 |
| All three initial camera arrays | 388 |
| First requested action chunks | 222 |

**170 pairs have identical initial camera arrays, public state, task instruction
and Gaussian noise, but different first action chunks.** In 137 of these,
the saved 2,064-dimensional policy features also match exactly. This directly
locates a substantial source of divergence in motor inference, before physics
has executed any policy action. For example, TurnOnSinkFaucet seed 540100
has identical observations, features and noise but a first-chunk command
maximum absolute difference of 0.0027397871.

Across the complete episodes, first-success outcomes disagree in 65/500
pairs (36 ID, 29 OOD); final-success outcomes disagree in 72/500. The aggregate
first-success rates are velocity-panel ID 44.0% / OOD 53.6% and KV-panel
ID 43.2% / OOD 56.4%. These are two executions of the original policy.

## Numerical execution changes with asynchronous batching

Production requests specify 48 lanes per GPU but do not bind a deterministic
batch schedule. `collect_bank` defaults `synchronize_initial_lanes` to False,
accepts whichever reset futures have completed, and invokes the BF16 motor on
the currently running lanes. Rank 0 starts with batch sizes
`1, 2, 15, 24, 26, 38` in the velocity run and
`1, 4, 28, 45, 45, 45` in the KV run. All eight rank pairs have different
batch schedules, despite the same requested lane limit.

Earlier allocated-GPU capacity qualification already established that this
frozen motor's numerical actions change with batch size, including initial
batched/singleton maximum differences up to 0.11414 at the 32/48-lane pin.
See the [capacity result](../robocasa-rollout-capacity-3248-20261103-r1/RESULT-SUMMARY.json).
The present input-matched disagreements show this lack of numerical
reproducibility in the actual baseline panels. Batch scheduling is a verified
execution difference and the established numerical mechanism; this audit did
not separately isolate CUDA kernel selection, row composition and device
arithmetic with a new controlled GPU replay.

The existing `zero_native_exact` check compares native and zero-steered
actions on the same prepared observation batch within one worker. It correctly
protects interface identity within that call, but does not establish identical
actions across independently scheduled collectors. A matching lane limit is
insufficient to match the actual numerical execution.

## Initial scenes also differ in some OOD cases

All 250 ID initial physical fingerprints match. Nineteen OOD pairs do not:
17 TurnOnSinkFaucet and 2 OpenStandMixerHead. Four have different compiled XML
and body placement, with two faucet cases also having different mesh/geometry
arrays. These four are genuine scene differences, not merely image roundoff:

- TurnOnSinkFaucet: seeds 540107 and 540109.
- OpenStandMixerHead: seeds 550109 and 550136.

Six initial public-state arrays differ by more than 1e-6; the maximum difference
is 1.7168 in a mixer case's position coordinates. Some remaining fingerprint
differences are much smaller, so the 19 should not all be described as changed
geometry. Initial camera differences include both unchanged-scene rendering
variation and changed scenes; their respective downstream effects are not
separated here.

The two arms use different `canonical_root` directories. The existing reset
check compares a case against its own arm's canonical snapshot, so both fresh
panels can pass without validating the same physical scene across arms.

The deployed upstream Counter code also deduplicates XML geometry elements
with `list(set(valid_geoms))`, then uses the resulting region order for seeded
placement sampling. Object-set ordering is not fixed by seeding Python random
or NumPy. This is a concrete upstream reproducibility hazard, but this audit
has not established that it caused each of the four observed XML differences.
Exact upstream attribution of the initial reset differences remains open.

## Required comparison changes

For a future comparison, bind a deterministic inference batch plan, including
row order, refill/tail handling and padding. A startup barrier alone does not
fix later asynchronous refills. Validate the original/zero-steering policy
across that actual plan and record batch membership, rather than checking only
the requested lane count. Cross-arm case validation must use a common physical
reference; same constructor seeds alone do not prove scene equality.

Preserve the existing panels as evidence. Report each as its own before-run
measurement; do not interpret their baseline gap as a steering effect or claim
exact paired execution from seeds alone. Controlled same-input GPU replay and
initial-scene reconstruction would be needed to apportion all 65 outcome flips
between numerical and reset/rendering mechanisms. No live contract, worktree,
checkpoint, job or training source was modified by this audit.

## Evidence and reproduction

- [SUMMARY.json](SUMMARY.json): aggregate counts, rank batch schedules,
  per-task outcome disagreements, source identities and representative artifact
  references with hashes.
- [INITIAL-PHYSICS-MISMATCHES.json](INITIAL-PHYSICS-MISMATCHES.json): all 19
  mismatched fingerprints and public-state deltas.
- [IDENTICAL-INPUT-INSTRUCTIONS.json](IDENTICAL-INPUT-INSTRUCTIONS.json):
  instruction equality for all 170 image/state/noise-matched action divergences.
- [SOURCE-EVIDENCE.json](SOURCE-EVIDENCE.json): hashed source excerpts from the
  actual pinned worker and upstream placement implementation.
- [compare_saved_baselines.py](compare_saved_baselines.py) and
  [inspect_initial_mismatches.py](inspect_initial_mismatches.py): CPU-only reads
  of existing artifacts; run on allocated Linux with CUDA hidden. The former
  takes the project `outputs` directory as its argument; the latter takes a JSON
  list of selected audit rows. Neither writes to the source runs.

The full local `SAVED-BASELINE-AUDIT.json` is untracked raw evidence; its SHA256
and byte count are retained in SUMMARY.json. Original trajectories, images,
checkpoints and model files stay on Baige. Routing and authentication stay
outside the repository.
