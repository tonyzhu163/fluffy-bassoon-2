# Fixed-goal motor crossovers in the archived LIBERO controls

**One BASE/S20 fitting reset has a complete action-tape witness.** For “push the plate to the front of the stove,” LIBERO-Goal task 5, repeat 1, retry pose 4, seed 19220504, only default execution succeeded on BASE and only the short wait succeeded on S20. The eight retry records share the initial physics fingerprint, public observation hashes, instruction, horizon and opening proposal. The original four-mode archive also verified identical first five controls across modes. The default support at pose 3 succeeded on BASE and failed on S20, so support reward alone distinguishes these two tapes.

| Fixed executable cadence | BASE first native hit | S20 first native hit |
|---|---:|---:|
| Default, replan after five controls | 137 | miss |
| Ten controls, seven waits (`short_wait`) | miss | 222 |
| Ten controls, fourteen waits (`paced`) | miss | miss |
| Gripper-dependent waits (`gripper_wait`) | miss | miss |

Misses exhausted 300 controls. [Exact ten Baige tape paths, file hashes, seeds and first hits](verified-tapes.json) include both supports and all eight retries. The paths follow `/mowenjie/fluffy-bassoon-2/outputs/libero-decision-pretraining-20261119-r1/menu02/episodes/fitting/libero_goal-5-r1-{BASE,S20}/{support,default,short_wait,paced,gripper_wait}/public.npz`. The read-only [tape check](verify_tapes.py) recomputes native hits from the saved rewards.

## Coverage

One group is one fixed goal and retry reset evaluated under both persistent conditions. A strict crossover requires a nonempty set of successful executable choices in each condition and an empty intersection across the measured menu. The primary endpoint is any native hit. “Reachable in both” counts groups where each condition has at least one measured success. Denominators include all paired resets in the stated panel. Panels that reuse resets, or add comparators to a menu, are reported separately.

| Archive and fixed menu | Strict / paired groups | Reachable in both | Strict IDs |
|---|---:|---:|---|
| BASE/S20 original two-cadence development | 0/16 | 5 | — |
| BASE/S20 four-cadence fitting | 1/16 | 4 | `libero_goal-5-r1` |
| BASE/S20 four-cadence development | 0/16 | 6 | — |
| First-controls fresh, default/paced, query pose 7 | 0/8 | 3 | — |
| Moving rack BASE/D02, three continuations, source poses 0–2 | 1/3 | 2 | 1 |
| Same menu, development poses 3–11 | 5/9 | 6 | 3, 4, 6, 7, 8 |
| Same menu, prospective poses 12–23 | 2/12 | 3 | 21, 22 |
| Same prospective poses, plus live VLA and camera controller | 1/12 | 8 | 22 |
| Failure diagnosis, bowl, two cadences per fixed destination | 0/8 | 6 | — |
| Failure diagnosis, bottle, two cadences per fixed destination | 0/12 | 6 | — |

The four-mode development group `libero_10-3-r0` reverses if the menu is restricted to default versus paced. The two wait modes each succeed under both timings. First-controls' fresh adaptive policies use the current or previous motion to pick a schedule and are distinct from its two fixed interventions. Failure diagnosis contributes four bowl and six bottle retry poses crossed with two fixed destinations. Each comparison holds both the prompted destination and the native predicate fixed. Duplicate support assignments reuse these same queries.

**Release-phase has zero crossed groups among eight retry pose/goal cells.** It has 30 practice tapes on five poses under normal, early and late opening. The eight retries at poses 24–27 are two destination instructions per pose, each with one nominal execution. The release perturbation was confined to practice. The 320 scored assignments cannot supply the missing condition-by-correction retry branches. [Saved phase records](../../libero-release-phase-20261119-r1/phase.json)

## Moving-rack evidence

All 24 pose groups have identical saved physics and pre-branch wrist-state hashes across the six branches. The instruction always specifies the bottle rack (LIBERO-Goal 9). BASE uses a stationary rack and D02 moves it near placement. Unshifted wrist tracking wins on BASE and shifted tracking wins on D02 at prospective poses 21 and 22. At 21, the live motor and current-camera policy each hit in both regimes. At 22, the full successful sets are BASE {track, live} and D02 {shift, camera}. Scoring only placement at control 300 gives strict crossovers at **both 21 and 22**, even with the expanded menu. In the archived cyclic pairing, the track support for query 21 (pose 20) missed in both regimes. The support for query 22 (pose 21) succeeded on BASE and missed on D02. [Support rewards](../../libero-moving-rack-20261119-r1/history-test.json)

The rack collector saved one branch record per (pose, regime, option), including lift control, first native hit and terminal predicate. The Baige archive has full tapes for the nominal openings and the live comparator. The three simulated continuations have JSONL outcomes and executable controller code, with no per-control continuation tapes saved by that collector. [Source branches](../../libero-moving-rack-20261119-r1/source-branches.jsonl) · [Development branches](../../libero-moving-rack-20261119-r1/development-branches.jsonl) · [Prospective branches](../../libero-moving-rack-20261119-r1/prospective-branches.jsonl) · [Continuation implementation](../../libero-moving-rack-20261119-r1/pilot.py)

Baige inputs are relative to `/mowenjie/fluffy-bassoon-2/outputs/`:

- Source/development witness openings at poses 1, 3, 4 are `libero-first-attempt-20261119-r1/transfer02/episodes/bottle/s{1,3,4}-BASE/support-g1-learned/public.npz`. The same pattern at poses 0–2 supplies the three tracking references.
- Openings at poses 6, 7, 8 are `libero-first-attempt-20261119-r1/transfer02/episodes/bottle/s{0,1,2}-BASE/query-g1/public.npz`, respectively.
- Prospective witness openings are `libero-moving-rack-20261119-r1/open02/episodes/pose{21,22}/public.npz`. Live comparisons are `libero-moving-rack-20261119-r1/open02/live/p{21,22}-{BASE,D02}/public.npz`, with native labels in the sibling `evaluator-predicates.npy`.

Reproduce the counts with `python3 crossover.py` from this directory. [Machine-readable denominators and condition-specific winning sets](counts.json). The analysis used existing files and zero GPU time.
