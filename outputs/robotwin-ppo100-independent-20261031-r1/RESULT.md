# RoboTwin full-memory PPO: iteration-100 endpoint

Independent saved-data audit, 2026-10-31. The full-memory and separately trained
reset arms are complete, and their shared outer continuation has exit code 0.
This receipt does not cover the SAC final endpoint or establish a memory benefit.

| Source iteration | Full source actions | Bottle normal / same-weight reset / reward-blind | Stapler normal / same-weight reset / reward-blind | Source validation |
| ---: | ---: | --- | --- | ---: |
| 10 | 93,851 | 6/8 · 7/8 · 6/8 | 0/8 · 0/8 · 0/8 | 1/24 |
| 25 | 238,595 | 8/8 · 8/8 · 8/8 | 0/8 · 0/8 · 0/8 | 2/24 |
| 50 | 479,674 | 7/8 · 7/8 · 7/8 | 0/8 · 0/8 · 0/8 | 4/24 |
| 100 | 964,875 | 7/8 · 6/8 · 8/8 | 1/8 · 1/8 · 1/8 | 2/24 |

All fixed stages were reported without selecting the best target checkpoint.
The iteration-100 normal target used 1,596 bottle and 2,985 stapler primitives.
Same-weight reset used 1,866/2,974; reward-blind used 1,557/2,985. Stapler's
equal success counts conceal different cases: normal and reward-blind succeed
on episode 7; same-weight reset succeeds on episode 2. Bottle normal's only
failure is episode 5; same-weight reset also fails episode 8; reward-blind
succeeds on all eight. These are eight-case development observations.

The source endpoint records 100 complete iterations, 600 complete K4 trials,
2,400 source episodes, 100 policy updates and 500 optimizer steps. Source
exposure is 964,875 actual actions, below the declared 1,100,000 source cap.
The episode count follows the saved complete-trial count and four-episode
contract; this pass did not rescan all source raw episodes.

The canonical owner snapshot charges 1,068,449 total actions: source 964,875,
source validation 40,874, targets 56,300, and native probes 6,400. Pending and
uncertain charges are zero. Summing saved episode records independently
reproduces every evaluation-phase total. The snapshot byte hash and canonical
header content hash match the terminal owner audit. No canonical event forest
was replayed.

All 64 stage evaluation groups match their saved stage files and contain their
requested 304 episodes, including 16 single-episode native probes. None of the
learned evaluation records is marked budget-truncated; native rows do not
carry that field. Target horizons are 400 actions; source validation uses the
declared 400/500/600 per-task limits. Source-validation stage comparisons are
not wholly paired: `handover_mic` has unchanged requested seeds but different
summary initial-observation hashes, while the other five goal banks match.

At iteration 100, all 16 target scene cases have matching requested **and
resolved actual** seeds, raw reset-array descriptors, first policy-observation
RGB/wrist/proprio/feature descriptors and language across normal, memory-reset,
reward-blind and the separately trained reset controller. The full model's three
conditions reference the same pure policy digest and archived policy bytes;
the trained reset controller has different weights.

For each task's first episode, normal, same-weight reset and reward-blind have
identical saved executed histories, including sampled noise and motor actions.
Later histories can diverge under the interventions. Across the 16 cases,
normal and memory-reset have two identical whole-episode history digests;
normal and reward-blind have eight. The trained reset controller has none
identical to normal. These comparisons rehashed 615 referenced target chunk
JSON records and compared their saved array/reward/action descriptors; they do
not claim an independent rehash of every raw array payload.

The summary's `initial_observation_sha256` field corresponds to a saved feature
hash in inspected endpoint records. The additional endpoint raw-descriptor
check above supports physical observation matching beyond that summary field.
Matching endpoint scenes does not make independently trained policies share
support histories, nor does it turn a one-case success difference into reliable
evidence of useful memory or reward-driven adaptation. The trained reset arm
scores 8/8 bottle and 0/8 stapler; reward-blind scores 8/8 and 1/8.

All 16 native probe rows use a separate seed/initial-observation bank, with no
matches to the learned target bank. They are not a paired native baseline, and
no earlier SAC5 panel is pooled here.

Independent byte checks passed for all 11 supplied references among 32 saved
metadata/checkpoint files, including fixed stage recoveries and both archived
target policies. The final coordinator recovery SHA256 is
`c439486c327116c8da4fc077217416757bdc2f21ffa819f8ae802cc3f7810fa5`;
the fixed100 source snapshot is
`d744a3691855bc0d79f6eee075e5e358b10528da018b40fe2d216a7b50c7ca7c`.
The terminal summary SHA256 is
`4fe0a6aff5b8454a71e1e97cdb21c0fa1b94d14e9f61ae9e20ccf9731390d83a`.
The saved frozen-motor fingerprint is unchanged:
`721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41`.

No terminal scheduler-duration or worker-clock receipt was found in this outer
output. GPU-seconds therefore remain unknown here. The 43,200-second driver
limit is a ceiling, and per-arm child times must not be summed as separate
GPU allocations: the shared pair's scheduler time is counted once by its owner.

This verification used read-only saved JSON and byte hashes. It did not
deserialize Torch models or arrays, run inference/tests/simulators, allocate
resources, control jobs, rewrite original artifacts or impose a new launch gate.
