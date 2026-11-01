# RoboTwin SAC: completed iteration-100 endpoint

Independent saved-data audit, 2026-10-31. The outer continuation and final slice
0061 both have exit code 0 and complete status. The final coordinator's SHA256
matches `f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8`;
its ZIP CRC check passes all 210 members without decoding pickle or tensors.

The saved algorithm contract identifies this as **feature-conditioned
DSRL-SAC-style**, using 2,062 frozen-VLA/public-state features and 1,024 noise
coordinates. It is not the authors' PixelSAC/DSRL-NA implementation. This exact
baseline uses bounded-uniform source warmup, batch 256, warmup 500 chunks and
UTD 20; its full contract is retained in the receipt.

| Source iteration | Source actions | Source SAC updates | Frozen bottle / stapler | Online bottle / stapler | Source validation |
| ---: | ---: | ---: | --- | --- | ---: |
| 10 | 96,789 | 52,580 | 3/8 · 0/8 | 1/8 · 1/8 | 1/24 |
| 25 | 248,082 | 150,320 | 4/8 · 0/8 | 3/8 · 0/8 | 1/24 |
| 50 | 502,356 | 314,500 | 3/8 · 0/8 | 5/8 · 1/8 | 3/24 |
| 100 | 1,003,790 | 638,500 | 6/8 · 2/8 | 5/8 · 0/8 | 2/24 |

All fixed stages are complete and reported without selecting a checkpoint from
target outcomes. All 40 requested evaluation groups contain their full 224
episodes, with no budget-truncation flags. The source endpoint records 600
complete K4 trials at 100 iterations: 2,400 episodes under the saved four-episode
contract. Source SAC updates total 638,500; its separately recorded optimizer
step count is 1,915,500. These units are not interchangeable with PPO iterations.

The current explicit source safety cap is 1,100,000 actions. Actual source cost
is 1,003,790, including the complete final trials. The saved canonical campaign
charges 1,088,029 actions against a 1,194,400 total cap: source 1,003,790,
source validation 41,032, frozen target 21,360 and online target 21,847. Pending
and uncertain charges are zero. Summing saved evaluation episodes independently
reproduces all three evaluation-phase costs. Canonical totals are bound saved
owner snapshot evidence, not a fresh replay of the event forest.

At iteration 100, frozen bottle/stapler use 1,693/2,707 primitives with no target
SGD. Online bottle/stapler use another 1,920/3,200 primitives and 1,240/2,080
target SAC updates, respectively (3,720/6,240 recorded optimizer steps). Each
online task independently restores the complete source learner, excludes the
source replay rows, and uses fresh target replay with replacement from its
first row, without a second warmup. Source costs remain separate and counted
once.

Episode indices **1/2/4/8 are sequential episode indices**, not automatically
support/query budgets. Online updates occur after every executed chunk,
including inside episode 1. Its eighth episode therefore is not a separate
frozen query after seven support episodes. Frozen evaluation keeps the source
actor fixed across all eight episodes. The receipt includes episode outcomes
and cumulative primitive/chunk costs at the reported indices.

All four stage source snapshots retain their declared byte hashes and report
restored source RNG state. Pinned evaluator code independently clones source
learners for evaluation and restores source RNG in `finally`; source clocks and
pending state are checked afterward. At the final endpoint, all frozen-target
and source-validation acting records use one policy digest,
`cca433fdaf4a77e42ccafecb68392add4a54633ddcbf593a3fb926ff76a8ab10`,
with optimizer clock fixed at 638,500 and `online_updates=false`. Its archived
policy bytes also match. This is saved acting-policy/clock evidence, not a new
deserialized parameter comparison.

Both SAC conditions match all 16 target scenes in the audited full-PPO and
trained-reset endpoints: requested/resolved seeds, raw reset-array descriptors,
first policy RGB/wrist/proprio/features, and language all agree. Frozen and
online SAC start each task's first episode with matching actor digest,
interaction RNG and first sampled noise; subsequent online updates change the
policy. Actual histories need not match the PPO policies. The 16 native probe
rows from the PPO endpoint have a separate seed/initial-feature bank, with no
SAC target matches; they are unpaired context. No earlier native panel is pooled.

Target summary seed/initial hashes match across stages. Source validation is
not wholly paired across stages: `handover_mic` uses the same requested seeds
with different initial-feature hashes; the other five goal banks match. No
cause or task exclusion is inferred from that discrepancy.

All 10 supplied reference hashes match among 26 rehashed metadata/checkpoint
files, including the four fixed source snapshots, final coordinator, final
target recoveries and frozen acting archive. Another 640 final-endpoint chunk
JSON references match their manifests. Raw array descriptors were compared;
this pass did not rehash every raw frame/array payload. The saved frozen motor
fingerprint remains unchanged at
`721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41`.

The continuation supervisor records 32,209.95 elapsed seconds, and the last
slice 923.55 seconds. These are saved process intervals, not a complete
scheduler-cost reconciliation or the total lifetime cost of prior source
prefixes. Resource accounting remains with the owner.

This verifies completion and limited-panel outcomes. It does not demonstrate
reliable adaptation, a sample-efficiency advantage, or broad robotic gains.
No tensors/models were loaded, no tests/simulators/rollouts or allocations ran,
and no jobs, ledgers or original artifacts were changed by this verifier.
