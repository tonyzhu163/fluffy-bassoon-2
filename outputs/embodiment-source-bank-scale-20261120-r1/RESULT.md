# Three generated demonstration banks lowered the weaker-arm score

**The three-bank motor scored 9/50 on Panda and 22/50 on Sawyer. The one-bank motor scored 23/50 and 16/50.** At equal updates, the weaker-arm score fell from 16 to 9 and the combined score fell from 39/100 to 31/100. Neither motor reached the registered cutoff of 38/50 on each arm. This closes the scale comparison at source development. [Registered plan](PLAN.md) · [Paired native receipts and layout IDs](evidence/native.json)

| Source arm | One bank, successes and Wilson 95% interval | Three banks, successes and Wilson 95% interval | Newly successful | Lost successes | Successful with both | Failed with both |
|---|---:|---:|---:|---:|---:|---:|
| Panda | 23/50 (33–60%) | 9/50 (10–31%) | 4 | 18 | 5 | 23 |
| Sawyer | 16/50 (21–46%) | 22/50 (31–58%) | 16 | 10 | 6 | 18 |

Both frozen finals used the same 50 paired simulated Goal 5 BASE starts, seeds 740000–740049, with five issued controls per replan and a 300-control horizon. The host-local scorer reproduced all donor bytes and four geometry fingerprints on every start. There were zero overlaps with fitting exposure or the 50 released donors. Both sets of 100 native reward tapes passed independent hash, reset, control and terminal-outcome readback. [One-bank report](../embodiment-probe-order-20261120-r1/EVAL-BROAD.md) · [Three-bank readback](evidence/native.json)

## What changed in training

| Data supplied to the action head | Panda, one bank → three | Sawyer, one bank → three |
|---|---:|---:|
| Unique released training tapes | 163 → 163 | 161 → 161 |
| Successful recovery suffixes | 17 → 17 | 32 → 32 |
| Generated training tapes | 66 → 257 | 94 → 345 |
| All unique training tapes | 246 → 437 | 287 → 538 |
| Distinct successful training layouts | 94 → 285 | 121 → 372 |
| Controls in unique training tapes | 24,406 → 41,951 | 29,580 → 53,973 |
| Generated anchors drawn out of 25,600 | 17,946 → 23,028 | 19,812 → 23,749 |

Generated campaigns 01, 02 and 03 supplied 66/94, 93/128 and 98/123 Panda/Sawyer tapes respectively. Their 96, 128 and 128 admitted layouts included 95, 128 and 126 distinct layouts with a successful tape on at least one arm. Two of Sawyer's 163 accepted released training trajectories were duplicates. The offline validation set stayed at one Panda and four Sawyer tapes. All 12 generated shard indexes matched the terminal producer manifests. [Bank digests and provenance](evidence/inputs.json) · [Per-arm fit and sampler counts](evidence/fit.json)

Both fits inherited the same categorical step-6,400 parent and normalization, started fresh AdamW, and ran another 6,400 updates with seed 20261120. The optimizer used head/LoRA learning rates of 0.0003/0.0001, weight decay 0.01, gradient clipping at 1.0, and global batch and microbatch eight. Each update weighted four anchors per arm equally. Within each arm, the sampler drew a successful layout uniformly, a unique tape uniformly on that layout, then a control uniformly in that tape. Replaying those draws reproduced both saved final sampler states. Widening the bank raised the generated share of actual draws from 70.1% to 90.0% on Panda and from 77.4% to 92.8% on Sawyer. The result measures wider coverage together with the resulting redistribution of training draws. [Matched contracts and sampling evidence](evidence/fit.json)

## Approach on the recorded starts

Across the first 50 controls, the median closest distance from Panda's end effector to the plate's **reset** center was 8.87 cm with one bank and 10.72 cm with three. The median paired increase was 2.12 cm across all 50 starts and 2.06 cm across the 18 lost successes. Sawyer also had a larger median paired distance (+1.18 cm) while its native success count increased. This post-outcome check used recorded world-frame displacements and matched opening-state and scene hashes on every pair. Actual touch timing and subsequent plate motion remain unresolved for these two categorical motors. [Per-start approach measurements](evidence/approach.json)

## Compute and final checkpoint

Baige Distributed Training jobs `job-29i0nmouha71` (fit) and `job-q4ftsy8od11t` (score) both succeeded on one qualified RTX PRO 6000D each. Combined allocated pod time was **0.459 GPU-hours** of the 3-hour budget, comprising 20m44s for fitting and 6m47s for scoring. Submission through terminal status totaled 0.466 GPU-hours. The registered three-bank final SHA-256 is `e3c07a2faa05a7fc69e60c2813c32c417173e70be3b7f2ba05bf2f821e60dea7`. The producer used `29f1db55` and the host-local scorer used `e2f794c4`. Both terminal jobs, all four stage exits and all four exclusive GPU probes were verified. [Scheduler and stage receipts](evidence/scheduler.json) · [Execution record](EXECUTION.md)
