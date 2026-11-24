# Restricting Kinova3 corrections to the observed line gave one rescue and one loss

The frozen policy succeeded on **3/6** layouts with its original commands, **3/6** with corrections restricted to the best-observed translation direction, **4/6** with the same rank-one forward fit and an unrestricted inverse, and **3/6** with smooth ridge matched for effective degrees of freedom. All six Kinova3 layouts were paired within the same job. Each method saw the same opening policy inputs and frozen action proposal, then replanned from its own trajectory. [Paired native outcomes](evidence/gpu01/paired.json) · [Protocol and model information](PLAN.md)

| Task and initialization | Untreated | Rank-one, line-constrained | Same fit, unrestricted | Matched smooth ridge |
|---|---|---|---|---|
| Drawer and bowl, 14 | Failure (300) | Failure (300) | Success (173) | Success (180) |
| Drawer and bowl, 15 | Failure (300) | Failure (300) | Failure (300) | Failure (300) |
| Two moka pots, 14 | Success (404) | Failure (520) | Success (470) | Failure (520) |
| Two moka pots, 15 | Failure (520) | Success (475) | Failure (520) | Failure (520) |
| Cream cheese and bowl, 14 | Success (95) | Success (93) | Success (94) | Success (93) |
| Cream cheese and bowl, 15 | Success (99) | Success (99) | Success (98) | Success (99) |

Parentheses give controls to native reward or timeout. The line-constrained inverse rescued one of three untreated failures and spoiled one of three untreated successes. Unrestricted inversion rescued one drawer layout and preserved the three untreated successes. Against the unrestricted inverse, the constrained inverse gained one outcome and lost two. Smooth ridge and line restriction each solved three layouts, including different third successes. The two moka-pot outcomes trade places under the restricted inverse. This pilot supplies no overall native-success advantage for restricting the correction and only one discordant success in favor of the unrestricted inverse over identity.

## What the command tapes establish

Each target task used a separate 15-control passive support from layout 11. Queries used layouts 14 and 15. All 24 native episodes used the Kinova3 arm, its native world-frame OSC and the frozen LIBERO π0.5 checkpoint. Independent replay of the action adapter from every stored bounded motor command reproduced every issued float32 command exactly. Wrist and gripper components were preserved throughout. The constrained correction’s largest component outside its registered line was 3.03 × 10⁻⁸ in normalized command units. Only three of its 1,787 controls newly hit a translation box face, all on the failed drawer layout 15. The two moka-pot flips occurred without a newly saturated translation coordinate. They establish sensitivity to the issued corrections. Native rewards alone leave the exact trajectory mechanism unresolved. [Method-specific command/reward audit](evidence/gpu01/readback.json)

The three tasks were selected using historical failures and a preserved-success control, with just two layouts per task. One within-job extra success for unrestricted inversion is smaller than the observed range across the untreated repeats. The original motion prediction and five-control replay results give an independent view of what changed in the early controls. [Pre-launch offline and replay findings](RESULT.md)

## Reproducibility across workers

Untreated Kinova3 succeeded on **2/6, 3/6 and 4/6** in three independent jobs. All three reproduced both cream-cheese successes and both drawer failures. The first succeeded on neither moka-pot layout, the current job succeeded on layout 14, and the native-reference job succeeded on both. This variability is large enough to change a one-success ranking.

In all six cross-job comparisons with the native-reference RTX 4090, every reset field except the renderer name matched. This includes initial physics, both raw camera frames, motor inputs, task prompt and the complete flow-noise bank. Nevertheless the opening frozen proposal hashes differed on all six. Numerical comparison of the five bounded opening actions per layout found RMS differences of 0.00078–0.00231 and maximum absolute differences of 0.00247–0.00658 in normalized action units. The present job’s four conditions shared an exactly equal opening proposal on each layout. Between-job variability remains unassigned to a particular GPU, library or kernel. [Cross-job readback by layout](evidence/gpu01/readback.json) · [Previous untreated repeats](../frozen-vla-visual-adapter-20261123-r1/RESULT.md)

Baige Distributed Training job job-495gtsfh6x6y used one NVIDIA RPBZZZ6 (RTX PRO 6000D pool). The adapter revision was d8d7b63c58d1cf37132e6bfa3582e24258005027 and the native bindings revision was a8a997e7d06bb131d3f9710204c58e106f0f0904. The worker exited 0 after 479 seconds, or 0.133 GPU-hours. All 24 full tapes are retained on the shared volume under /mowenjie/fluffy-bassoon-2/outputs/frozen-vla-supported-correction-20261123-r1/gpu01. Their SHA-256 identities are recorded in the Linux CPU readback. [Readback program](readback_native.py) · [GPU receipt](evidence/gpu01/gpu.json) · [Launch](evidence/gpu01/launch.txt) · [Exit](evidence/gpu01/exit.txt)
