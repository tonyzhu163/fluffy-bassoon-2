# Camera-timed drawer cue on Kinova3 and IIWA

**Keep the tuned motor’s original instruction on this panel.** It finished 6 of 8 matched starts. A single camera-timed placement cue finished 4 of 8, losing two tuned successes and rescuing neither tuned failure. The clock-timed cue also finished 4 of 8, with opposite outcomes on two starts. These are terminal LIBERO-Pro2.0 BASE rewards on one drawer-and-bowl task, one policy-noise bank per layout, and four independent released layouts. [Registration](PLAN.md) · [Independent readback](evidence/native01/independent.json)

## Paired native outcomes

Each entry is the sum of the native per-control rewards, equal to the binary terminal success. The event time counts controls already consumed at the policy query. The last column gives the first **issued control** that differed between visual and clock, numbered from 1. At each listed difference the two branches still had identical public observations, transformed motor inputs and query noise.

| Body | Row | Raw | Tuned | Clock | Visual | Visual cue time | Clock cue time | First changed control vs clock |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Kinova3 | 34 | 1 | 1 | 1 | 1 | 55 | 60 | 56 |
| Kinova3 | 35 | 1 | 1 | 1 | 0 | 55 | 60 | 56 |
| Kinova3 | 36 | 0 | 0 | 0 | 0 | 60 | 60 | Identical throughout |
| Kinova3 | 37 | 0 | 0 | 0 | 0 | 60 | 60 | Identical throughout |
| IIWA | 34 | 1 | 1 | 1 | 1 | 55 | 60 | 56 |
| IIWA | 35 | 1 | 1 | 1 | 1 | 60 | 60 | Identical throughout |
| IIWA | 36 | 1 | 1 | 0 | 1 | 65 | 60 | 61 |
| IIWA | 37 | 1 | 1 | 0 | 0 | 60 | 60 | Identical throughout |

| Visual compared with | Baseline successes | Rescues / losses / ties | Difference in success | Descriptive 95% layout interval |
|---|---:|---:|---:|---:|
| Tuned | 6/8 | 0 / 2 / 6 | −25 pp | −50 to 0 pp |
| Clock | 4/8 | 1 / 1 / 6 | 0 pp | −37.5 to +37.5 pp |
| Raw | 6/8 | 0 / 2 / 6 | −25 pp | −50 to 0 pp |

Kinova3 had 2/4 raw, 2/4 tuned, 2/4 clock and 1/4 visual successes. Visual lost one and tied three against each comparator (−25 pp, interval −75 to 0). IIWA had 4/4 raw, 4/4 tuned, 2/4 clock and 3/4 visual. Visual lost one and tied three against tuned and raw (−25 pp, −75 to 0), and rescued one and tied three against clock (+25 pp, 0 to +75). IIWA offered zero tuned failures to rescue.

We enumerated all 256 ordered resamples of four layouts. Both arms and every condition moved together in each draw. Per-layout visual-minus-tuned differences for rows 34–37 were 0, −50, 0 and −50 pp. Against clock they were 0, −50, +50 and 0 pp. Against tuned, 240 draws had a negative difference and 16 tied. Against clock, 93 were negative, 70 tied and 93 positive. The intervals use empirical ranks 7 and 250. Four layouts leave considerable uncertainty about performance on another layout. [Joint count frequencies and arm-specific intervals](evidence/native01/independent.json)

## What the change in timing did

All eight visual branches crossed 0.5 and issued “put the bowl in the top drawer” exactly once. All eight clock branches issued the same instruction at control 60. The next policy query always restored “open the top layer of the drawer and put the bowl inside”. Raw and tuned used the original instruction throughout. Visual fired one replan earlier on three starts, at the same replan on four, and one replan later on one. Visual-only, clock-only and neither each occurred zero times. Applying both thresholds along the untreated tuned trajectory gave the same time pair on every start. [Per-episode events, controls and returns](evidence/native01/independent.json)

Every visual cue changed the proposal and the immediately following issued control against tuned on matched histories. The clock also changed the immediately following control in all eight pairs. Equal-time visual and clock runs agreed at every query and issued control, including terminal rewards. On Kinova3 row 35, the early cue changed control 56 and the subsequent rollout timed out. Both the clock and tuned runs succeeded. On IIWA row 36, clock changed control 61 and timed out. Visual stayed with tuned until its later cue changed control 66 on the common visual–tuned history. Both of those runs succeeded. On IIWA row 37, both cue conditions lost the same tuned success. These paired outcomes support effects of the assigned cue schedules on these individual runs. Later actions on already diverged histories combine the text change with its effects on subsequent observations.

Both tuned failures were Kinova3 rows 36 and 37. The detector did fire, the motor changed its issued command, and neither failure became a success. The live tapes retain score histories and public-image digests. They lack framewise ground-truth drawer travel, grasp and placement state. Thus physical mistiming, reclosure, and ineffective execution after a suitable cue remain possible explanations for an unsuccessful intervention. Different arrival times relative to the clock establish a schedule difference. They cannot establish which cue coincided with a physically open drawer.

PR 10 previously registered Kinova3 rows 36–39 at the same row-based seeds for a selector choosing among continuing, translation correction and language requery. Its rows 36 and 37 overlap here. The distinct hypothesis tested here is whether a Panda camera detector improves the timing of a once-only placement cue over its source-fitted clock on the **same tuned command interface**, including a second body, IIWA. The overlapping rows count once in this panel. No PR 10 outcomes, CPU fixture rewards or archived replay scores entered the native estimates. [PR 10 registration](https://github.com/tonyzhu163/fluffy-bassoon-2/blob/3d7f18f1a5573af296239135f2e38536b5c8046e/outputs/frozen-vla-selector-20261124-r1/README.md) · [Earlier fixed-command study](../frozen-vla-phase-transfer-20261123-r1/RESULT.md)

## Readback and provenance

Baige Distributed Training job `job-wnafkdyqfdu5` used one NVIDIA GeForce RTX 4090 and the registered checkout `e15eefb3f76e475acdf0795e2e17ba0e1ee46fae`. The `gpu01` launcher ran from 11:21:45 to 11:35:41 Shanghai on November 24 and exited zero after 836 seconds (0.2322 launcher GPU-hours). The 32 episode loop timers sum to 300.189 seconds. The independent CPU readback used cores 38 and 39 with CUDA hidden. It recomputed all 32 returns from 7,800 per-control rewards across 1,569 queries. Every control count, endpoint, timing count, registered first difference and paired contrast agreed with the official summary. [Source and CPU qualification](evidence/native01/provenance.json) · [Checker](cue_readback.py)

For every quartet we checked the released row against the original 50 × 79 bank, seed 42710300 + row, task, body, settled reset, public pose and images, motor inputs, full generated noise bank and every query’s noise, frozen-model identifiers, support-derived correction and opening proposal. Each native reset was accepted on its first attempt. The three tuned-interface branches also had identical opening commands. Raw first differed at issued control 1 because it used identity retargeting. Across bodies the release scene, row and noise matched. CPU03 and GPU01 also agreed on release row, scene and noise. Their full physics digests differed on every matched body/row across hosts. All endpoint comparisons use only the exactly matched GPU01 quartets.

The development host’s matrix multiply reproduced CPU03 exactly. On GPU01 it differed in 1,989 logged XY components by at most 1.20 × 10⁻⁷. Rounding each of the two float32 products before adding reproduced every GPU01 issued command exactly across all seven components. The independent checker uses that arithmetic for this job. No numerical tolerance or alteration of the recorded actions was needed. [Host arithmetic and reset checks](evidence/native01/host-readback.json)

The archive contains the 32 original episode JSONs with actual prompts, proposals, input and noise digests, issued actions and rewards, plus the official result, launcher, GPU and exit receipts. All 36 files were checked against their source SHA-256 after transfer. [Original tapes and receipts](evidence/native01/native-tapes.tar.gz) · [Original-file hashes](evidence/native01/original.sha256) · [Readback file hashes](evidence/native01/SHA256SUMS)
