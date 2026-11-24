# The frozen motor uses the weak support direction during later manipulation

**Pursue a low-cost test of query-directed calibration.** On all eight held-out UR5e tasks, the planar direction least covered by the 15 passive support controls carried more energy on an independent frozen-policy query. Its share rose from **0.63% on support to 34.24% on query**, a paired **+33.61 percentage points** (95% whole-task bootstrap interval **+23.92 to +44.55**). The query’s own opening also followed the support line. Its first 15 controls put 3.75% of XY energy along the weak direction, rising to 41.95% thereafter. The late-minus-early difference was positive on every task. This establishes downstream command demand. The experiment leaves the value of an intervention and its effect on reward open. [Registration](PLAN.md) · [Task estimates and intervals](cpu02/summary.json) · [Independent raw-tape readback](cpu02/readback.json)

## Which commands the comparison measures

Each arm supplied one 15-control passive layout-11 support and one complete frozen OpenPI π₀.₅ query on independent official BASE layout 14 for each of eight tasks. UR5e contributed 1,522 executed controls in 307 replans, IIWA 1,493 in 301. All sixteen raw episodes were included. Both panels used the native world-frame relative OSC_POSE controller at 20 Hz. The nominal yaw, gain and sign make the simulator-bound local commands equal to the bounded, postprocessed motor commands. The trace contains the executed prefix of each proposal, including partial final chunks. Each later proposal was generated from that same arm’s actual feedback. The shifted late-oracle archive retains hashes of its numerical commands. Directional demand under a physically imposed general 2×2 map remains unmeasured. [Native tape capture](../frozen-vla-native-transfer-20261123-r1/native_rollout.py) · [Official query and model receipts](../frozen-vla-multiarm-confirmation-20261123-r1/evidence/production-native.json) · [Shifted archive schema](../symmeta-late-oracle-20261123-r1/rollout.py)

Each support retains the means of three consecutive five-control chunks. For its 3×2 uncentered issued-XY matrix, we took the right singular vector with the smaller singular value. For each query, we divided the sum of squared projections of the chunk means on that vector by total squared XY magnitude, weighting by actual chunk length. Results average tasks equally. Intervals resample whole paired tasks 20,000 times and condition on these layouts, policy-noise draws, calibrated supports and reader fits. IIWA reuses the eight task identities. The nominal settings are a subset of the broader PR7 support panel. [Scorer](score.py) · [Per-tape identities, hashes, support vectors and measurements](cpu02/tasks.json)

| Weak-direction energy or paired difference | UR5e | IIWA |
| --- | ---: | ---: |
| Layout-11 support | 0.63% | 0.64% |
| Whole layout-14 query | 34.24% | 34.24% |
| Query minus support, pp [95% interval] | +33.61 [+23.92, +44.55] | +33.60 [+24.89, +44.98] |
| Query controls 1–15 / controls 16–end | 3.75% / 41.95% | 4.04% / 40.74% |
| Controls 16–end minus 1–15, pp [95% interval] | +38.20 [+29.47, +47.22] | +36.70 [+28.60, +46.12] |
| Query controls 1–25 / controls 26–end | 4.47% / 50.43% | 4.34% / 49.05% |
| Controls 26–end minus 1–25, pp [95% interval] | +45.95 [+36.94, +54.44] | +44.71 [+36.54, +53.13] |
| Whole query using individual controls | 34.56% | 34.54% |
| Whole query using other tasks’ weak axes | 50.00% | 51.29% |

The support’s own weak fraction is small by construction. The independent early-to-late rise removes that self-comparison. The seven wrong-task axes also show that the matched support geometry retains predictive information about the whole query. Matched minus wrong-task fraction was −15.76 pp on UR5e and −17.05 pp on IIWA. A uniformly random fixed 2D direction has expected fraction 50% on any nonzero tape. Re-expressing both query and support in a frame rotated by 90° preserved all primary fractions. The individual-control sensitivity gives almost the same whole-query fraction, although individual support controls are unavailable. [Directional controls and temporal splits](cpu02/tasks.json)

## Magnitude and phase

The common XY output scale is 50 mm per unit of native command. On UR5e, total planar RMS requested setpoint offset fell from **36.14 mm on support to 24.65 mm on query**, while the weak component rose from **2.33 to 13.78 mm**. It measured 5.65 mm in the independent first 15 controls and 14.42 mm thereafter. Controls 16–end contained 77.7% of query XY energy on the average task. The fraction therefore grew at a meaningful absolute command magnitude. The corresponding IIWA weak component rose from 2.11 to 13.80 mm. These offsets specify controller targets. The displacement residuals below use measured end-effector motion. In native command units, the UR5e and IIWA whole-query XY RMS values were 0.493 and 0.488. Total duration-weighted chunk-mean XY energy across tasks was 323.09 and 325.46 squared command units, respectively, versus 65.23 and 58.96 on their supports. [Absolute and normalized energy by task and window](cpu02/summary.json)

| Query phase, individual-control XY | UR5e controls / tasks | UR5e weak energy | IIWA controls / tasks | IIWA weak energy |
| --- | ---: | ---: | ---: | ---: |
| Before first positive closing command | 716 / 8 | 18.0% | 721 / 8 | 16.4% |
| Positive close or hold command | 667 / 7 | 55.9% | 586 / 7 | 53.7% |
| Later open or release command | 139 / 4 | 59.5% | 186 / 6 | 43.1% |
| Object carried throughout a five-control same-layout replay block | 440 / 7 | **57.4% [40.2, 74.7]** | 395 / 7 | **55.6% [39.6, 72.9]** |

The first three rows partition actual query controls. The last row selects a subset of close commands using archived physical contact observations on a replay of the *same commands, body, start and seed*. Each of its five endpoints had the same free object in bilateral finger contact without external support. The UR5e replay identified 88 such blocks out of 125 eligible blocks. They carried 43.2% of all query XY energy on the seven contributing tasks, with a mean 19.64 mm weak RMS requested offset. IIWA had 79 of 104 blocks. Goal 3 issued no close command on either arm. A later open command establishes the attempted release phase only. All UR5e replay reward tapes and durations matched the originals, with maximum TCP and finger deviations of 0.44 and 0.70 mm. Two IIWA replay reward tapes diverged, and spatial/0 reached a 32.16 mm TCP discrepancy. Its contact labels describe the replay. All command projections use the original frozen-policy tapes. [Contact definition](../frozen-transfer-object-20261124-r1/capture.py) · [Task-level phase coverage and replay fidelity](cpu02/tasks.json)

## Task heterogeneity

Fractions are percentages. The two difference columns are percentage points. The carry count is the number of physically labeled five-control replay blocks. One official query start per task precludes within-task sampling intervals.

| Task | UR5e controls | Support | Query first 15 | Query 16–end | Full query | UR5e query − support | Carry blocks | IIWA query − support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| libero_10/5 | 182 | 0.04 | 0.9 | 35.7 | 32.2 | +32.2 | 20 | +31.9 |
| libero_10/8 | 470 | 1.72 | 8.8 | 67.9 | 65.2 | +63.5 | 26 | +68.6 |
| libero_goal/3 | 300 | 0.09 | 0.5 | 54.0 | 45.9 | +45.8 | 0 | +39.4 |
| libero_goal/6 | 123 | 0.11 | 0.1 | 31.9 | 19.1 | +19.0 | 6 | +17.6 |
| libero_object/1 | 129 | 0.86 | 6.2 | 50.6 | 43.1 | +42.2 | 10 | +35.0 |
| libero_object/8 | 149 | 1.41 | 7.7 | 26.0 | 23.5 | +22.1 | 16 | +21.6 |
| libero_spatial/0 | 81 | 0.68 | 2.2 | 28.5 | 17.9 | +17.2 | 6 | +26.1 |
| libero_spatial/8 | 88 | 0.08 | 3.4 | 41.0 | 27.0 | +27.0 | 4 | +28.6 |

## What the fitted readers establish

The known nominal inverse reproduces each raw command exactly. For each matched support, we also applied the archived seed-41/42/43 tuned Perceiver and paper SymMeta yaw/gain inverse to each *logged raw proposal prefix* with native clipping. On UR5e, equal-task, seed-averaged RMS XY error against the oracle was **3.10 mm tuned and 3.15 mm SymMeta**, paired difference −0.05 mm [−1.09, +1.03]. IIWA gave **2.27 and 2.15 mm**, difference +0.12 mm [−0.75, +0.92]. These pointwise errors apply at states visited by the raw motor. Later states and rewards under a changed control would require fresh feedback. The retained PR7 full-map run saves aggregate errors across relabeled general maps and does not expose their per-support predictions or trained full-map weights. An error for those fitted general maps on these queries cannot be recovered from its summary. [Archived factor-fit contract](../symmeta-perceiver-retry-20261123-r1/evidence/prepare01pro/contract.json) · [Per-seed held-command errors](cpu02/tasks.json) · [PR7 output code](../symmeta-wider-family-20261123-r1/eval_wider_family.py)

For illustration, let v be the support’s weak unit axis. The identity and the reflection I−2vvᵀ give the same response on the perpendicular line, and their pre-clipping difference at command q has norm 2|q·v|. More generally, two maps agreeing on that line differ by at most their operator-norm separation times |q·v|. Equality on every support command follows only in the noiseless rank-one case. The eight measured UR5e supports had finite second singular values. For these illustrative maps, twice the measured weak RMS gives a 4.65 mm support versus 27.55 mm query pre-clipping discrepancy, averaged over tasks. Component clipping can reduce it. Neither map was fitted in this illustration.

There is an empirical reference scale for predicting *motion* from support, in different units from those requested setpoint discrepancies. The original fitted current-and-previous-command law on 192 nominal Panda/Sawyer supports had 3.97 mm planar and 4.23 mm 3D RMS measured displacement residual per five controls (576 observations). Those in-sample residuals contain dynamics and model error. The archive supplies no repeated identical-motion observation series that isolates a transferable sensor-noise floor here. It would be unjustified to declare two finite-support maps observationally indistinguishable from this comparison. [Source fit](../frozen-vla-motion-models-20261123-r1/evidence/source-fit.json) · [Residual readback](cpu02/summary.json)

## Updated prior and next decision

The unused-direction explanation loses support on both nominal arm panels. The motor repeatedly asks for the weak component later, including while carrying an object. The support still predicts the opening direction, and the two existing readers have similar held-command error on the identity interface. This gives a reason to test **when another directional observation is worth obtaining**. No result here establishes that an active architecture outperforms the tuned Perceiver.

The next discriminating comparison would hold the actual general command map, frozen motor, released start, estimator and control budget fixed. At a replan, use only the current proposal and past support to choose the direction of one short intervention. Compare it with using the same number of ordinary policy controls for an online passive update, alongside the frozen tuned-Perceiver baseline. A later command can itself supply the missing excitation. If passive updating yields equally accurate held-command corrections, the extra intervention has little rationale. The prior fixed three-direction probe worsened forecasts on every target arm. This result warrants testing the narrower, causal decision only. [Prior probe result](../frozen-vla-motion-models-20261123-r1/HANDOFF.md) · [Task-optimal exploration motivation](https://proceedings.mlr.press/v139/wagenmaker21a.html)

## Execution

CPU02 ran on Baige development CPUs 38/39 at Git pin `3a6f90a25fbb6b2de0b06b02a9c11aff1f34b3f3`, with 2.26 s wall time and 0.57 s active scoring CPU. GPU usage was zero. CPU01 stopped before scoring because the independent manifest inlines numerical opening controls only for the first task. CPU02 checked the registered shape/dtype/bytes opening digest on all 16 original raw traces and retained full-receipt hashes. A second calculation using the analytic 2×2 principal-axis angle and the original raw tapes reproduced all 32 support and query fractions to within 1.4 × 10⁻¹⁶. [CPU01 stop](cpu01/run.log) · [CPU02 invocation](cpu02/launch.txt) · [Timing](cpu02/time.txt) · [Raw readback](cpu02/readback.json)
