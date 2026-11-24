# Calibrated support yaw improved native UR5e retry success

**The calibrated reader completed 41 of 64 retries. The Perceiver trained with the same camera calibration and source augmentation completed 32.** On identical starts, A alone succeeded 15 times and B alone succeeded six times. The net gain was nine completions, or **14.06 percentage points** (95% paired whole-task bootstrap interval **+3.13 to +26.56**). Every assigned support view was physically re-rendered in simulation. [Registered comparison](PLAN.md) · [64 paired outcomes and applied angles](evidence/gpu01/paired-outcomes.csv) · [Native results](evidence/gpu01/native-results.json)

## Problem and hypothesis

The prior PR7 study found lower upright yaw error for the tuned Perceiver on UR5e, yet it completed fewer OOD retries than SymMeta. Both readers changed gain and gripper as well as yaw in that study. A subsequent mechanism test showed that moving the adapter's image coordinates could substantially shift inferred command yaw on unchanged physics. We tested whether putting observations into a calibrated reference view before estimating yaw improves *task completion* beyond a learned, pose-conditioned Perceiver exposed to the same camera and control variations. [Prior retries](../symmeta-perceiver-retry-20261123-r1/RESULT.md) · [Mechanism](../symmeta-perceiver-mechanism-20261124-r1/RESULT.md)

## Test and controls

Before receiving new model predictions, we fixed seed 41, 64 OOD task–map starts, the direction of the support-camera orbit, the BASE retry layout and the native endpoint. Each task and each map had four +30° and four −30° support views. Both trained readers used the same 15 issued controls on layout 11 and the same public calibration. A explicitly registered the observed images to the reference frame. B consumed the observed images and calibration in its learned reader. Source-only selection chose the same 5,000-update checkpoint budget for both. The parent released the full physical panel after paired offline scoring. [Registered cases](cases.csv) · [Post-view predictions and hashes](evidence/paired-post-angle.json.gz) · [Parent release](evidence/parent-release.txt)

| Correction on the frozen π₀.₅ retry | Mean yaw error | Native success |
|---|---:|---:|
| A, calibrated reference view | 11.47° | 41/64 (64.1%) |
| B, pose-conditioned, camera-augmented Perceiver | 16.95° | 32/64 (50.0%) |
| Shipped frozen Perceiver on the challenged view | 26.88° | 18/64 (28.1%) |
| True yaw, gain and gripper map | 0° | 52/64 (81.3%) |

A's yaw error was 5.48° lower than B's on the assigned views (95% paired task interval for the reduction 2.87° to 7.84°). The shipped reader's offline nominal-view error was 5.86° on these same replays. The parent selected the four native arms in the table and omitted nominal-view and identity retries. A, B and shipped challenged shared the same gain and gripper estimates from PR7's seed-41 tuned reader on the corresponding original nominal-view support. Only yaw varied across these three native arms. True map changed all three factors and gives an empirical task-success reference. [Exact native corrections](evidence/prepare01/panel.json.gz) · [Release inputs](evidence/decision.json)

## Paired native evidence

| Task | A / 8 | B / 8 | A only | B only |
|---|---:|---:|---:|---:|
| LIBERO-10 5 | 3 | 2 | 2 | 1 |
| LIBERO-10 8 | 1 | 2 | 0 | 1 |
| Goal 3 | 4 | 1 | 3 | 0 |
| Goal 6 | 6 | 6 | 1 | 1 |
| Object 1 | 6 | 4 | 3 | 1 |
| Object 8 | 7 | 4 | 3 | 0 |
| Spatial 0 | 8 | 7 | 1 | 0 |
| Spatial 8 | 6 | 6 | 2 | 2 |

A gained completions on five tasks, tied on two and lost one completion on LIBERO-10 8. At −30°, A and B completed 25 and 20 of 32. At +30°, they completed 16 and 12 of 32. Compositional maps gave 21 versus 16 successes and continuous-angle maps gave 20 versus 16. All 64 starts contributed, including starts where A's yaw error increased. The primary interval resamples the eight whole tasks, keeps all eight maps and both arms together, and uses the exact distribution over 8⁸ task draws. It is conditional on the single reader seed and registered layouts. [Task, sign and regime results](evidence/gpu01/native-results.json)

Across the eight task means, the sample covariance of B−A yaw-error reduction in degrees with A−B native success gain in percentage points was **+22.10 degree·percentage points**. The A/B task-mean success covariance was 0.0625 on the probability scale. On Object 1, A completed two more retries even though its mean yaw error was 0.39° higher. The covariance describes these eight tasks. The complete result retains within-task paired native success covariance. [Per-task measures](evidence/gpu01/native-results.json)

Both arms used the same frozen motor and native UR5e OSC controller. Within each start, all four episodes ran in one process on one RTX 4090 with identical release-row reset, prompt, raw camera observations, policy-noise bank and first uncorrected motor proposal. Later proposals were closed loop. The in-job scorer verified physical commands before recording case success. Independent CPU readback checked 64,249 native controls and 12,899 policy calls, reconstructed all outcomes from native rewards and termination, and reproduced the exact paired interval from the 64 recorded cases. Job `job-t8kuarphk38v` exited 0 after 3,345 launcher seconds, **0.929 GPU-hours**. [Execution record](EXECUTION.md) · [Probe](evidence/gpu01/gpu.json) · [Independent readback](evidence/gpu01/independent-readback.json)

## Update and scope

This result raises confidence that explicit reference-view registration can turn better physical-support yaw estimates into more successful frozen-policy retries than equally informed learned pose conditioning. It supports that decision for the tested oblique support cameras, held-out control maps and fixed nuisance estimates. The controls isolate the effect of substituting the two learned yaw outputs in this native interface. They cannot attribute each rescue to a smaller absolute angular error. The true-map arm still failed on 12 starts.

The camera manipulation was confined to a separate calibration support. The physical support replay held each archived five-control chunk mean for five controls. Each replay's nominal and moved cameras observed exactly the same control sequence and motion. The retry used the released agentview and wrist camera, the native command and proprioception shift, and layouts 26 or 27. The experiment does not measure how the frozen motor itself performs after its cameras move, or how a complete adapter estimates gain and gripper from a single challenged camera. There were 64 physical-support cases and zero digital-only native cases. [Verified view and trajectory joins](evidence/qualified-view-inputs.json)

Raw per-control records, full proposals and noise/reset fingerprints remain on Baige under `baige-dev:/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-frame-native-20261124-r1/gpu01/pairs/`. Each record's SHA-256 is included in the portable case table. The CPU qualification caught an oblique-angle rounding discrepancy in the readback. The corrected in-job verifier matches the native simulator's per-control forward multiplication exactly; the independent cross-host verifier checks a float64 reference with a bound derived from float32 roundoff. [Qualification diagnosis](evidence/cpu02/diagnosis.txt) · [Final CPU qualification](evidence/cpu03/cpu-readback.json) · [Portable verifier](portable_readback.py)
