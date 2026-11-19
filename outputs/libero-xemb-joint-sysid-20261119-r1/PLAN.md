# Can public arm joints recover relative command yaw?

The released interface rotates the command and local wrist axes together, and leaves the seven arm encoders in their physical convention. A local Jacobian learned at the nominal interface may supply a stable reference for the three reader-pilot cases. This is a sensor identification experiment. Its endpoint is the error of the parameter estimate used for a subsequent native query.

## Locked protocol

Use the active support in the [first reader pilot](../libero-xemb-sysid-20261119-r1/FIRST-POC.md) and the [native archive](../libero-xemb-sysid-20261119-r1/evidence/gpu02col/archive/archive.json) from commit `65c83998`. Use the same native startup and action wrapper, BASE, both original camera configurations and the exact three bursts of five controls. Read the public joint, wrist and jaw values at 0, 5, 10 and 15 controls. The cameras render on Mesa CPU for simulator compatibility. Pixels are excluded from calibration and prediction. Capture joint velocities for provenance. The estimator uses only position differences. Four fitting supports and three evaluation supports cost 105 controls total. There is one evaluation task and one physical evaluation startup.

| Purpose | LIBERO Goal task | Support reset | Embodiment | Archive rows |
| --- | --- | --- | --- | --- |
| Fit | 5, 7 | 0, 1 each | nom only | 0, 7, 14, 21 |
| Evaluate | 1 | 11 | fam_nom, comp_180_lo_inv, cont_45_nom | 28, 29, 30 |

Use the archived seeds, in order 20110500, 20110501, 20110700, 20110701 and shared 20210111. Check the native startup physics digest and the 15 issued controls for every replay. Compare the starting public wrist and fingers and all three wrist/jaw differences against the original active tapes. Permit up to 0.01 mm replay difference for these floating-point sensors. Record the maximum discrepancy. Require complete, unterminated support in agreement with the archive.

## Frozen estimator and decision

1. Stack the twelve nominal pulse endpoint differences into Q (12 by 7, radians) and D (12 by 3, nominal public wrist metres). Fit J by solving (QᵀQ + λI)J = QᵀD without an intercept. Fix λ = 0.001 times the largest eigenvalue of QᵀQ before observing outcomes. Fit all three axes jointly, including the return/up pulse. Report singular values and XYZ residuals.
2. Predict three physical XYZ displacements as Δq J. For planar yaw, use the angle of the summed command–predicted-motion dot and cross products over the three pulses, using XY only after the XYZ fit. For relative yaw, subtract the corresponding pooled angle between nominal training commands and observed nominal wrist motion. This corrects a fixed turning bias from the pulse sequence and controller lag. Wrap to [−180°, 180°). Save the raw uncorrected alignment for inspection. The yaw function receives only issued commands, seven joint position differences and train-frozen coefficients.
3. Use the pilot’s analytic proprioceptive comparator with local wrist XYZ, jaw change and issued commands. Calibrate its scalar gain on the same four nominal fitting traces. Retain its original wrist yaw, and also report the wrist yaw with the same nominal offset correction. Use its scale and gripper sign for the live handoff in both arms. Only yaw is claimed from the seven joints. No URDF, images, privileged kinematics, evaluation wrist targets, condition labels or fitting on evaluation cases enter joint yaw inference.
4. Score all three circular absolute errors after serializing predictions. Strong recovery requires error below 20° on **each** of the three cases (the pilot uses 20° as its continuous-angle gate), and smaller error than both wrist yaw comparisons on **each shifted** case. A miss on either shifted case counts against strong recovery. No tuning or further starts. Save the three predictions in the native motor's yaw-degree, positive-scale, signed-gripper convention, keyed by support identity for the parent to use at query reset 13. Native success requires a separate paired live query.

Single CPU thread per numerical library, hidden CUDA devices, software EGL, seven supports, 0 GPU-hours, no VLA or test partition. Stop at the result or a replay mismatch. The plan and executable code are committed before collection. The parent owns coordination, findings and push.
