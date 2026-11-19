# Public arm encoders recover relative yaw on the reader pilot

**The seven public joint positions and issued commands recovered yaw on all three held-out Goal 1 interfaces.** Mean absolute circular error was **2.07°**, compared with **78.52°** for the pilot’s wrist yaw and **6.37°** for its archived image-motion estimator. Both shifted cases beat the original and nominal-referenced wrist controls. All three joint estimates passed the registered 20° per-case criterion. This falsifies a requirement for vision to identify yaw from the released raw support on this panel. The existing compact reader tape omits the seven joint positions. [Registered plan](PLAN.md) · [Scores](cpu01/scores.json) · [Original pilot fit](../libero-xemb-sysid-20261119-r1/evidence/gpu02fit/train/offline-audit.json)

| Goal 1 support, reset 11 | True yaw | Joint yaw | Joint error | Wrist error | Visual error¹ | Scale estimate (true) | Grip estimate (true) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | 0° | 0.59° | 0.59° | 8.59° | 9.63° | 0.9993 (1.00) | +1 (+1) |
| Withheld composition | 180° | −175.07° | 4.93° | 172.42° | 6.81° | 0.5802 (0.65) | −1 (−1) |
| Unseen oblique yaw | 45° | 44.29° | 0.71° | 54.56° | 2.68° | 0.9847 (1.00) | +1 (+1) |

¹ The visual errors come from the original pilot’s active-support audit on the same archive. Scale uses local wrist XYZ and grip uses signed finger motion in both sensor arms. The common scale estimate was 10.7% low on the composition. [Comparison readback](cpu01/readback.json) · [Per-case predictions](cpu01/predictions.json)

## What identified yaw

Four nominal supports on Goal 5 and 7, resets 0 and 1, provided twelve joint-space and wrist XYZ displacements. A single ridge-regularized 7 × 3 local Jacobian mapped the joint changes to the nominal physical wrist axes. Its fitting RMS errors for X, Y and Z were 0.43, 0.30 and 0.54 mm. On Goal 1, the joint yaw estimator used the three issued XY commands and the joint motion at the same pulse endpoints. Positive X and positive Y pulses gave directional information. The return/up pulse entered the full XYZ calibration. [Frozen estimator](joint_reader.py) · [Fitted coefficients and singular values](cpu01/calibration.json)

The registered nominal alignment correction was −8.75°. Without applying it, the joint estimator still had errors of 8.16°, 3.82° and 9.46°. Applying that same correction to the wrist gave errors of 0.17°, 178.83° and 45.80°. The wrist recovered the nominal reference on the shared start and remained near zero on both shifts. The unrotated joint excursions carried the shifted-direction signal. [Scores](cpu01/scores.json)

## Pairing and scope

The seven native CPU replays used the pilot’s original seeds, BASE physics and fixed 15-control sequence. All seven startup physics fingerprints matched their archive rows. The recorded startup wrist and finger positions, three local wrist displacements and jaw traces matched the original active tapes with maximum discrepancy 0.0 mm. All three evaluation embodiments shared one physical startup. An independent augmented least-squares solve and complex-angle calculation recovered the serialized yaw predictions to within 2.3 × 10⁻¹³ degrees. [Native replay logs and raw joint, wrist and finger traces](cpu01/) · [Readback](cpu01/readback.json)

This result establishes observability at one development posture under the synthetic command-frame intervention, with a known nominal calibration convention. The limited excitation of this local Jacobian and the single held-out reset leave transfer to other poses, physical robot kinematics and passive motion open. The official test partition was untouched. The experiment measured sensor estimates. Native task success under the joint estimates awaits a paired motor query.

## Live-query handoff

For Goal 1 BASE query reset 13, select each row by its support archive index and matching embodiment. Pass the full-precision `yaw_deg`, `xy_scale`, `gripper_sign` triple to the same motor retargeter. Rows 28, 29 and 30 correspond to `fam_nom`, `comp_180_lo_inv` and `cont_45_nom`. The same file is on Baige under `/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-joint-sysid-20261119-r1/cpu01/predictions.json`. Its SHA-256 is `913bffcec050ca818fbc0bd04f3281887c5eceb29c7228d6fb2fad46530f79f8`. [Motor-ready predictions](cpu01/predictions.json)

The registered code ran at `ee28eb78173703f171b47eabb33baa4c113c2be4` on the Baige development CPU with Mesa llvmpipe and the qualified native release `e5500c69`. Seven supports used 105 controls in 99 seconds, with 0 GPU-hours. The existing GPU pilot supplied the read-only comparison archive. [Launch record](cpu01/launch.txt) · [Exit record](cpu01/exit.txt) · [Native collector](experiment.py)
