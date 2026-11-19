# Public calibration rescued frozen π0.5 on two held-out interfaces

The first native reader POC passed its prespecified same-case rescue rule on both shifted conditions of LIBERO Goal 1, “put the bowl on the stove.” After a 15-control support on reset 11, a frozen factored reader selected a correction for reset 13. Both shifted queries succeeded. The nominal motor, the same reader with its support cleared, and the same reader given the wrong support all exhausted 300 controls on each shift. The arm-level readback verified reset physics, reader and motor inputs at frame zero, motor noise, opening proposals for direct-query arms, all 33 native reward sequences, fitting provenance, and the recorded decisions. [Pilot contract](FIRST-POC.md) · [Independent score and same-case witnesses](evidence/scoring/pilot.json)

## Native success by controller

Each entry is one query on the same physical start. The 180° case also scaled translation to 0.65 and inverted the gripper command. The 45° case changed yaw alone. The seven fitting conditions contained nominal and single-factor variations on two other tasks. Both the combination and the 45° angle were withheld. Every intervention operated on the same frozen motor and nominal simulated robot, with native task reward.

| Calibration available to frozen π0.5 | BASE | 180° combination | 45° yaw |
| --- | ---: | ---: | ---: |
| Nominal | 1 | 0 | 0 |
| True factors | 1 | 1 | 1 |
| Analytic wrist and jaw motion | 1 | 0 | 0 |
| Joint history transformer, active support | 1 | 0 | 0 |
| Factored reader without the equivariant head, active support | 1 | 1 | 1 |
| Factored equivariant reader, active support | 1 | 1 | 1 |
| Factored equivariant reader, passive support | 1 | 1 | 1 |
| Active reader, cleared support | 1 | 0 | 0 |
| Active reader, swapped support | 1 | 0 | 0 |
| Factored reader, 15 controls in the current episode | 1 | 1 | 1 |
| Closed-form visual-motion calibration | 1 | 1 | 1 |

The joint transformer fitted the training yaw labels to 0.7° mean error. Its errors on the composition and oblique case were 178.2° and 52.0°. The equivariant reader gave 7.3° and 36.1°. It estimated just 8.9° for the 45° case and still completed the task. Closed-form visual calibration gave 6.8° and 2.7°. This panel separates native success from accurate latent recovery, and it gives no native advantage to the equivariant head or to retaining a previous attempt. The active and passive support modes each used 465 controls across the 31 fitting/development supports. Current-prefix controls counted against the query horizon. [All live outcomes and predicted interfaces](evidence/gpu03eval/live/live-eval-summary.json) · [Frozen fit and factor errors](evidence/gpu02fit/train/offline-audit.json)

## Fresh allocation with full joint proprioception

A four-controller replay reused the frozen predictions, native starts and noise. All twelve fresh episodes had identical frame-zero inputs and the same opening motor proposal within this allocation. That proposal's hash differed from the original allocation. The nominal 45° query now succeeded at control 236, after timing out at 300 in the original experiment. The 180° composition again defeated nominal, and the calibrated controllers again succeeded. This makes the composition the reproducible native rescue in this study; the oblique rescue depended on the allocation. The source of the difference in the opening proposal remains unresolved. [Fresh paired scores](evidence/scoring/joint.json) · [Native rewards and fresh opening proposal](evidence/gpu05joint/joint.json)

| Fresh controller | BASE | 180° combination | 45° yaw |
| --- | ---: | ---: | ---: |
| Nominal | 1 | 0 | 1 |
| Retained active factored reader | 1 | 1 | 1 |
| Analytic visual calibration | 1 | 1 | 1 |
| Analytic public-joint calibration | 1 | 1 | 1 |

The public-joint baseline used a local Jacobian fitted from four nominal supports, and a wrist-and-jaw estimator for scale and grip. It reached the combined-shift goal in 95 controls, compared with 101 for the learned reader. All three control families solved this pose without tuning the VLA. The full-joint baseline provides a counterexample to the premise that proprioceptive system identification is blind to this constructed yaw. [Joint sensor predictions and calibration](../libero-xemb-joint-sysid-20261119-r1/RESULT.md)

## Eight-task motor gate

On the registered 48 matched trials, the privileged transform produced **47 successes**, versus **9** for the uncorrected motor, a gain of 79.2 percentage points. The 40 shifted trials contributed 39 versus 1. Each controller achieved 8/8 on BASE. The sole oracle failure was LIBERO-10 task 3 on the −90°/high-scale/inverted-gripper combination. The eight-task result establishes motor headroom for this action interface. The learned reader was evaluated on one development task, one query reset and one fitting seed. [Native motor pairs](evidence/gpu01gate1/gate1/gate1-summary.json) · [Independent gate readback](evidence/scoring/gate1.json)

## What the physical and sensor controls establish

- With an actual optical-axis rotation of the external camera, a 16 × 16 image template classified 21/24 development reset images, versus 8/24 after replacing each orientation with its nominal view. The seven perfectly classified tasks and one entirely missed task show how a camera benchmark can reward static appearance. The main reader pilot changes the relative command/wrist frame with fixed cameras. [Physical camera-roll study](../libero-xemb-camera-20261119-r1/RESULT.md)
- The released joint encoders also provide a reference frame. A local Jacobian fitted on four nominal support traces estimated yaw without RGB or a URDF. Absolute errors were 0.59°, 4.93° and 0.71° on the three pilot cases. The original wrist-only baseline averaged 78.52° error. [Public joint-angle calibration](../libero-xemb-joint-sysid-20261119-r1/RESULT.md)
- Reducing actual finger friction changed contact while leaving the measured precontact channels identical in the initial bottle-placement pilot. Across eight development tasks, replaying the fixed default tapes gave 8/8 ordinary and 6/8 reduced-friction successes. Choosing between default and paced tapes with the true material tied fixed default at 14/16. These open-loop endpoints are limited by the recorded tape lengths. This menu provides no measured aggregate benefit to material identification. [Eight-task friction readback](../libero-xemb-friction-dev-20261119-r1/RESULT.md) · [First-contact controls](../libero-xemb-contact-20261119-r1/RESULT.md)

The demonstrated mechanism is inference-time calibration of a constructed action interface from public interaction, with zero gradient updates at deployment. The reader used supervised training-factor labels before freezing. Real arm or tool transfer, a benefit specific to learned vision, and a need for cross-attempt memory remain open. The proposed next discriminator is a held-out change to the physical tool center or kinematics for which the present static-image and joint-Jacobian controls both lose native success while a privileged correction recovers it. We stopped at the first POC. The 96-case, three-seed reader panel and official test split remain for a later stage. All five Baige jobs have reached a terminal state. Combined scheduler runtime was 0.559 of the 12 GPU-hours available, including the unscored attempt. [Original design](PLAN.md) · [Execution and source revisions](EXECUTION.md)
