# Frozen-VLA architecture factory

The factory tests which information a lightweight adapter needs to control an unfamiliar robot with π₀ or π₀.₅ frozen. The primary benchmark is LIBERO-Pro2.0. Each lane starts from revision `757a44cd4374833cad790852df8b87ddf412e638`. For native comparisons, use paired task starts, identical proposal noise and native task reward. Keep each support budget and observation privilege explicit.

Workflow `wfr-92d79c14c63342a292b2e0ea047f678a` launched all twelve workers. The native worker owns accelerator submissions with an incremental limit of 12 GPU-hours. Eleven other workers have CPU-only allocations. Existing Perceiver and FK jobs retain their original owners. The archived workflow preserves the exact lane prompts, response schema and stop conditions for this run. [Executed workflow source](workflow.py)

Three additional CPU workers test adapters applied directly to the inputs of a public LIBERO-trained SmolVLA. These comparisons use frozen forward passes with paired action noise. The prior [S15 camera experiment](../xemb-camera-dev-20261122-r1/RESULT.md) motivates this test because native success fell to zero even with the exact action transform.

| Worker | Experiment |
| --- | --- |
| [map](https://app.devin.ai/sessions/35af26b68f86493290db17816027c03e) | Constrained residual versus ridge and similarity for general planar maps |
| [excitation](https://app.devin.ai/sessions/e931694d8a8c446ebe485c1920bf7db9) | Command directions at a fixed support budget |
| [delay](https://app.devin.ai/sessions/cf7253586c924edc879ee411823217b3) | Causal temporal response and spatial gain |
| [camera](https://app.devin.ai/sessions/d55ace1279f14de6b95656e804cbde66) | Reference-frame correction under camera perturbation |
| [fusion](https://app.devin.ai/sessions/816c249b0bd74fb8bcab93218ab7a562) | Confidence-weighted visual and kinematic yaw |
| [so3](https://app.devin.ai/sessions/896f5365a7114137a24de6b1df8657c8) | Gain-aware rotation in three dimensions |
| [contact](https://app.devin.ai/sessions/b231564ab95b43f9a42a2cb533444d8c) | Causal contact-conditioned action selection |
| [gripper](https://app.devin.ai/sessions/ea746bd559534c11a167b2c30d159808) | Direction-dependent gripper response |
| [morphology](https://app.devin.ai/sessions/15f0e8be884c4f949d0ab08a5518ae50) | Shared versus arm-specific response across robot bodies |
| [timing](https://app.devin.ai/sessions/087df1d39f1f44e788d28e5ea0048e4f) | Phase-dependent pacing versus fixed execution |
| [uncertainty](https://app.devin.ai/sessions/f254f8fea7cd4ee3b6e52e24decf5838) | Selective adaptation with paired native outcomes |
| [native](https://app.devin.ai/sessions/e44daa6ab0e641aea157c4586a8f71f1) | Matched frozen-policy native rollout qualification |
| [view](https://app.devin.ai/sessions/19e372773f55456cbc86ebb7fb3ccaf9) | Canonicalize the input view before frozen proposal generation |
| [state](https://app.devin.ai/sessions/22460af5b549476e853fb7b5e1c5b01b) | Map target proprioception to the motor's state representation |
| [appearance](https://app.devin.ai/sessions/a51278acfb5e47d2859b4f2cde292d5a) | Control for changes in arm appearance at a fixed viewpoint |

The coordinating session also ran [temporal pooling](pooling/RESULT.md) for both archived visual readers. Each readout reproduced its frozen archive baseline on all 96 UR5e supports.

On the original paired native retries, the smaller support yaw error identified 12 of the 15 held-out cases in which only one reader succeeded. [Calibration metric versus native success](calibration-metric/RESULT.md)

## Architecture decisions from the first twelve workers

| Decision | Evidence and scope |
| --- | --- |
| Begin with a scaled rotation on the recorded yaw/gain interface. | Across 192 passive UR5e supports, a full ridge fit had forward error 0.262 against 0.080 for scaled rotation. A low-rank morphology update also lost to scaled rotation on the recorded cross-arm displacement screen. [Excitation](excitation/RESULT.md) · [Morphology](morphology/RESULT.md) |
| Use an independent second direction when the interface may shear or reflect. | On 384 new simulated UR5e supports with matched jitter, two-direction probing reduced relative full-map error from 0.560 to 0.270. On 360 archived FK supports with a synthetic mirror, a parity-aware fit reduced direction error from 90.00° to 4.61°. Weak two-direction supports also benefited from damping the inverse, reducing clipped command error from 0.308 to 0.184 on frozen action tapes. [Support directions](excitation/RESULT.md) · [Parity](map/RESULT.md) · [Inverse damping](inverse/RESULT.md) |
| Register the support view to a common frame. | With trusted extrinsics on physical ±30° camera orbits, mean visual yaw error fell from 28.98° to 7.61°. Trusted FK was more accurate than the tested fusion rules under optical rolls. The frozen motor's query camera stays fixed in the prepared native retry. [Camera](camera/RESULT.md) · [Fusion](fusion/RESULT.md) · [Native qualification](native/RESULT.md) |
| Estimate delay before gain when calibrating a delayed actuator. | On the archived S20 trajectories, using the true delay reduced gain MAE from 0.593 to 0.027. Source-calibrated correlation plus a finite impulse response fit recovered 31/32 injected FIFO delays and 27/32 delays on S20. [Delay](delay/RESULT.md) · [Timing](timing/RESULT.md) |
| Keep simple baselines for the remaining heads. | The gain-aware 3D fit improved the weaker half by 0.34°. Motion-only jaw forecasting had lower one-control error than bounded memory. The contact gate tied unchanged continuation at 27/32, and the uncertainty rules gave no held-out native advantage over the fixed reader. [3D rotation](so3/RESULT.md) · [Gripper](gripper/RESULT.md) · [Contact](contact/RESULT.md) · [Uncertainty](uncertainty/RESULT.md) |

The native lane qualified a six-condition, 96-episode camera comparison on a fixture. Factory GPU consumption was zero at qualification. The 96-episode comparison needs authenticated Baige Distributed Training access and a verified new job receipt. The cameras, interfaces, observation privileges and motor checkpoints vary across these screens. A pooled success rate across lanes would have no common denominator.

## Physical arm changes at the native command convention

The two-direction result concerns identification of a deliberately changed planar command map on UR5e. A concurrent repository study tested a different question on stock arms. Its three-direction probe increased error when predicting subsequent policy motions on Kinova3, IIWA and UR5e relative to passive policy-like calibration. Historical unadapted π₀.₅ succeeded on 8/8 nominal UR5e queries and 6/8 each on Kinova3 and IIWA. A useful next native test includes untouched source and target arms, checks for regressions on nominal commands, and crosses input appearance and command correction on the same task starts. [Independent motion study](../frozen-vla-motion-models-20261123-r1/HANDOFF.md) · [Native arm panel](../frozen-vla-native-transfer-20261123-r1/HANDOFF.md)

On ten pose-matched Panda/UR5e rendered LIBERO-Goal pairs, changing only the visible arm changed frozen SmolVLA translation proposals by 0.0539 RMS. An aligned source-arm overlay reduced that drift to 0.0304 across ten pairs, while a target-only gray mask raised it to 0.0941. The overlay needs matching source pixels. [Appearance controls and pose audit](appearance/RESULT.md)

On public UR5e and IIWA recordings, a local affine joint-to-pose adapter reduced first-five-action RMSE from 0.1670 to 0.0143 relative to the measured tool-pose reference. It used 288 paired calibration frames from both arms on separate tasks. Removing the state gave 0.1001. Sixteen held-out frames contributed to the task-level interval. [State adapter, calibration and per-arm estimates](state/RESULT.md)

On eight Panda tasks with quarter-turn camera perturbations, an image template reduced full-chunk action RMSE from 0.2069 to 0.0172. Undoing the known mount rotation gave 0.0014. On the basket scene the template rotated the upright camera too, adding 0.1070 RMSE on the nominal input. Keep an upright preservation control when testing an installation whose mount alignment is unknown. [Camera adapter, false corrections and frozen proposals](view/RESULT.md)
