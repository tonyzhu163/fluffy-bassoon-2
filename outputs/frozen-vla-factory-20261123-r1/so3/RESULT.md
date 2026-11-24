# A gain-aware SO(3) fit improves weak-support rotation error by 0.34°

On the 48 less-excited UR5e supports, fitting three positive gains with a proper rotation cut mean compound-axis error from **29.74° to 29.40°** relative to raw polar alignment (paired reduction **0.34°**, 95% whole-task bootstrap **0.15–0.53°**). It beat raw polar on 58.3% of draws. Ordinary least squares reached 85.95° on the same supports. The 90th-percentile error of the constrained head was still 59.06°. These are offline identification errors for synthetic coordinate changes on recorded motions.

| Head | Weaker half, mean error | Control minus gain-aware, paired 95% interval | Stronger half, mean error | All supports, inverse RMS |
| --- | ---: | ---: | ---: | ---: |
| SO(3) + positive diagonal gains | 29.40° | Reference | 17.40° | 0.433 |
| Raw polar + scalar gain | 29.74° | 0.34° [0.15, 0.53] | 17.42° | 0.422 |
| Full-map least squares | 85.95° | 56.55° [48.94, 63.49] | 65.72° | 1.864 |
| Full-map ridge toward zero | 29.77° | 0.37° [0.15, 0.58] | 17.51° | 2.913 |
| Full-map ridge toward raw polar + scalar gain | 29.74° | 0.34° [0.14, 0.53] | 17.41° | 0.421 |

Inverse RMS is the root mean squared vector error after correcting the three unit canonical translation commands and applying the true map. Every head uses the same [0.25, 4] singular-value bounds during inversion. On weaker supports the constrained and raw-polar heads scored 0.462 and 0.478 (paired raw minus constrained 0.017, interval −0.007 to 0.037). On stronger supports the constrained head increased inverse RMS by 0.037 [0.018, 0.056]. Ridge toward zero shrank weakly supported directions and scored poorly on inversion despite recovering a similar polar rotation. With isotropic injected gains, raw polar had **lower** overall rotation error, 21.20° versus 21.58° (paired advantage 0.38° [0.20, 0.53]). [All contrasts and task means](metrics.json)

## What the heads received

The [Panda development replays](../../libero-xemb-friction-dev-20261119-r1/cpu01/sources.json) supplied 72 nonoverlapping windows of frozen π₀.₅ actions and measured world-frame tool positions across eight tasks. The mean of the eight per-task fitted motion scales was 0.05883 metres per unit of five-control mean command. Source geodesic error on single-axis draws selected shrinkage 1.0 for each regularized model. The gain-aware head alternates proper Kabsch alignment with a positive, regularized diagonal least-squares update. The raw-polar head aligns commands directly to displacements and fits one scalar gain. Full maps use their closest proper rotation for angular scoring.

The target [UR5e joint-FK export](../../symmeta-camera-command-20261123-r1/WORLD.md) preserves measured XYZ motion and clipped issued chunk means from 96 passive frozen π₀.₅ supports. Each head saw only three relabeled command means and their three observed world-frame displacements, from up to 15 controls. There was one start and 12 native interfaces per task. All eight task IDs differ from the source set. The [original collector](../../libero-xemb-ur5e-scale-20261120-r1/eval_test_official.py) was deployed at `5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b`. The archived π₀ cross-arm export supplies planar commands and displacements only. This 3D transfer measurement covers π₀.₅.

The script changes command coordinates by the inverse of a fresh three-axis rotation and diagonal gain, keeping recorded motion, lag and tracking residuals together. Only the scorer reads the native yaw and scalar gain to compose the label. Both gain conditions share 32 frames per support (3,072 paired cases each). The taskwise weaker and stronger halves had median original σ₃/σ₁ of 0.0063 and 0.0261. Intervals resample eight whole tasks 10,000 times after averaging draws per trace. They are conditional on these starts, fits and draw distribution. Of 3,072 anisotropic draws, 354 produced an out-of-box mean command. Restricting the weak half to its 1,373 in-box draws gave a 0.27° [0.08, 0.47] gain over raw polar. The archive cannot certify bounds on each relabeled primitive from the means.

**Recommendation.** Retain raw polar with a scalar gain as the baseline for a passive action head. The three-gain fit had a small rotation advantage on weak, anisotropic supports, and its paired interval for the inverse-error difference included zero. Near-collinear motion leaves twist around the dominant motion direction weakly observed. To test whether the residual can be reduced, compare the heads at a matched support budget with a deliberately different second motion direction. This screen assumes registered FK, alters command coordinates offline and measures neither visual motion quality nor native reward. The earlier [30-control active SO(3) benchmark](../../xemb-icml-campaign-20261121-r1/evidence/stage2d-so3-summary.json) used a different sensor and support protocol.

## Reproduction

From the repository root on the lane revision, with NumPy 2.5.4 in the repo environment, run

```sh
d=outputs/frozen-vla-factory-20261123-r1/so3
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python "$d/screen.py"
```

Research baseline `757a44cd4374833cad790852df8b87ddf412e638`. Local CPU execution receipt `091653` exited 0 in 5.92 s. Independent process `217ea6` exited 0 and reproduced the metrics byte for byte (SHA-256 `31c7059eed4f1a71370ee6e27e0371db82f00899698e477a6582a6e30355351f`). Both ran numerical checks for known rotations, gain order and inverse correction. Accelerator jobs, GPU-hours and motor parameter updates were all zero. [Plan](PLAN.md) · [Executable](screen.py)
