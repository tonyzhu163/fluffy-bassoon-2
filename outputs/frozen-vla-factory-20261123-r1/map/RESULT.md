# Shear and reflection recovery from 15 frozen-policy controls

The SVD-bounded residual reduced shear error by 1.04° versus proper Procrustes (95% paired task-bootstrap interval 0.77° to 1.33°). Its advantage over source-tuned ridge was 0.002° (0.000° to 0.005°). On mirrors it increased error by 2.36° versus ridge (0.76° to 4.21°). Allowing Procrustes to select the handedness of the map from support gave the largest mirror improvement.

| Command interface | Identity | Proper Procrustes | Ridge | Bounded residual | Either-parity Procrustes | Either-parity + bounded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recorded yaw/gain | 90.00 | 2.71 | 4.89 | 4.89 | 4.61 | 4.61 |
| Synthetic shear | 90.00 | 12.34 | 11.30 | 11.30 | 14.04 | 14.00 |
| Synthetic mirror | 90.00 | 90.00 | 79.20 | 81.56 | 4.61 | 4.61 |
| Synthetic mirror + shear | 90.00 | 92.61 | 82.42 | 84.64 | 11.24 | 11.21 |

Entries are mean commanded-direction errors in degrees, lower is better. Each inverse correction acts on 128 uniformly spaced unit directions. The known composite interface maps each output back to world commands for scoring. Source tuning, held-out scoring and the analytic noiseless check run from [screen.py](screen.py). [Metrics, paired intervals, per-task and per-panel errors](metrics.json)

## Inputs and pairing

The source comprised the first 15 default controls and measured end-effector travel from each of eight development tasks, ordinary-material replays of frozen π₀.₅. The target comprised 360 numerically qualified passive supports across eight different tasks, 96 each from UR5e × π₀ and π₀.₅, and 84 each from IIWA and Kinova3 × π₀.₅. The two seven-joint arms each lack one qualified reset task. Three issued XY means and three world XY displacements from encoder FK entered each fit, following the archive's five-controls-per-chunk convention. Absolute joint reset and arm geometry are trusted calibration inputs. The raw target tapes omit the primitive counts of the final chunk. [Source NPZ provenance](../../libero-xemb-friction-dev-20261119-r1/RESULT.md) · [UR5e × π₀.₅ reconstruction](../../symmeta-camera-command-20261123-r1/WORLD.md) · [Cross-arm qualifications](../../symmeta-fk-crossarm-20261123-r1/RESULT.md) · [OpenPI revision and weight provenance](../../symmeta-fk-crossarm-20261123-r1/QUERY_CONFIG.json)

For each synthetic family the same eight independent matrix draws are applied to all held-out tapes. New issued commands are `B⁻¹ x`, physical observations stay fixed, and the scorer composes the recorded interface with `B`. Scaling each transformation guarantees box feasibility even for individual bounded primitives. Fits see only the remapped support commands and motion. The motor, task identity, family label, original truth and scoring rays never enter the fit. Each task mean averages its paired supports, available panels and draws. The 95% percentile intervals resample eight whole tasks 10,000 times conditional on the fixed draws and source fit. The 360 tapes and their relabelings share tasks and resets.

## Where the cap fails

Ridge fits the inverse linear map from observed travel to issued direction, shrinking toward a support-fitted proper similarity. The bounded variant clips each singular value of the residual relative to the similarity's scale. Source-only mean direction error across similarity, shear and mirror selected ridge penalty 0.1. Caps 0.9, 1.5 and 2.5 tied on source; the smallest was used. The 0.9 cap preserves positive determinant because every residual has spectral norm smaller than the similarity's smallest singular value. Even the tied, parity-permitting 2.5 cap scored 79.28° on mirrors. The paired FK command singular-value ratio had median 0.095 on recorded supports, leaving little motion along the second axis.

Either-parity Procrustes selects the better orthogonal fit using only the three observed pairs. It recovered mirror parity on 352/360 FK supports. Its mirror error equals its recorded error under orthogonal coordinate relabeling. This symmetry explains the repeated cells. The same estimator recovered parity on only 76/96 UR5e × π₀.₅ supports with upright, source-calibrated brightness-flow displacement, scoring 27.08° on mirrors versus 4.69° using FK on those same tapes. On visual shear, proper Procrustes reached 16.23°, ridge 18.85° and the bounded residual 18.36°. The FK shear improvement reversed with this noisier displacement measurement. [Visual projection](../../symmeta-camera-command-20261123-r1/score.py)

Advance an explicit parity choice for interfaces that may reflect. Retain proper Procrustes as the recorded-family control and source-tuned ridge as the shear control. A native retry should check the effect of incorrect parity under weak second-axis motion before deploying a parity switch. This offline counterfactual changes coordinates on fixed trajectories. It measures direction independently of gain and leaves on-policy excitation, command magnitude, clipping on retry, and task success unmeasured. The source tuning panel has only eight episodes.

## Reproduction and receipt

Input revision `757a44cd4374833cad790852df8b87ddf412e638`. Run from the repo root at the lane commit in the project's NumPy environment. The 32 source draws per training family use seed 4101. Eight test draws per synthetic family use seed 4199. Task bootstrap seed 4137. The mirror–shear composition was held out of source tuning.

```sh
env CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src uv run --no-sync python \
  outputs/frozen-vla-factory-20261123-r1/map/screen.py
```

Local Linux CPU job completed twice with NumPy 2.5.4, exit 0 in 2.29 s and 2.58 s, 0 GPUh. Both produced SHA-256 `3717ed98ff6b2dfa8347c1fe75438275fe32ca4285e6891f1d44ddbf856c797f` for the metrics. A separate check against the original FK yaw errors matched all 360 recorded-case Procrustes angles within 5.2 × 10⁻¹⁴ degrees. The four-panel endpoint here is offline direction error.
