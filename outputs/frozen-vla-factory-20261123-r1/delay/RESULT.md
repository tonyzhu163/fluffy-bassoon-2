# Explicit delay separates gain from latency on frozen action tapes

On released LIBERO-Pro2.0 S20, a five-tap causal filter with a fitted delay and translation gain recovered gains **0.607 and 0.995** for targets 0.6 and 1.0. Forcing zero delay raised gain MAE from **0.027 to 0.593** and raised next-step motion error from **1.11 to 5.27 mm**. A source-calibrated correlation rule selected the same lag as joint fitting on every S20 trial. Keep that simpler lag selector and a calibrated response filter as the timing baseline for a frozen VLA adapter. Its effect on closed-loop task completion remains open.

## Primitive-rate native physics replay

The 16 nominal Panda command tapes came from the frozen OpenPI `pi05_libero` policy at collection pin `c45aa965db73ad0644382da578ff03a30b2083d7`: eight development tasks at reset 1 with a default and paced program per task. We replayed the same 80-control prefix under each condition and gain in the released simulator. S20 supplied both native seven-control queues. The five-control translation FIFO and scalar gain were experimental interventions in native physics. BASE with gain 1 used the original commands. Timing labels are additional end-to-end delay beyond the nominal controller response. [Tape provenance and checkpoint](../../libero-xemb-friction-dev-20261119-r1/cpu01/sources.json) · [Original nominal replay](../../libero-xemb-friction-dev-20261119-r1/RESULT.md)

For each held-out task, the nominal scalar impulse response came from the first 64 controls of the other seven tasks' two archived nominal tapes, pooling XYZ. The target adapter saw only issued XYZ and public world-frame wrist increments. At 20 Hz it used 64 controls (3.2 s), scored all integer lags 0–16 on the same 44 post-startup control intervals, and fitted a continuous nonnegative scalar gain. Future commands entered only to predict the next 16 wrist increments with the fitted parameters fixed. Each entry below averages across both programs and both gains. RMS is the root mean squared 3D displacement error per control, computed within each continuation and then averaged across tasks.

| Target | Exact joint lag | Joint gain MAE | Zero-lag gain MAE | Joint / zero-lag continuation RMS (mm) | Paired RMS reduction (mm), 95% task interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native BASE, 0 controls | 28/32 | 0.026 | 0.026 | 0.786 / 0.753 | −0.033 [−0.069, −0.005] |
| Injected BASE FIFO, 5 controls | 27/32 | 0.033 | 0.241 | 0.799 / 2.846 | 2.047 [1.464, 2.611] |
| Native S20, 14 controls | 27/32 | 0.027 | 0.593 | 1.114 / 5.272 | 4.158 [2.924, 5.467] |

The paired S20 reduction in gain MAE was **0.567 [0.487, 0.645]**. Fixing the gain at 1 and fitting lag raised S20 continuation error to 2.109 mm. Instantaneous centered correlation had 1.03 controls of S20 lag MAE. In a secondary control we calibrated its nominal peak offset on the same seven source tasks. Every fold gave one control. Selecting lag with that offset and fitting gain with the same FIR yielded **29/32, 31/32 and 27/32** exact delays for BASE, FIFO and S20. The respective continuation errors were 0.775, 0.806 and 1.114 mm. Joint fitting established no predictive advantage over this calibrated rule. It reported delay on four undelayed cases and raised BASE motion error slightly.

## Transfer at the original 15-control support budget

A second experiment used 360 qualified passive supports from eight official tasks disjoint from all eight Panda tasks. Data comprised UR5e × π₀.₅ (96), UR5e × π₀ (96), IIWA × π₀.₅ (84), and Kinova3 × π₀.₅ (84). The export excludes one unqualified reset on each seven-joint arm. Each support contained three five-control command means and three trusted world-frame FK increments. We calibrated metres per chunk-command once on all 16 Panda tapes (0.06348 m). The target solver fitted a planar rotation and scale independently for each of two candidate integer chunk lags. Both candidates used the last two observed endpoints. The one-chunk condition shifted the recorded motion **offline**; physical S20 was evaluated only in the Panda experiment. [UR5e π₀.₅ FK inputs](../../symmeta-camera-command-20261123-r1/evidence/world.json.gz) · [Native yaw/gain scoring labels](../../symmeta-camera-command-20261123-r1/evidence/roll.json.gz) · [Other panels and FK qualifications](../../symmeta-fk-crossarm-20261123-r1/RESULT.md) · [Policy pins](../../symmeta-fk-crossarm-20261123-r1/QUERY_CONFIG.json)

| Recorded-motion alignment | Joint lag exact, equal-task mean | Gain MAE, joint / fixed zero | Yaw MAE, joint / fixed zero |
| --- | ---: | ---: | ---: |
| Original | 96.3% [93.4, 98.4] | 0.096 / 0.090 | 2.86° / 2.74° |
| Synthetic +1 chunk (250 ms) | 86.8% [75.5, 95.7] | 0.202 / 0.379 | 4.44° / 11.62° |

Under the injected shift, paired reductions were **0.177 [0.106, 0.251]** in gain MAE and **7.18° [3.17°, 12.51°]** in yaw MAE. UR5e × π₀ had the lowest exact recovery (79.2%) among the four shifted panels. Only two endpoints remain after matching overlap, so lag and spatial gain can trade off on smooth commands. The synthetic shift also uses earlier recorded endpoints than the unshifted comparison and bypasses feedback and contact dynamics.

All intervals resample whole tasks 10,000 times with seed 20261123. Interventions, models, programs and arms sharing a task stay paired. These intervals describe task variation conditional on the fitted folds and available resets. The primitive-rate panel has one reset per task. A single command–motion stream identifies combined latency under an assumed plant response; separating sensor and actuator delays would require additional timestamps or measurements. The most direct next test is the calibrated correlation/FIR baseline on full-rate multi-arm supports with physical delay interventions, followed by paired closed-loop retries.

## CPU receipt and reproduction

The parent revision was `757a44cd4374833cad790852df8b87ddf412e638`. Local CPU replay PID 13167 ran four workers from 10:26:59 to 10:28:14 UTC on Nov 23, 2026, for 74.7 seconds, 0 GPU-hours. All 96 replays completed 80 steps (7,680 controls). All six conditions for each program had the same fresh reset fingerprint. None of the 16 nominal fingerprints matched the archived fingerprint byte for byte. The maximum nominal wrist discrepancy across all 1,296 reset/post-step samples was 0.0361 mm. Every S20 diagnostic recorded both seven-step delays. Model weights were never loaded in the replay. [Receipt](replay-receipt.json) · [Public wrist sequences](wrist.npz) · [All controls and intervals](metrics.json) · [Original plan](PLAN.md)

From the repository root, recompute every reported score from the committed command, FK and wrist evidence with the project's NumPy environment:

```sh
lane=outputs/frozen-vla-factory-20261123-r1/delay
uv run python "$lane/score.py" --output /tmp/delay-scores.json
```

For fresh physics replay, use Linux, Python 3.11, the pinned benchmark, and a separate CPU runtime. This session used these core packages and the CPU PyTorch wheel. The benchmark's rendering path is disabled.

```sh
work=$(mktemp -d)
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0.git "$work/release"
GIT_LFS_SKIP_SMUDGE=1 git -C "$work/release" checkout e5500c690c1498eca1b7050350654964190cdf70
uv venv --python 3.11 "$work/env"
py="$work/env/bin/python"
uv pip install --python "$py" numpy==1.26.4 mujoco==3.2.3 numba==0.60.0 scipy \
  gym==0.26.2 bddl==1.0.1 easydict==1.9 pyyaml opencv-python-headless==4.11.0.86 \
  h5py imageio termcolor tqdm future==1.0.0 httpx==0.24.1 msgpack==1.0.8 \
  hydra-core websockets fastapi matplotlib==3.9.4
uv pip install --no-deps --python "$py" robosuite==1.4.0 robomimic==0.2.0
uv pip install --python "$py" torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$py" "$lane/replay.py" "$work/release" --workers 4 --out "$work/replay"
uv run python "$lane/score.py" --input "$work/replay" --output "$work/recomputed.json"
```
