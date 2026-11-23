# Does a joint-to-pose adapter transfer to an arm absent from calibration?

**Removing the target body from calibration made the adapter worse than dropping state in both transfer directions.** Using 144 frames of IIWA support gave 0.2346 action RMSE on UR5e. Using 144 UR5e frames gave 0.3193 on IIWA. Across tasks, fitting only on the opposite arm increased RMSE over dropping state by 0.1768 (paired 95% interval 0.1203–0.2384). Median Cartesian position error was 68.9 cm across the 16 queries.

| Calibration or control | UR5e target | IIWA target | Both targets, mean RMSE (95% task interval) |
| --- | ---: | ---: | ---: |
| Other arm only, 144 frames | 0.2346 | 0.3193 | 0.2770 (0.2222–0.3410) |
| Target only, 15 frames | 0.0164 | 0.0118 | 0.0141 (0.0081–0.0201) |
| Paired calibration, 288 frames | 0.0127 | 0.0159 | 0.0143 (0.0063–0.0233) |
| Raw joint slots | 0.0995 | 0.2346 | 0.1670 (0.1393–0.1989) |
| Drop state | 0.1042 | 0.0960 | 0.1001 (0.0635–0.1376) |
| Known TCP oracle | 0 | 0 | 0 |

Fifteen target frames reduced median position error to 2.76 cm. They improved action RMSE over source-only calibration on 16/16 queries (paired mean reduction 0.2629, 95% interval 0.2097–0.3262) and over dropping state on 15/16 (0.0860, 0.0509–0.1224). The paired difference between pooled calibration and 15 target frames was 0.0002 (−0.0056 to 0.0061). These observations support calibrating this adapter on the target arm. An equivalence claim or a minimum frame count would require additional evidence.

RMSE compares the first five actions, clipped per component to [−1, 1], with the known-TCP output at identical images, prompt and noise. Average the two target arms within each of the eight held-out tasks and bootstrap eight tasks with replacement (10,000 draws, seed 915). Intervals describe variation across these tasks at the sampled frames and noise seeds. Over the entire unclipped 50×7 chunk, mean RMSE was 0.5484 for other-arm-only, 0.0186 for 15 target frames and 0.0346 for pooled calibration. [Per-query errors and paired intervals](scores.json)

All 96 policy calls ran on CPU with the model in eval mode and all 450,046,176 parameters frozen (0 GPU-hours). All 16 oracle action hashes and both errors for all four repeated controls matched the preceding run exactly. This measures frozen forward-pass consistency on recorded simulation frames. Native rewards, robot execution, source-frame registration and transfer to additional morphologies remain unmeasured.

## Calibration and scoring

The fit uses the same sine/cosine joint encoding, zero-radian seventh-joint padding for UR5e, 32-nearest local affine ridge regression (penalty 0.001) and measured finger positions as the preceding state experiment. The adapter receives only encoder angles at query time. SmolVLA receives two RGB views, the instruction, the predicted Cartesian pose and both measured finger positions. The arm identifier selects the support set for each experimental condition and never enters the regressor or policy.

| Condition | Calibration for each target |
| --- | --- |
| Other arm only | 144 synchronized joint/TCP pairs from six episodes on the opposite arm. Zero target examples. |
| Target 15 | 15 synchronized pairs from three episodes on the target arm. Zero source examples. The same local fit uses all 15 available neighbors. |
| Paired calibration | The preceding experiment's pooled 288 pairs from twelve episodes, six per arm. |
| Raw slots | First six joint angles and two measured finger positions. |
| Drop state | Checkpoint mean in all eight state channels, giving a zero normalized state token. |
| Known TCP oracle | Measured target pose with the archived rotation-vector convention and measured finger positions on the query frame. |

Full calibration uses 24 evenly spaced frame indices, including endpoints, from each of the first three episodes of tasks 0 and 1. UR5e episode IDs are 0, 1, 2, 41, 42, 43. IIWA IDs are 0, 1, 2, 46, 47, 48. In both few-shot fits, use the fourth and fifth episodes of task 0 and the fourth of task 1. Take five frames per episode at equally spaced positions including endpoints, rounding down to integer indices.

| Target | Task | Episode | Few-shot frame indices |
| --- | ---: | ---: | --- |
| UR5e | 0 | 3 | 0, 35, 71, 107, 143 |
| UR5e | 0 | 4 | 0, 28, 56, 84, 113 |
| UR5e | 1 | 44 | 0, 34, 68, 102, 137 |
| IIWA | 0 | 3 | 0, 41, 82, 123, 165 |
| IIWA | 0 | 4 | 0, 35, 71, 107, 143 |
| IIWA | 1 | 49 | 0, 40, 80, 120, 161 |

Tasks 2–9 contribute one previously selected midpoint frame per arm per task, sixteen query frames from sixteen separate episodes. All query episodes are disjoint from every calibration episode on the corresponding arm. The few-shot episodes are also disjoint from the full-calibration episodes. The complete manifests identify every calibration pair and scoring frame by arm, task, episode and frame index. [Manifests and per-query scores](scores.json)

## Reproduction

Use the pinned assets and CPU environment from the [original state experiment](../state/RESULT.md) at `89bfe0ced881a001687c69a3bcde8af147edc545`. The run strictly loads `lerobot/smolvla_libero@a24f1f380461b665158d52acd140152c79e8b36d` and applies the checkpoint's pre/postprocessors. The scores record the SHA-256 digests of the executing script and model weights. Run from this directory.

```bash
assets="$HOME/state-exp"
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=1
"$assets/venv/bin/python" transfer.py "$assets" --out scores.replay.json
```
