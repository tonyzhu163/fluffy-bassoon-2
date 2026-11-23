# Offline camera-roll scoring

Run from the parent-deployed Git checkout on Baige Linux. The command reads the frozen artifacts on shared storage and creates one JSON report at a fresh path. It uses the existing Python 3.11 environment, CPU JAX 0.5.3, Flax 0.10.2, NumPy 1.26.4 and the campaign's OpenCV DIS overlay. It needs no simulator, policy server, training or GPU allocation.

```bash
code=$(git rev-parse --show-toplevel)
data=/mowenjie/fluffy-bassoon-2
run=outputs/symmeta-camera-command-20261123-r1
py=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
overlay=$data/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 \
    PYTHONPATH="$overlay:$code/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$py" -Bu "$code/$run/score.py" --data-root "$data" --output "$data/$run/cpu01/scores.json"
```

The default panel has 288 scored views from 96 support files across eight tasks and 192 inverse-rotation diagnostics. Elapsed runtime is unmeasured. The report records elapsed seconds and library versions. The verified headless overlay supplies OpenCV 5.1.0 with DIS. [Linux CPU import check](RUNTIME.md)

## CLI and replay subsets

```text
score.py --output FRESH_JSON [--data-root SHARED_ROOT] [--descriptor PANEL_JSON]
         [--rolls 0 [90] [-90]]
```

`--data-root` defaults to `/mowenjie/fluffy-bassoon-2`. The official panel is the default. An optional descriptor supplies a subset or newly rendered passive supports, using the same checkpoint, calibration and scoring path. The angle list defaults to `0 90 -90` and must begin with zero. To score independent simulator renders exactly as provided, pass `--descriptor /path/to/replays.json --rolls 0` with the environment above. Give each replay condition its own row ID.

```json
{
  "name": "ur5e_replay_subset",
  "rows": [
    {
      "id": "libero_10/5/fam_yaw90/replay",
      "index": 1,
      "suite": "libero_10",
      "task_id": 5,
      "embodiment": "fam_yaw90",
      "camera_condition": "simulator replay",
      "passive_npz": "supports/01-fam_yaw90-passive.npz",
      "chunk_steps": [5, 5, 5]
    }
  ]
}
```

Required descriptor fields are `name`, `rows` and each row's `id`, `suite`, `task_id`, registered `embodiment` and `passive_npz`. Paths may be absolute or relative to the descriptor. Each NPZ needs a boolean `valid[K]`, byte RGB `frame0_img[H,H,3]`, `img_before[K,H,H,3]`, `img_after[K,H,H,3]` and float32 `issued_mean[K,7]`, `delta_eef[K,3]`, `delta_jaw[K,2]`. Supply the actual `chunk_steps[K]` in the NPZ or descriptor. The scorer verifies five controls for each of the first three valid slots and records their original indices. Optional row fields are `index`, `shard`, `camera_condition`, `physics` and an expected hex `sha256` of the NPZ. Published parity uses the locked official panel. Custom rows still get a direct native-model check at zero and the inverse diagnostics for any added rolls.

## Inputs and provenance

All paths below are relative to `--data-root`. The default loader checks the file hashes and the row identities. The support and reference set digests hash the ordered lowercase SHA-256 hex digests, one per line including the final newline, in shard and official embodiment order.

| Input | Location | SHA-256 |
|---|---|---|
| Frozen seed-41 passive reader | `outputs/xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack` | `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4` |
| Frozen HS calibration | `outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/train/offline-audit.json` | `af0d1933bb23d4c040b0a4488279cdf500f88e69f906e9612523dd2385763456` |
| Source training archive identified by both artifacts | `outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/archive/archive.json` | `37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076` |
| 96 official passive UR5e π₀.₅ NPZs | `outputs/libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live/shard-{0..7}/support-meta.json` points to the passive tape for each transform | Set `8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554` |
| Published native predictions, both methods | `outputs/xemb-icml-campaign-20261121-r1/{gpu03s2a,gpu04s2agoal6}/live/shard-N/live-shard.json`. Shard 3 comes from `gpu04s2agoal6`; the other seven come from `gpu03s2a`. | Set `52fa154a6becd989d0bdaa19246694289e9ac33d1fbfe1b299b870c81cf179ae` |

The collector ran at `5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b`, reader training at `d563845717e1f9312878b228328908484c4752f9`, the first seven reference shards at `437da7f677632e9495c813aaa52071e207ba834e` and the replacement shard at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`. All 96 official masks contain exactly three true slots. The collector used a 15-control budget and five-control chunks. The published native loaders assign five steps per valid slot. The archives retain chunk means and omit individual primitive commands and a per-chunk step count.

## Readouts

Both native methods use the released code. The baseline called HS in the campaign fits one image translation to brightness gradients per frame pair using motion-weighted least squares. Its native readout removes vertical parallax with the stored active-nominal Panda/Sawyer projection and projects the motion into XY. The learned reader recomputes dense DIS and its gradient, luminance and spatial features, then reads out its learned XY vector, dz correction and confidence per chunk. The command yaw, gain, gripper sign and all non-image inputs stay fixed across rolls. Positive roll is counterclockwise in the displayed image. All three before/after images and the reset image rotate on their two spatial axes with NumPy quarter turns.

The two matched predictions use the same issued XY, a unit visual direction, a unit command direction and uniform weights on the same chunks with nonzero directions from both readers. This gives each contributing chunk one vote independent of vector scale or learned confidence. The yaw is the angle of the summed dot and signed cross products. The report retains the raw vectors, native confidences, common weights and resultant strength for audit. The native estimators retain their published magnitude and confidence weighting.

The score for each view uses wrapped angular error against the original command yaw. The zero-roll check compares the two native outputs by query with the archived two-decimal predictions. Each nonzero roll is undone at the pixel level and scored again, with all features recomputed. These diagnostic scores are evaluated against the original zero-roll prediction. The checkpoint's visual head reads before/after images and local dz. Frame zero participates in every image transform and inverse pixel check.

## Reading the report

- `rows` has one paired record per trajectory and angle, including predictions, circular absolute errors, true transform, shifts from zero in [−180°, 180°), matched model-minus-HS absolute-error difference, vector diagnostics and the data/checkpoint hashes. Negative error differences favor the model. A matched resultant indistinguishable from zero in float64 gets the shared 0° fallback and `identified=false`. The summaries include fallback rows and count them.
- `supports` lists NPZ identities, original row keys, selected indices, issued commands and local displacements. `sources` gives the frozen checkpoint and calibration. Join a world-control result by official `index` and verify `suite`, `task_id` and `embodiment`.
- `parity_by_id` retains each native archived yaw and its wrapped discrepancy. `checks` reports the maximum discrepancy and how many values agree within the 0.005° half-width of the published rounding bin plus 0.0001° for float32 arithmetic. It also checks the learned readout against a separate call to the released `predict_embodiment_v2` at zero. Inverse rescoring and the direct inference check use 0.0001°. Per-query disagreement leaves a report on disk and returns a nonzero exit code. [Published per-query rows](../xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl)
- `summary` reports each angle for all 96 supports, the 64 OOD supports, each regime, and each task (overall and OOD). Mean signed shift is the arithmetic mean of the wrapped per-trajectory shifts. Mean absolute shift measures displacement without sign cancellation. Paired MAE changes and matched model-minus-HS differences are in degrees. Repeated views of a tape share the same support ID.

Sources for the official panel, native estimators and 15-control protocol are [official panel builder](../libero-xemb-ur5e-scale-20261120-r1/eval_test_official.py), [Stage 2A inference and calibration](../xemb-icml-campaign-20261121-r1/eval_stage2a_live_ur5e.py), [15-control ablation](../xemb-icml-campaign-20261121-r1/eval_stage3_scaling.py), [HS and native planar solver](../../src/fluffy_bassoon/xemb/readers.py), [DIS features](../../src/fluffy_bassoon/xemb/baselines.py) and [V2 reader](../../src/fluffy_bassoon/xemb/equiv_v2.py).
