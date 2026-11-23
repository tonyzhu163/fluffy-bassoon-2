# Frozen UR5e reader, source calibration and π₀.₅ audit inputs

Verified on `baige-dev` on November 23, 2026. The published BASE result used `ours_equiv_v2_passive`, training seed 41, and 96 archived passive UR5e supports. The source audit already contains the full-precision camera projection and passive nominal gain. Reading two JSON fields supplies the classical baseline calibration with zero training-tape reads. The checks below used CPU imports, file metadata, digests and existing results. No reader, policy, simulator or evaluation ran.

## Frozen checkpoint

```text
/mowenjie/fluffy-bassoon-2/outputs/xemb-icml-campaign-20261121-r1/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack
```

The file is 1,434,585 bytes, SHA-256 `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4`. Its MessagePack header fields identify tag `equiv_v2_dis_dz`, mode `passive_attempt1`, seed 41, architecture `factored_equiv_v2`, visual mode `dis_lumaflow`, `use_dz_parallax=true`, maximum 32 chunks, CNN width 96 and MLP width 192. I read the metadata while skipping the parameter and optimizer payloads. The Stage 2A evaluator loads this exact basename for `ours_equiv_v2_passive`. Its learned visual input uses `compute_dis_flow_maps` (OpenCV DIS FAST, normalized by image height). [Evaluator](../xemb-icml-campaign-20261121-r1/eval_stage2a_live_ur5e.py) · [Flow implementation](../../src/fluffy_bassoon/xemb/baselines.py) · [Launch provenance](../xemb-live-wave-20261121-r1/evidence/pi05-provenance.json)

## Source-only calibration

Read `/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/train/offline-audit.json`. Its SHA-256 is `af0d1933bb23d4c040b0a4488279cdf500f88e69f906e9612523dd2385763456`. The tracked compact audit has the same fields and values. [Compact source audit](../libero-xemb-ur5e-scale-20261120-r1/evidence/offline-audit-compact.json)

| Quantity for the passive headline baseline | JSON field | Verified value |
| --- | --- | --- |
| HS camera projection, rows local x/y/z, columns image u/v, pixels per metre | `modes.passive_attempt1.analytic_visual_sysid_clean_proj.projection` | `[[-9.993283182421333, 68.66860390710232], [-86.12249450418228, -5.1102503185736765], [-21.378071976097814, -40.07758529904117]]` |
| Passive gain, metres per issued-command unit | `modes.passive_attempt1.nominal_gain` | `0.054141017027535904` |

Stage 2A fits the camera projection once per shard from **nominal active** Panda/Sawyer supports and shares it across both modes. The JSON field above equals `modes.active_probe.analytic_visual_sysid.projection` exactly. The similarly named `modes.passive_attempt1.analytic_visual_sysid.projection` holds a different, passive-fitted matrix. Stage 2A fits gain separately for each mode. The passive gain agrees with both the seed-41 checkpoint metadata and the Stage 1 report. [Source fit and storage](../libero-xemb-ur5e-sysid-20261120-r1/train_ur5e_readers.py) · [Stage 2A calibration and passive prediction](../xemb-icml-campaign-20261121-r1/eval_stage2a_live_ur5e.py)

For a rolled view, keep that projection fixed. The published `analytic_visual_hs_passive` uses the repository’s `image_motion`, a motion-weighted brightness-constancy least-squares translation per valid frame pair. Its native `analytic_visual_yaw` subtracts local dz times projection row 2, maps the residual through the pseudoinverse of the first two rows, and aligns with issued XY over the valid chunks. The HS baseline has its own image-motion implementation. The separate classical DIS comparator uses OpenCV DIS MEDIUM and its own source projection, saved as `dis_projection` in `/mowenjie/fluffy-bassoon-2/outputs/xemb-camera-dev-20261122-r1/gpu01/readers.json`. [Native motion and yaw](../../src/fluffy_bassoon/xemb/readers.py) · [Saved DIS calibration](../xemb-camera-dev-20261122-r1/evidence/readers.json)

If reconstruction is needed, the source manifest is `/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/archive/archive.json`, SHA-256 `37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076`. The checkpoint, both calibration records and manifest agree on this digest. Filter the manifest **before loading NPZs**, preserving archive order, to `split == "train"` and `embodiment == "nom"`. This gives 192 of 1,344 training rows, 96 per source arm. All have 15 controls per mode. For HS, use the three valid five-control chunks from each selected `active_path`, run `image_motion` on those 576 pairs, and call `fit_visual_projection` once on the concatenated motions, `delta_eef` and valid masks. For passive gain alone, the selected `passive_path` files supply `issued_mean`, `delta_eef` and `valid`. All 384 referenced paths exist. The source-audit and headline revisions have identical implementations of the relevant loading, motion and projection functions. This fallback was traced in source. The stored calibrations were read directly.

## Exact supports and headline results

| Asset on Baige | Verified path |
| --- | --- |
| Eight original support manifests | `/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live/shard-{0..7}/support-meta.json` |
| Published 96 rows, with per-method predictions and native outcomes | `/mowenjie/fluffy-bassoon-2/outputs/xemb-live-wave-20261121-r1/pi05/rows.json` |
| Strict merged summary and per-shard provenance | `stage2a-summary.json` and `provenance.json` beside `rows.json` |

Join each manifest row to the result at `index = 12 × shard + position` (zero-based) and check `embodiment`. I verified the ordered join in all eight shards, the published shard hashes, and all 192 referenced active/passive file paths. Select `passive_npz` for this run. All 96 passive masks have shape `(3,)` and three valid chunks. Shards 0–7 correspond in order to `libero_10` tasks 5 and 8, `libero_goal` 3 and 6, `libero_object` 1 and 8, then `libero_spatial` 0 and 8. Support used layout 11 and query used layout 13. [Official test plan](../libero-xemb-ur5e-scale-20261120-r1/eval_test_official.py)

For results, seven shards come from `gpu03s2a` at `437da7f677632e9495c813aaa52071e207ba834e`. Goal 6, shard 3 (indices 36–47), comes from `gpu04s2agoal6` at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c` and uses the original `gpu09test` supports. The eight accepted shard files concatenate exactly to the 96 merged rows, with 12 arms per row. Serializing them as compact JSONL reproduces the tracked rows’ SHA-256 `1863e924cebc57bbac19ed65195f85171c2d3757018b752a631491a140105185`. For an identity-view comparison, read `arms.ours_equiv_v2_passive.yaw_hat` and `arms.analytic_visual_hs_passive.yaw_hat` in degrees. The audit rounded yaw to two decimals and scale to three. Published native success was 76/96 for the learned reader and 56/96 for HS. [Per-query rows](../xemb-live-wave-20261121-r1/evidence/pi05-rows.jsonl) · [Summary](../xemb-live-wave-20261121-r1/evidence/pi05-summary.json) · [Shard paths and full hashes](../xemb-live-wave-20261121-r1/evidence/pi05-provenance.json)

## Verified Linux CPU runtime and Git base

The run-local Git object store is `/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-command-20261123-r1/git`. It contains the plan’s base `cccadb8cffe4c97e53f5eac9294ef974e3b291cb` and its registration commit `68ebdf6fe1e6a382c86af4ac4205e6ae78eddafa`. The existing detached checkout `registered` beside that store contains the latter commit and imports the reader modules from its own `src/`. The historical checkout `/mowenjie/fluffy-bassoon-2/setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a` also imports those modules at the recovery pin.

The following import check succeeded on Linux x86_64. The existing OpenPI environment supplies Python 3.11.17, NumPy 1.26.4, JAX/JAXlib 0.5.3 and Flax 0.10.2. The campaign’s headless overlay supplies OpenCV 5.1.0 with the DIS API. JAX reported only `CpuDevice(id=0)`.

```sh
root=/mowenjie/fluffy-bassoon-2
code="$root/outputs/symmeta-camera-command-20261123-r1/registered"
overlay="$root/setups/xemb-icml-campaign-20261121-r1/cv2-headless-overlay"
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH="$overlay:$code/src" \
  /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B -c \
  'import cv2, numpy, jax, jaxlib, flax; from flax import serialization; from fluffy_bassoon.xemb import baselines, equiv_v2, readers; print(jax.devices())'
```
