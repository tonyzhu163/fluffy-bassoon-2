# UR5e π₀.₅ paired query runtime

Read-only SSH audit of retained Baige files, 2026-11-23. All 96 published cases join to an original passive tape, a full-precision joint-FK yaw and an upright seed-41 reader record. For the yaw comparison, run both methods fresh on each query. Compute the native SymMeta tuple once per tape and pass the **same unrounded scale and gripper sign** to both methods. Change only the yaw. This measures the effect of replacing the yaw estimator. An independently estimated FK gain/grip would answer a different, full-adapter question.

## Inputs and checks

Paths below use these exact shared-storage roots.

```sh
R=/mowenjie/fluffy-bassoon-2
O=$R/outputs
P=xemb-icml-campaign-20261121-r1
A=$O/symmeta-camera-command-20261123-r1
T=$O/libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live
X=$R/setups/$P
C=$X/source-ff2ddf0a
```

| Input | Path and readback |
| --- | --- |
| Original passive support | `$T/shard-{0..7}/support-meta.json`, each with 12 ordered records. Follow the absolute `passive_npz` and retain the manifest's `physics`. Original collector pin `5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b`. |
| Trusted yaw | `$A/world01/world.json`, SHA-256 `f6b5dcbcb9f79cd57210b23239c898a46926a0f8783e60cd1db6177728fef3b6`, status complete. Use `rows[].yaw_hat_deg` in degrees. The prior run accumulated the tape's six measured joint increments from a qualified UR5e reset, evaluated world grip-site FK and aligned the three world displacements to issued XY. All eight nominal active siblings reproduced archived joint and TCP increments at sensor precision, although the whole-scene support hashes differed across hosts. |
| Published reference | `$O/xemb-live-wave-20261121-r1/pi05/rows.json`, `arms.ours_equiv_v2_passive.{yaw_hat,scale_hat,grip_hat,success,controls}`. Historical result 76/96 overall and 52/64 OOD. Yaw and scale were rounded **after** execution to 2 and 3 decimals. Gripper sign is exact. |
| Frozen reader | `$O/$P/gpu03v2/stage1/checkpoints/equiv_v2_dis_dz-passive_attempt1-seed41.msgpack`, verified SHA-256 `c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4`. |
| Independent yaw readback | `$A/score02/scores.json`, select `roll_deg == 0` and `yaw_deg.model_native`. This is the CPU readback. Historical CPU checks in `$A/reference01/source-{s2ab-v2,ff2ddf0a}.json` agree with it exactly. The largest CPU vs rounded published GPU yaw difference was 0.051°. The score rows' `true_scale` and `true_grip` are labels. |

**Recover the full learned tuple** with the pinned Stage 2A functions below. This reproduces its single-support input shape and DIS preprocessing. Save both adapter tuples before rollouts and record their differences from the published rows. A rerun estimates the original full-precision values on its execution backend. The rounded rows cannot establish bitwise identity to the original floats.

```python
import numpy as np
import eval_stage2a_live_ur5e as s

params, cfg, _ = s.load_v2_checkpoint(checkpoint_path)  # pathlib.Path, once per worker
with np.load(passive_npz, allow_pickle=False) as z:
    b = {k: z[k][None] for k in z.files}
b["chunk_steps"] = np.where(b["valid"], s.CHUNK_SIZE, 0).astype(np.int32)
b["dis_flow"] = s.compute_dis_flow_maps(b["img_before"], b["img_after"], b["valid"])
y, g, h = s.predict_embodiment_v2(params, cfg, b)
symmeta = (float(y[0]), float(g[0]), int(h[0]))
fk = (world_row["yaw_hat_deg"], *symmeta[1:])
```

**Case key.** I read and hashed all 96 passive NPZs on Baige. Each digest equals both `world.rows[].archive_sha256` and the upright score's `support_sha256`. Index, suite, task and embodiment agree with the published panel. Index is `12 * shard + slot`. Shards 0–7 are, in order, `libero_10/{5,8}`, `libero_goal/{3,6}`, `libero_object/{1,8}`, `libero_spatial/{0,8}`. Slots 0–3 familiar, 4–7 compositional, 8–11 continuous. Use the pinned `eval_test_official.build_test_plan()` for names/order. Example index 0 is `libero_10/5/fam_nom`, FK +0.8057568941°, CPU SymMeta −14.5111751556°, published (−14.51°, 1.069, +1). All supports use layout 11, three valid five-control chunks. The tape stores issued chunk means and joint deltas. Querying the saved fits avoids approximate primitive-action replay.

## Pinned runtime and reset contract

Use the recovery checkout `$C` at `ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c`. Seven published shards used `$X/source-s2ab-v2` at `437da7f677632e9495c813aaa52071e207ba834e`. I verified both HEADs and identical reader implementation (`equiv_v2.py`) and motor (`ur5e_motor.py`) bytes across them. Use a two-arm driver importing the pinned modules. The stock `$C/outputs/$P/run_stage2a_8gpu.sh` runs twelve methods, fits unrelated calibrators, and its merge requires all twelve.

```sh
S=$R/setups/libero-meta-learning-20261117-r1
D=$C/outputs
export CROSSARM_RELEASE=$R/sources/libero-pro2-e5500c69
export LIBERO_PRO_ROOT=$S/simulator-source LIBERO_CONFIG_PATH=$S/libero-config
export PYTHONPATH="$X/cv2-headless-overlay:$C/src:$CROSSARM_RELEASE:$LIBERO_PRO_ROOT:$D/libero-crossarm-20261120-r1:$D/libero-xemb-ur5e-sysid-20261120-r1:$D/libero-xemb-ur5e-scale-20261120-r1:$D/$P"
export OPENPI_DATA_HOME=/mowenjie/fluffy-bassoon/cache/openpi-215abfb
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Motor Python is `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`, model config `pi05_libero`, frozen files `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1/{assets,params,READY.json}`. The simulator subprocess uses `$S/env/bin/python` and inherits the path. CPU-only module discovery verified the pinned evaluator/motor, crossarm `arm_env` and `sim_reset`, LIBERO release and simulator source, and headless OpenCV origins. The motor resolves OpenPI under `/mowenjie/fluffy-bassoon/third_party/openpi-215abfb/src`. Use the stock launcher's EGL/GLVND block verbatim, including its filtered driver library path and `desktop-libs02/nvidia-vendor.json`. Use CUDA JAX, preallocation off, NVIDIA EGL, job-local `NCCL_P2P_DISABLE=1`, and one visible inference GPU plus its physical EGL ordinal per worker. Preserve the exclusive scheduler container.

For suite rank 0–3 in the order above, support seed is `31800000 + 10000*rank + 100*task_id + 11`; query is `41800000 + 10000*rank + 100*task_id + 13`. Use `UR5e`, `BASE`, released prompt and layout 13 for both methods. Pinned `ur5e_motor.run_episode` uses NumPy default_rng(query seed + 817117), standard normals cast to float32 of shape `(ceil(horizon/5), model.action_horizon, model.action_dim)`, indexed by five-control chunk. Use the full native horizon and record native success, rewards and executed controls.

Call `s.run_paired_query(s.run_episode, ...)` for each method with one shared reset map per shard, key `(suite, task_id, 13)`, original support physics and the accepted prompt. Save the reset receipt. Before inference the entire reset dictionary must match, including physics distinct from support, scene/perturbation hashes, public observations, motor/reader frame zero, prompt, renderer, action schema, step index, model identity, horizon and noise-bank digest. Fresh within-job equality is the pairing check. The historical success ledger is a separate reproducibility reference.

**Goal 6** is shard 3, indices 36–47, query seed 41810613. Original `$O/$P/gpu03s2a` rejected its reset before a complete row. Use the whole accepted `gpu04s2agoal6/live/shard-3/live-shard.json` for that reference, retaining the original `gpu09test` tapes. Follow recovery behavior on all shards: one unscored zero-action create before the first case, then at most three identical creates per arm on a pre-inference mismatch. Keep rejected-field diffs. A persistent mismatch fails the shard. Never retry an executed trajectory or replace its layout. Use fresh attempt output and merge only complete unique case pairs (96 pairs, 192 native outcomes).

## Fastest documented shard layout within 8 GPUh

Use one Baige Distributed Training node with **8 RTX PRO 6000D, one task shard per GPU**, 12 cases × 2 consecutive arms per worker; load π₀.₅ once per worker. Both historical probes (`gpu03s2a/gpu.json`, `gpu04s2agoal6/gpu.json`) identify NVIDIA RPBZZZ6, 97,887 MiB each. The original eight-card, twelve-method attempt ran 2,200 s (36m40s) and finished seven shards; the single-card twelve-method Goal 6 recovery took 1,166 s (19m26s). These are historical launcher times, not a two-arm timing measurement. Use a **45-minute scheduler wall limit, automatic retries off** (6 GPUh maximum), reserving 2 GPUh for any separately diagnosed recovery on only the missing shards. Check capacity and actual device count/type at launch; count all allocations against the 8 GPUh cap. Qualify CUDA and renderer with the first real scored pair and retain its receipts. No queue state, GPU execution or new query result was checked in this audit.
