# Phase 3 Scaled 8-GPU Run: Appearance-Invariant Cross-Morphology SysID (`libero-xemb-ur5e-scale-20261120-r1`)

Registered plan, November 20, 2026. Scales Phase 3 (`libero-xemb-ur5e-sysid-20261120-r1`) to **8× Pro6K GPUs** across **all 24 LIBERO-Pro2.0 training tasks** (`SPLITS.json["train"]`) × 4 resets × 2 physical 7-DoF source arms (`Panda`, `Sawyer`) = **`1,344` training trajectories** (`6×` data scale, `3×` task diversity) with an **appearance-invariant `luminance_flow` visual motion encoder** (strictly invariant to RGB channel permutations / arm colors, subsuming brightness-constancy optical flow) and **8-GPU sharded live closed-loop evaluation** across all **8 dev tasks × all 12 evaluation embodiments = `96` paired held-out 6-DoF `UR5e` queries (`1,056` live episodes)**. Official 8-task test split (`SPLITS.json["test"]`) remains strictly untouched.

## Key Hypothesis Tested vs. Phase 3 Baseline (`libero-xemb-ur5e-sysid-20261120-r1`)

- In the 224-tape raw-RGB baseline (`gpu02fit`), `factored_equiv` reached `18.4°` (`52.4%` all-correct) on `active_probe` of held-out 6-DoF `UR5e` (vs `65.0°` `joint_transformer`, `57.9°` `factored_no_equiv`, `19.0°` `analytic_visual_sysid`), **but** degraded to `52.9°` on zero-probe `passive_attempt1` while grayscale `analytic_visual_sysid` reached `11.9°` (`70.8%`).
- **Falsifiable Hypothesis:** Combining `visual_mode="luminance_flow"` (7 color-invariant channels: temporal luminance diff `dt`, color-agnostic motion magnitude, spatial luminance gradients `gx, gy`, `luma_before`, and CoordConv `xx, yy`) with `6×` multi-task cross-arm training data (`1,344` 7-DoF `Panda`+`Sawyer` trajectories, `cnn_width=96`, `mlp_width=192`, `3,000` updates, 3 seeds) will drop `factored_equiv` yaw error on held-out 6-DoF `UR5e` below **`10°`** on `active_probe` AND below **`15°`** on zero-probe `passive_attempt1` (closing or beating `analytic_visual_sysid` while crushing monolithic `joint_transformer`, `factored_no_equiv`, `frame0_probe`, `analytic_proprio_sysid`, and `analytic_joint_jacobian`).

## Compute Budget (8× NVIDIA RPBZZZ6, `NCCL_P2P_DISABLE=1`)

- **Job S1 (8× Pro6K, ~20 min wall-clock = ~2.7 GPUh):** 8-shard parallel collection (`1,440` trials: `1,344` train + `96` `UR5e` dev) + 3-seed training (`luminance_flow` + `rgb` ablation) & full Direction 3 offline audit.
- **Job S2 (8× Pro6K, ~22 min wall-clock = ~3.0 GPUh):** 8-shard parallel live closed-loop frozen-$\pi_{0.5}$ evaluation (1 dev task per GPU, all `96` paired `UR5e` queries × `11` controllers = `1,056` live episodes).
- **Total Phase 3 Scaled Budget:** `~5.7` GPU-hours (cumulative with Phase 2 + Phase 3 baseline well within the 50 GPU-hour autonomous budget).
