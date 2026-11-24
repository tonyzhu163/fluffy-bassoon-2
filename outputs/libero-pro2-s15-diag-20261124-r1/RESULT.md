# Track 2 CPU diagnostic: LIBERO-Pro2.0 S15 and S16 camera geometry

Zero-GPU CPU diagnostic on `baige-dev` (`taskset -c 52,53`, `MUJOCO_GL=osmesa`) across all 8 official held-out LIBERO-Pro2.0 tasks on Panda (layout 13, seed 42). [Plan](PLAN.md) · [Raw measurements](s15_s16_diag.json)

## Findings

| Task | S15 `agentview` shift (in frame) | S15 `wrist` shift (in frame) | S16 `agentview` shift (in frame) | S16 `wrist` shift (in frame) |
| --- | ---: | ---: | ---: | ---: |
| `libero_spatial/0` | 29.9 px (Yes) | 1043.3 px (**No**) | 8.2 px (Yes) | 12.7 px (Yes) |
| `libero_spatial/8` | 10.9 px (Yes) | 156.7 px (**No**) | 7.7 px (Yes) | 12.7 px (Yes) |
| `libero_object/1` | 26.6 px (Yes) | 1043.3 px (**No**) | 18.7 px (Yes) | 30.6 px (Yes) |
| `libero_object/8` | 26.6 px (Yes) | 1043.3 px (**No**) | 19.6 px (Yes) | 30.6 px (Yes) |
| `libero_goal/3` | 29.6 px (Yes) | 1043.3 px (**No**) | 19.8 px (Yes) | 30.6 px (Yes) |
| `libero_goal/6` | 29.0 px (Yes) | 1043.3 px (**No**) | 18.6 px (Yes) | 30.6 px (Yes) |
| `libero_10/5` | 11.7 px (Yes) | 156.7 px (**No**) | 6.5 px (Yes) | 12.7 px (Yes) |
| `libero_10/8` | 11.0 px (Yes) | 156.7 px (**No**) | 8.1 px (Yes) | 12.7 px (Yes) |

## Implications for next GPU experiment
1. **`S15` completely removes the gripper from the wrist camera (`in_frame=False` on 8/8 tasks):** The `[0.1, -0.1, 0.07]` m offset shifts the wrist camera 15.8 cm off the gripper, moving the grip site 157–1,043 px outside the 256×256 image while `agentview` stays in frame (11–30 px shift). 2D image warping on the wrist camera cannot recover out-of-frame pixels.
2. **`S16` keeps both cameras in frame on 8/8 tasks** (`fov_scale=1.35`, `image_shift_xy`, `crop_ratio=0.04`, `distortion_strength=0.04`), making it a direct target for **test-time self-supervised visual-proprioceptive camera rectification** (fitting the inverse warp from a short robot motion trace + known `robot0_eef_pos`, without retraining $\pi_{0.5}$ or reading privileged camera parameters).
3. **Next decisive GPU experiment:**
   - **Factorial `S15` causal ablation** (`BASE` vs `S15_full` vs `S15_ext_only` vs `S15_wrist_only`) to prove whether the `0/48` collapse is driven by the shifted wrist view, `agentview`, or both, and test whether detecting/rectifying or masking a corrupted wrist stream rescues `S15`.
   - **`S16` test-time self-supervised rectification** (`S16_raw` vs `S16_oracle_rectified` vs `S16_selfcal_rectified`) on the 8 held-out tasks with frozen $\pi_{0.5}$.

## Closed-loop native GPU evaluation (`job-so8o7jd64vh8`, 1× RTX 4090, 0.201 GPUh, exit 0)

Evaluated frozen $\pi_{0.5}$ on Panda across 8 paired starts (4 held-out suites × layouts `13, 14`, 64 native episodes). [Live summary](live-summary.json)

| Condition | Native successes | Mean controls |
| --- | ---: | ---: |
| `BASE` | 7/8 | 184.2 |
| `S15_full` | 0/8 | 330.0 |
| **`S15_ext_only`** (external shifted, wrist nominal) | **8/8** | **149.6** |
| **`S15_wrist_only`** (wrist shifted, external nominal) | **0/8** | 330.0 |
| `S15_wrist_zero` (shifted wrist blanked to gray) | 0/8 | 330.0 |
| `S16_raw` | 3/8 | 265.6 |
| **`S16_oracle_rect`** (exact 2D inverse warp) | **8/8** | **149.8** |
| `S16_selfcal_rect` (12-step centroid warp) | 2/8 | 311.8 |

**Conclusion:** `S15`'s `0%` collapse is **100% caused by the 15.8 cm wrist-camera offset** (`S15_ext_only = 8/8` vs. `S15_wrist_only = 0/8`), and blanking the corrupted wrist view does not rescue it (`0/8`). `S16` is 100% recoverable by exact 2D image rectification (`3/8 → 8/8`), confirming it is a pure 2D camera-geometry inversion problem.
