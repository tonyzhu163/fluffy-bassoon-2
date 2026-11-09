# Repaired Motus (`pi0.5_robotwin2`) + RLinf PPO/GRPO Plan (2026-11-09)

## 1. Objective & Scope

Establish a verified, statistically significant (`>= +10 pp`, two-sided McNemar $p < 0.05$) online RL improvement over the joint 50-task Motus SFT checkpoint (`motus-robotics/pi0.5_robotwin2@effc6e1a`, native $H=32$ delta-joint transforms) on RoboTwin 2.0 using RLinf (`c605b9d55ed93a180119b099e6f836abff293b08`), starting on `place_phone_stand` and then expanding across the 6 VLRLA source tasks (`handover_mic`, `pick_diverse_bottles`, `place_a2b_left`, `place_object_stand`, `place_phone_stand`, `scan_object`).

- **Approved Compute Budget (Epoch 2):** `100.0` GPU-hours (`360,000` GPU-s) approved on 2026-11-09 (`LATEST-BUDGET.json`); prior epoch (`359,665` GPU-s) archived in `HISTORICAL-BUDGET-EPOCH1-THROUGH-20261108.json`.

---

## 2. Four Root-Cause Repairs (`--repaired-recipe`)

| Component | Legacy Setting (`motusbehavior*01`) | Failure Mechanism | Repaired Setting (`--repaired-recipe`) |
| --- | --- | --- | --- |
| **1. Physical Controller** | RLinf `gen_sparse_reward_data` (`compress_path` + 33-waypoint polyline `TOPP` + `downsample_trajectory(max_points=500)` + uniform gripper slicing) | Destroys Motus's per-step delta-joint trajectory timing, desynchronizes gripper actuation from contact, and silently freezes both arms for all 32 steps whenever 33-point `TOPP` fails (`SFT = 6/64 = 9.38%`). | `patches/robotwin-dca9ec68-waypoint-controller.patch` (`waypoint_controller=True`): matches RoboTwin/Motus `take_action` per-waypoint 2-point `TOPP`, synchronizes arm+gripper at every waypoint, and adds a 50-step linear joint interpolation fallback when `TOPP` fails on a non-stationary waypoint. |
| **2. Denoising & Horizon** | `num_steps = 5`, `horizon = 192` (`6` chunks) | Coarse ODE/SDE integration and premature 6-chunk episode truncation on multi-stage bimanual tasks. | `actor.model.num_steps = 10`, `horizon = 384` (`12` chunks of `32`). |
| **3. GRPO Tied-Group Filtering** | `algorithm.filter_rewards = False`, `clip_ratio_high = 0.20` | `57%` (`73/128`) of scene groups are `0/8` ties; with `filter_rewards = False`, tied groups have `advantage = 0` in the numerator of `token-mean` (`masked_mean`) loss but `loss_mask = 1` in the denominator, diluting effective gradients by $2.33\times$. | `algorithm.filter_rewards = True`, `rewards_lower_bound = 0.1`, `rewards_upper_bound = 0.9`, `clip_ratio_high = 0.28` (`0/8` and `8/8` groups get `loss_mask = 0`, verified in `motus_learner_reference.py`). |
| **4. PPO Critic Architecture & Warmup** | `value_after_vlm = False` (suffix critic), `critic_warmup_steps = 0` | Rollout averages suffix values across `K` noisy diffusion steps (`forward_value_from_vlm`), whereas training replay evaluates only step `0` (`diff ~ 0.03–0.15`, `explained_variance ~ 0`). | `actor.model.openpi.value_after_vlm = True` (deterministic VLM-prefix critic, rollout-vs-replay `diff = 0.0`, enforced via `require_value_parity = True`) + `actor.optim.critic_warmup_steps = 15` (1 full collection of critic warmup before actor updates). |

---

## 3. Staged Execution Plan (Within 100 GPU-Hours)

### Stage 1: `place_phone_stand` Paired SFT Controller Verification (~0.5–1.5 GPU-h)
- **Script:** `outputs/robotwin-restoration-20261106-r1/verify_motus_waypoint_controller.py`
- **Protocol:** Evaluate frozen `motus-robotics/pi0.5_robotwin2@effc6e1a` on the 16 precommitted source DEV scenes (`MOTUS-DEV-CASES.json`) under:
  1. `waypoint_k5_h192`: `waypoint_controller=True`, `num_steps=5`, `horizon=192`
  2. `waypoint_k10_h384`: `waypoint_controller=True`, `num_steps=10`, `horizon=384`
  - Compare directly against the recorded legacy `gen_sparse_reward_data` baseline on the identical 16 DEV scenes (`2/16 = 12.5%`).
- **Verification Gate:** Confirm base SFT competence recovery (`>= 25%` on DEV16) and zero silent full-chunk arm freezes before launching multi-iteration RL.

### Stage 2: `place_phone_stand` Repaired GRPO & PPO Training (`C0 -> C6..C10`, ~35–40 GPU-h)
- **Protocol:**
  - **Repaired Motus GRPO (`group_size=8`, `filter_rewards=True` `[0.1, 0.9]`, `B32/accum8`):** Run fresh online collections and updates (`15` Adam steps/rank per iteration) using the repaired waypoint controller (`K=10`, `H=384`), evaluating checkpoints on the 64-scene DEV bank (`MOTUS-SOURCE-READOUT-CASES.json`).
  - **Repaired Motus PPO (`group_size=1`, `value_after_vlm=True`, `critic_warmup_steps=15`, `B32/accum8`):** Run matched fresh online collections and updates, verifying `rollout_vs_replay_value_max_abs_diff <= 1e-4` and positive critic `explained_variance`.
  - **Confirmation Gate:** Evaluate the better-performing method on the disjoint 64-scene fresh Confirmation bank (`>= +10 pp`, two-sided McNemar $p < 0.05$).

### Stage 3: 6-Source-Task SFT Baseline & Per-Task RL Expansion (~50–55 GPU-h)
- Once `place_phone_stand` passes the confirmation gate:
  1. Run a 6-source-task Motus SFT baseline panel (`handover_mic`, `pick_diverse_bottles`, `place_a2b_left`, `place_object_stand`, `place_phone_stand`, `scan_object`) under the repaired `take_action` waypoint controller (~5 GPU-h).
  2. Scale the verified winning RL algorithm (GRPO or PPO) per-task across the remaining source tasks within the remaining ~50 GPU-hours.
