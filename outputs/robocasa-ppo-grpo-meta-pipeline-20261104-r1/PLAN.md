# Corrected Flow-SDE PPO & GRPO Source Collection + Multi-Interface Meta-Learning Plan

2026-11-04. **Prospective execution and implementation plan.**

Governing references:
- [docs/PAPER.md](../../docs/PAPER.md)
- [docs/FLORA-PPO.md](../../docs/FLORA-PPO.md)
- [docs/literature/robocasa-ppo-review.md](../../docs/literature/robocasa-ppo-review.md)
- [outputs/robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md)
- [outputs/robocasa-fixed-critic-source-20261104-r1/PLAN.md](../robocasa-fixed-critic-source-20261104-r1/PLAN.md)
- [outputs/flow-sde-kv-bridge-20261103-r1/PLAN.md](../flow-sde-kv-bridge-20261103-r1/PLAN.md)

---

## 1. Motivation & Diagnosis Summary

The initial 5-task 10M-control Flow-SDE PPO campaign (`OpenDrawer`, `CloseToasterOvenDoor`, `PickPlaceDrawerToCounter`, `CoffeeSetupMug`, `TurnOffStove`) failed its +10 pp TRAIN improvement gate across all five tasks (`+4.61`, `+8.91`, `+1.56`, `+4.84`, `-0.08` pp), and post-hoc DEV64 checkpoint sweeps ([PARTIAL-CURVE-R2.md](../robocasa-flow-sde-checkpoint-selection-20261104-r1/PARTIAL-CURVE-R2.md)) showed mid-run policy oscillation/collapse (`PickPlaceDrawerToCounter` peaked at iteration 40 at `56.25%` and collapsed to `31.25%` by iteration 90).

Auditing our deployed configuration against RLinf and StructRL identified four concrete bottlenecks:
1. **Critic feature port & rollout/replay value mismatch (`value_after_vlm: false`):**
   - With `value_after_vlm: false`, the critic attaches to the 300M action-expert suffix (`width=1024`), which lacks direct visual tokens and depends on the noisy action $x_t$ and denoising step $k$. Worse, unpatched RLinf averaged values across all 10 denoising steps during rollout (`prev_values`) while evaluating only a single sampled step (`denoise_inds[:, 0]`) during PPO replay.
   - Setting **`value_after_vlm: true`** (while keeping `train_expert_only: true` and `detach_critic_input: true`) attaches the fp32 `ValueHead` (`2048 -> 1024 -> 512 -> 256 -> 1`) to the frozen VLM's vision+language+proprioception `prefix_out` (`width=2048`). Because `prefix_out` is computed once before the denoising loop, rollout and replay critic values are automatically identical and state-conditioned without unfreezing the VLM.
2. **Final denoising step variance spike (`ignore_last: false`):**
   - In RLinf's `flow_sde` sampler, noise scale $\sigma_k \propto \sqrt{t/(1-t)}$ drops from `0.30` at $k=0$ to `0.0316` at the final step $k=9$, creating a $90\times$ inverse-variance ($1/\sigma_k^2$) log-probability spike when $k=9$ is sampled. Setting **`ignore_last: true`** restricts stochastic exploration to steps $\{0,\dots,8\}$.
3. **Aggressive off-policy epoch reuse (`update_epoch: 4`):**
   - Reusing ~1,000 unmasked chunks for 4 PPO epochs (24 AdamW steps per iteration at `lr=5e-6`) drove `approx_kl` to `0.08–0.13` and `clip_fraction` to `~0.25` every iteration. StructRL uses **`update_epoch: 1`** (1 pass per rollout batch).
4. **Lack of within-scene outcome contrast in PPO vs. GRPO:**
   - Standard PPO (`group_size: 1`) runs 1 rollout per reset seed across 128 distinct seeds per iteration, relying entirely on $V(s)$ for credit assignment.
   - **Critic-free GRPO (`adv_type: grpo`, `loss_type: actor`, `group_size: 8`, `add_value_head: false`)** runs $G=8$ parallel rollouts from the **exact same reset seed** of the single target task (`16 seeds × 8 rollouts = 128 episodes/iteration`), computing empirical within-scene relative advantages:
     $$\hat{A}_{g,i} = \frac{R_{g,i} - \mu_g}{\sigma_g + \epsilon}$$
   - Crucially, GRPO's 8-way same-scene groups also generate **true same-scene failure-to-success trajectories** that match our downstream 5-attempt same-scene deployment protocol.

---

## 2. Code & Pipeline Architecture

### 2.1 Completed Changes on `main`
- **Dual PPO/GRPO contract & grouped seeding ([`rlinf_robocasa_flow_ppo.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo.py)):**
  - Appended `TurnOnSinkFaucet=8` to `FLOW_TASK_NUMBERS`.
  - Extended `flow_contract(cfg)` to validate both GAE PPO (`adv_type='gae'`, `loss_type='actor_critic'`, `group_size=1`, `add_value_head=True`) and GRPO (`adv_type='grpo'`, `loss_type='actor'`, `group_size>=2` dividing `train_envs`, `add_value_head=False`), plus `value_after_vlm`, `ignore_last`, `noise_level`, and `update_epochs`.
  - Extended `train_case_seeds` so `group_size=G` assigns identical reset seeds to each contiguous group of $G$ local environment slots per worker.
  - Added `flow_episode_ids` (`task/seed` for PPO and eval; `task/seed/g00..g07` for GRPO training) and `verify_grpo_group_reset` (enforcing identical MuJoCo XML/state fingerprint, prompt, and initial public state across all $G$ members of every GRPO group at reset).
- **Runtime & Driver ([`rlinf_robocasa_flow_ppo_runtime.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo_runtime.py), [`scripts/[train]robocasa_flow_ppo.py`](../../scripts/[train]robocasa_flow_ppo.py)):**
  - Wired `flow_episode_ids`, `verify_grpo_group_reset`, and critic-optional `flow_logprob_parity` / `flow_value_parity` checks.
- **Resolved Configs (`configs/native_ppo/`):**
  - [`robocasa365_pi05_flow_sde_ppo_vlm_critic.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_ppo_vlm_critic.yaml)
  - [`robocasa365_pi05_flow_sde_grpo.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_grpo.yaml)
- **Generalized Corpus Export & Multi-Interface Meta-Learners ([`flow_history.py`](../../src/fluffy_bassoon/meta_rl/flow_history.py)):**
  - Generalized `export_corpus` to support arbitrary `(num_streams, num_versions * chunks_per_episode, 25)` schedules (including 130-iteration $H=600$ PPO and GRPO corpora) across all 4 meta-learning objectives (`ad`, `dpt`, `dit`, `ic_iql`).
  - Added `FlowBridgeConfig` and `HistoryFlowSteeringPolicy` supporting both **`kv_bridge`** (causal history decoder + `ReactiveKVResidual` through the frozen $\pi_{0.5}$ suffix) and **`velocity_bridge`** (causal history decoder + zero-init velocity residual head) trained with advantage-weighted flow-matching loss (`flow_matching_loss`) for **AD**, **DPT**, **DIT**, and **IC-IQL**.

### 2.2 Remaining Pipeline Enhancements to Implement Before Stage 1 Launch
1. **In-Run Periodic DEV64 Evaluation (`rlinf_robocasa_flow_ppo.py`, `rlinf_robocasa_flow_ppo_runtime.py`):**
   - Update `flow_contract(cfg)` to read `runner.val_check_interval` and include periodic mid-run 64-case evaluation panels (`iterations // val_check_interval - 1` intermediate panels in addition to `before` and `after`) in `nominal_policy_controls` and `reset_reserve_controls` when `0 < val_check_interval < iterations`.
   - Update `FlowRunner._log_step_metrics` to execute paired DEV64 evaluation whenever `0 < val_check_interval < max_steps` and `self.global_step % val_check_interval == 0` (for `0 < global_step < max_steps`), isolating and restoring rollout/env RNG around `self.evaluate()` and writing `evaluations/step-{self.global_step:03d}.json` plus updating `evaluations/curve.json`.
   - Preserve exact backwards compatibility when `val_check_interval == max_epochs` or `val_check_interval <= 0` (the legacy `before`/`after`-only schedule).
2. **GRPO Within-Scene Multi-Attempt Stream Builder (`flow_history.py`):**
   - Add `export_grpo_group_trials(...)` to [`flow_history.py`](../../src/fluffy_bassoon/meta_rl/flow_history.py) so each 8-rollout group (`g00..g07` sharing one physical scene `seed`) can also be exported as a **within-scene multi-attempt trial** (ordering failed attempts before successful attempts on the identical scene, capped at 5 attempts to match deployment memory length), alongside the standard cross-update stream export (`export_corpus`).
3. **Closed-Loop Native Evaluation Wrapper for Learned Bridges (`flow_history.py`):**
   - Add `FlowBridgeCodec` and `FlowBridgeMotor` in [`flow_history.py`](../../src/fluffy_bassoon/meta_rl/flow_history.py) so `HistoryFlowSteeringPolicy` (`kv_bridge` and `velocity_bridge`) plugs directly into [`collect_case_attempts`](../../src/fluffy_bassoon/meta_rl/steering_history_eval.py) and [`NativeAttemptBackend`](../../src/fluffy_bassoon/meta_rl/steering_history_native.py) for 5-attempt RoboCasa evaluation with zero-init native parity.
4. **Core Unit Tests (`tests/test_robocasa_flow_ppo.py`, `tests/test_flow_history.py`):**
   - Test periodic evaluation contract arithmetic in `test_robocasa_flow_ppo.py`.
   - Test `export_grpo_group_trials` and `FlowBridgeMotor` / `FlowBridgeCodec` in `test_flow_history.py`.

---

## 3. Two-Stage Execution & Launch Plan

### Stage 0 — Allocated Linux CPU Qualification (< 0.5 CPU-h, 0 GPU-h)
- Sync verified commit on `main` to Baige shared NVMe (`/mowenjie/fluffy-bassoon-2`).
- Run `pytest tests/test_robocasa_flow_ppo.py tests/test_flow_history.py tests/test_pi05_residual.py` inside the qualified Python 3.11 environment on an allocated Linux CPU container.
- Verify ` patches/rlinf-e1a82ebf-critic-chosen-step-value.patch` is applied in the RLinf checkout.

### Stage 1 — 1-Task Pilot Gate on `OpenDrawer` (40 Iterations = ~3.1M Controls/Run)
Before committing 5×2 full 130-iteration runs, run a controlled 40-iteration (~3.1M TRAIN controls, periodic DEV64 evaluation every 10 iterations at steps `0, 10, 20, 30, 40`) comparison on `OpenDrawer` (8 GPUs per run):
1. **Arm A — Fixed VLM-Critic PPO (`robocasa365_pi05_flow_sde_ppo_vlm_critic.yaml`):**
   - `adv_type: gae`, `value_after_vlm: true`, `train_expert_only: true`, `ignore_last: true`, `update_epoch: 1` (and compare against `update_epoch: 4` if needed), `val_check_interval: 10`, `max_epochs: 40`.
2. **Arm B — Critic-Free GRPO (`robocasa365_pi05_flow_sde_grpo.yaml`):**
   - `adv_type: grpo`, `loss_type: actor`, `group_size: 8`, `add_value_head: false`, `train_expert_only: true`, `ignore_last: true`, `update_epoch: 1`, `val_check_interval: 10`, `max_epochs: 40`.
3. **Stage 1 Gate Criteria:**
   - Pre-SGD `logprob-parity` (and `value-parity` for Arm A) passes ($\le 10^{-3}$) at iteration 1.
   - `frozen_vlm_unchanged == True` and all losses finite across all 40 iterations.
   - `approx_kl` remains bounded ($\le 0.03$) without clip-fraction saturation.
   - Clear upward learning trajectory on TRAIN and paired DEV64 panels ($\ge +10\text{ pp}$ gain over step 0 without mid-run collapse).

### Stage 2 — Full 5-Task Source Campaign (130 Iterations = ~10M Controls/Task)
On the revised 5-task source portfolio:
1. `OpenDrawer` (`train_seed=2100000`, `eval_seed=610100`)
2. `PickPlaceCounterToCabinet` (`train_seed=2200000`, `eval_seed=620100`)
3. `CoffeeSetupMug` (`train_seed=2500000`, `eval_seed=650100`)
4. `CloseToasterOvenDoor` (`train_seed=2700000`, `eval_seed=670100`)
5. `TurnOnSinkFaucet` (`train_seed=2800000`, `eval_seed=680100`)

- Run the qualified Stage 1 recipe(s) (Fixed VLM-Critic PPO and/or GRPO) to 130 iterations per task with periodic DEV64 evaluation every 10 iterations (`0, 10, 20, ..., 130`).
- Lock the best DEV64 checkpoint per task before running a fresh paired 128-case TEST evaluation against policy 0.

### Stage 3 — Downstream Multi-Interface & Meta-Learner Training Matrix
On both admitted corpora (**PPO Corpus** and **GRPO Corpus**, including GRPO's within-scene group trials):
- **Steering Interfaces (3):**
  1. `kv_bridge` (`HistoryFlowSteeringPolicy` + `ReactiveKVResidual`, flow-matching loss through frozen $\pi_{0.5}$)
  2. `velocity_bridge` (`HistoryFlowSteeringPolicy` + zero-init velocity residual, flow-matching loss through frozen $\pi_{0.5}$)
  3. `flow_physical_action` (`HistoryNoisePolicy` + `DirectActionCodec`, direct 600-D action chunk baseline)
- **Meta-Learning Objectives (4):**
  1. **AD** (behavior action targets, uniform weight $1.0$)
  2. **DPT** (DEV-selected teacher action targets queried on saved observations, historical context unchanged, uniform weight $1.0$)
  3. **DIT** (behavior action targets weighted by $\exp(\text{clamp}((Q_{\text{DIT}} - V_{\text{DIT}})/\beta))$)
  4. **In-Context IQL (`ic_iql`)** (behavior action targets weighted by $\exp(\text{clamp}((Q_{\text{IQL}} - V_{\text{IQL}})/\beta))$)
- **Evaluation:** 5-attempt closed-loop RoboCasa evaluation on held-out ID and OOD tasks via `NativeAttemptBackend`, comparing full history (`mode="full"`), per-episode memory reset (`mode="episode"`), static (`mode="static"`), and reward-blind (`reward_blind=True`) controls.
