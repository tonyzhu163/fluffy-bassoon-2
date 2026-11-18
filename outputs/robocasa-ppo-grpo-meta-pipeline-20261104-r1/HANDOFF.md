# Handoff: RoboCasa365 Fixed VLM-Critic PPO + GRPO + Multi-Interface Meta-Learning Pipeline

**Date:** 2026-11-04
**Run Directory:** `outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/`
**Plan:** [`PLAN.md`](PLAN.md)
**Linux CPU Qualification Receipt:** [`CPU-QUALIFICATION.json`](CPU-QUALIFICATION.json) (`46/46` pytest cases passed on Baige Linux CPU + Hydra contract verification for both configs)

---

## 1. What Is Implemented, Tested, and Deployed

All code changes from [`PLAN.md`](PLAN.md) are complete, verified on allocated Baige Linux CPU (`CUDA_VISIBLE_DEVICES=""`), committed to `main`, pushed to `origin/main`, and deployed to a clean detached worktree on Baige:

1. **Source Training (Single-Task Fixed VLM-Critic PPO & Single-Task Critic-Free GRPO):**
   - [`configs/native_ppo/robocasa365_pi05_flow_sde_ppo_vlm_critic.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_ppo_vlm_critic.yaml):
     - Single RoboCasa task per run (`task_filter: ['${oc.env:ROBOCASA_FLOW_TASK}']`, `group_size: 1`, 128 parallel environments).
     - `value_after_vlm: true`, `add_value_head: true`, `ignore_last: true`, configurable `noise_level` (default `0.3`, override with `ROBOCASA_FLOW_NOISE_LEVEL=0.5`), `update_epoch` (default `4`), `val_check_interval` (default `10`), `save_interval` (default `5`).
   - [`configs/native_ppo/robocasa365_pi05_flow_sde_grpo.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_grpo.yaml):
     - Single RoboCasa task per run (`task_filter: ['${oc.env:ROBOCASA_FLOW_TASK}']`, `adv_type: grpo`, `loss_type: actor`, `group_size: 8`, `add_value_head: false`).
     - 128 parallel single-task environments organized as **16 procedural scene seeds of that task $\times$ 8 parallel attempts per seed** (`g00..g07`) per iteration.
   - [`src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo.py) & [`src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo_runtime.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo_runtime.py):
     - Added `TurnOnSinkFaucet=8` to `FLOW_TASK_NUMBERS`.
     - Added `flow_episode_ids` (`task/seed/g00..g07` for GRPO) and `verify_grpo_group_reset` (enforcing identical MuJoCo XML/state fingerprint, prompt, and initial public state across all 8 group members at every reset).
     - Added periodic in-run DEV64 evaluation hook in `FlowRunner._log_step_metrics`: every `val_check_interval` iterations, saves training RNG via `pilot_training_rng()`, syncs rollout weights, restores the paired evaluation RNG, evaluates on the fixed DEV64 bank (`output/evaluations/step-XXX.json`), and restores training RNG before the next iteration.
   - [`scripts/[train]robocasa_flow_ppo.py`](../../scripts/%5Btrain%5Drobocasa_flow_ppo.py):
     - Supports both PPO and GRPO contracts, configurable `update_epoch` and `noise_level`, and `train_envs % (world * group_size) == 0`.

2. **Downstream Corpus Exporters, Steering Interfaces (`kv_bridge`, `velocity_bridge`, `flow_physical_action`), and 4 Meta-Learners (`ad`, `dpt`, `dit`, `ic_iql`):**
   - [`src/fluffy_bassoon/meta_rl/flow_history.py`](../../src/fluffy_bassoon/meta_rl/flow_history.py) & [`src/fluffy_bassoon/rl/pi05_residual.py`](../../src/fluffy_bassoon/rl/pi05_residual.py):
     - `export_corpus`: Generalized to arbitrary `(num_streams, num_versions * chunks_per_episode, 25)` shapes (including 130-iteration $H=600$ PPO and GRPO corpora) for `ad`, `dpt`, `dit`, and `ic_iql`.
     - `export_grpo_group_trials`: Packages each 8-rollout same-scene GRPO group into a 5-attempt within-scene adaptation trial (ordering failed attempts before successful attempts on the identical scene seed) for `ad`, `dpt`, `dit`, and `ic_iql`.
     - `FlowBridgeConfig`, `FlowBridgeCodec`, `HistoryFlowSteeringPolicy`, and `FlowBridgeMotor`: Full support for `kv_bridge` and `velocity_bridge` trained via advantage-weighted flow matching (`flow_matching_loss`) across `ad`, `dpt`, `dit`, and `ic_iql`, and compatible with `FrozenHistoryController` and `collect_case_attempts` for closed-loop 5-attempt evaluation.

---

## 2. Baige Paths & Verified Pins

- **Patched RLinf Vendor (includes chosen-step suffix critic fix):**
  - Path: `/mowenjie/fluffy-bassoon-2/sources/critic-selected-a1229239-r1`
  - Commit: `8f9bfa7661928ef8c17494b04f4d7bf3125634e1`
- **Python Environment:**
  - `/mowenjie/fluffy-bassoon-2/setups/robocasa365-native-20261101-r1/env/bin/python`
- **Launch Script:**
  - [`outputs/robocasa-ppo-grpo-meta-pipeline-20261104-r1/COMMAND.sh`](COMMAND.sh)

---

## 3. Next Steps for the Receiving Agent

1. **Stage 1 Pilot Gate (40 iterations on `OpenDrawer`, 8 GPUs per arm):**
   - Launch Fixed VLM-Critic PPO (`TASK_CONFIG=robocasa365_pi05_flow_sde_ppo_vlm_critic`) and Critic-Free GRPO (`TASK_CONFIG=robocasa365_pi05_flow_sde_grpo`) for `ROBOCASA_FLOW_MAX_EPOCHS=40`, `ROBOCASA_FLOW_VAL_INTERVAL=10`, `ROBOCASA_FLOW_SAVE_INTERVAL=10`, `ROBOCASA_FLOW_TASK=OpenDrawer` using [`COMMAND.sh`](COMMAND.sh).
   - Check `progress.json`, `updates/iteration-XXX-rank-00.json` (`approx_kl`, `clip_fraction`, `explained_variance`), and `evaluations/step-{010,020,030}.json` + `after-evaluation.json`.
2. **Stage 2 5-Task Source Collection (130 iterations / ~10M controls per task):**
   - Portfolio: `OpenDrawer`, `CloseToasterOvenDoor`, `CoffeeSetupMug`, `PickPlaceDrawerToCounter` (or `PickPlaceCounterToCabinet` per human portfolio selection), `TurnOnSinkFaucet`.
3. **Stage 3 Downstream Meta-Learner Matrix & Closed-Loop Evaluation:**
   - Export corpora via `export_corpus` (PPO & GRPO chronological streams) and `export_grpo_group_trials` (GRPO within-scene multi-attempt trials).
   - Train `ad`, `dpt`, `dit`, and `ic_iql` across `kv_bridge`, `velocity_bridge`, and `flow_physical_action`, then evaluate closed-loop 5-attempt adaptation on the 5 ID and 5 OOD tasks using `FlowBridgeMotor` / `FlowDirectMotor` and `collect_case_attempts`.
