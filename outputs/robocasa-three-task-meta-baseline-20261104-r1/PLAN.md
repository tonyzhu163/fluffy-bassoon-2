# Exploratory 3-Task Meta-Learning Baseline on Positive-Trend 10M Flow-SDE PPO Corpora

**Date:** 2026-11-04  
**Run ID:** `robocasa-three-task-meta-baseline-20261104-r1`  
**Status:** Prepared and CPU-qualified; ready to launch in parallel with Stage 1.  
**Scope Classification:** Exploratory baseline on the historical `0988b5f9` 10M Flow-SDE PPO corpora (logged separately from the official Stage 3 comparison on the new fixed-PPO / GRPO Stage 2 sources, preserving the student-admission rule in [`outputs/coordination.md`](../coordination.md)).

---

## 1. Source Tasks & Checkpoint Selection (`global_step_130` + Full `0..130` History)

We use the three positive-trend tasks from [`outputs/robocasa-flow-sde-10m-per-task-20261103-r1`](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md), keeping all **130 chronological policy versions** (`16,640` episodes / `199,680` H50 decision chunks / `9,984,000` physical controls per task) for the rollout history corpus and selecting the **last checkpoint (`global_step_130`)** as the `DPT` teacher across all three tasks:

| Task | BOS Cold Archive (`native-130-r1.tar`) | Archive SHA256 | Step-130 Teacher (`global_step_130`) Boundary SHA256 | Evidence Summary |
| --- | --- | --- | --- | --- |
| `OpenDrawer` | `/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1/corpora/OpenDrawer/native-130-r1.tar` | `c0817a8be0a250c357d8f40d57ec1508da1d634a5bac7e54ba98f3d2c1ef44c6` | `10deeac6761a5018ce8aa6eaed978d449e0f642552cfe68c8ae26bb56c613e57` | TRAIN `+4.61 pp`; DEV64 `44 -> 47/64`; 512-case `331 -> 369/512` (**+7.42 pp**, $p=0.0024$) |
| `CloseToasterOvenDoor` | `/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1/corpora/CloseToasterOvenDoor/native-130-r1.tar` | `788b6e3357bdd17786fbec1fabb6811a74b1f4e8ef1d985a05e28d6c64fd322a` | `5e525b1b8df3a41df79cd8d4d76ccae918d93eb12eac58c0ad9d9e49449583ef` | TRAIN `+8.91 pp`; DEV64 `22 -> 25/64`; 512-case `168 -> 212/512` (**+8.59 pp**, $p=0.0011$) |
| `CoffeeSetupMug` | `/mnt/bos/adgfyl/strong-plant-archives/robocasa-flow-sde-10m-20261103-r1/corpora/CoffeeSetupMug/native-130-r1.tar` | `1ad7c6674e642ca0da8d54687e62c0b31b16d1ddca01e0757b00d5514ef02da1` | `d04c3136edc34189ff7337983725a6378844a823c79c55cb28849532a94bc09f` | TRAIN `+4.84 pp`; DEV64 `19 -> 26/64` (`+10.9 pp`) |

### Rationale for `global_step_130` + Full `0..130` Trajectory
- **No corpus truncation loss:** Truncating at the 64-case DEV max (`step 30`, `step 1`, `step 20`) would discard 77%–99.2% of collected transitions (notably on `CloseToasterOvenDoor`, where DEV64 spiked at `step 1` due to small-sample noise while `step 130` is confirmed at `+8.59 pp`, $p=0.0011$ on 512 paired cases).
- **Consistent teacher:** `AD`, `DIT`, and `IC-IQL` train directly on the `(observation, executed_noise, reward, done)` rollout history; only `DPT` queries a teacher checkpoint to relabel `target_noise`. Using `global_step_130` across all three tasks eliminates post-hoc checkpoint selection bias and uses the checkpoints already retained on PFS.

---

## 2. Zero-Unpack BOS Streaming Preparation (`PREPARE-STREAM.py`)

To protect Baige PFS free space (>500 GiB free; 200 GiB floor) while Stage 1 is running, [`PREPARE-STREAM.py`](PREPARE-STREAM.py) **never unpacks the 110+ GiB `.tar` archives onto PFS**:
1. Verifies `<Task>/native-130-r1.tar.manifest.json` against the prepared receipt in `outputs/robocasa-flow-sde-10m-per-task-20261103-r1/CORPUS-ARCHIVE-<Task>-R1-PREPARED.json`.
2. Streams `<Task>/production-r1/environment/rank-{00..07}/training-i{001..130}-round00-chunk{01..12}.pt.gz` (`12,480` members = `199,680` H50 decision chunks) directly from `/mnt/bos/.../native-130-r1.tar` in memory, verifying every member's SHA256 against the manifest and stopping as soon as all `12,480` environment chunks are read (skipping `sampling/` and `training-rollouts/`).
3. Records the exact `(tar_data_offset, bytes, sha256)` of each `environment/` member in `TAR-INDEX.json` so downstream `kv_bridge` and `velocity_bridge` trainers can `os.pread` raw query-step observations on demand without unpacking.
4. Computes the frozen original VLM `2073`-d feature vector (`state25 + prefix2048 RMS`) and the `global_step_130` teacher's `50 x 12` physical action (using deterministic per-query `teacher_noise(member_sha256, row)`) in GPU batches of 32, checkpointing resumable shard groups every 50 batches.
5. Exports the four canonical `.npz` datasets (`ad`, `dpt`, `dit`, `ic_iql` $\times$ `train` slots `0..111` / `validation` slots `112..127`) via [`export_corpus`](../../src/fluffy_bassoon/meta_rl/flow_history.py).

---

## 3. Meta-Learner Training & Steering Interfaces

For each task (and pooled 3-task comparison):
1. **`flow_physical_action` (`DirectActionCodec` + `HistoryNoisePolicy` / `HistoryValueLearner`, deployed via `FlowDirectMotor`):**
   - `ad` and `dpt` trained via `fluffy_bassoon.meta_rl.vla_history_train` (`--context-length 192 --max-episodes-per-window 12`, 12 chunks/episode).
   - `dit` and `ic_iql` trained via `fluffy_bassoon.meta_rl.history_value_train` (independent causal $Q, V$ fit followed by frozen-value weighted policy extraction).
2. **`kv_bridge` and `velocity_bridge` (`FlowBridgeCodec` + `HistoryFlowSteeringPolicy`, deployed via `FlowBridgeMotor`):**
   - Uses [`sample_bridge_query_batch`](../../src/fluffy_bassoon/meta_rl/flow_history.py) over the same `.npz` history streams (`0..111` train, `112..127` validation).
   - Minimizes [`HistoryFlowSteeringPolicy.flow_matching_loss`](../../src/fluffy_bassoon/meta_rl/flow_history.py) through the frozen VLA suffix (`kv_bridge` rank-4 K/V edits or `velocity_bridge` bounded velocity delta), weighted by `1.0` (`ad`, `dpt`) or $\exp((Q - V)/\tau)$ from the trained causal critics (`dit`, `ic_iql`).

---

## 4. Closed-Loop 5-Attempt Evaluation Protocol

- **In-Distribution (ID):** `OpenDrawer`, `CloseToasterOvenDoor`, `CoffeeSetupMug` (held-out evaluation seeds disjoint from `2100000..2116639`, 50 cases/task $\times$ 5 attempts/case, horizon 600 / 12 H50 chunks).
- **Out-of-Distribution (OOD):** `CloseFridge`, `TurnOnMicrowave` (50 cases/task $\times$ 5 attempts/case).
- **Controls:**
  - `full` cross-episode memory vs. `episode`-reset memory (`mode="episode"`).
  - Native frozen anchors: `policy_0` (initial SFT) and `policy_130` (step-130 PPO teacher).

---

## 5. Compute Budget & Resource Plan

| Phase | Allocation | Estimated Wall Clock | GPU-Hours Ceiling |
| --- | --- | --- | --- |
| **Phase A:** Zero-unpack BOS streaming feature + `step_130` teacher export (`OpenDrawer`, `CloseToasterOvenDoor`, `CoffeeSetupMug`) | 3 $\times$ 1 GPU (or 1 $\times$ 4 GPUs sharded) | ~2.7 h per task on 1 GPU | **9.0 GPU-h** |
| **Phase B:** Meta-learner training (`ad`, `dpt`, `dit`, `ic_iql` across `flow_physical_action`, `kv_bridge`, `velocity_bridge`) | 3 $\times$ 1 GPU | ~2.0 h per task | **6.0 GPU-h** |
| **Phase C:** Closed-loop 5-attempt evaluation (ID + OOD panels) | 3 $\times$ 1 GPU | ~2.0 h per task | **6.0 GPU-h** |
| **Total Exploratory 3-Task Baseline Ceiling** | | | **21.0 GPU-h** |
