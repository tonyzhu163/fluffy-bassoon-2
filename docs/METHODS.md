# Methods and shared setup

The setup that current and future runs share. Per-run contracts go in the
run's `outputs/<run-id>/PLAN.md`. Older method contracts (Meta-World, ManiSkill,
RoboMimic, LIBERO source40) were removed on 2026-11-01; find them in
git history before that date. RoboTwin docs are in
[robotwin/](robotwin/), and literature reviews in [literature/](literature/).

## Motors

| Motor | Identity | Used for |
| --- | --- | --- |
| Official OpenPI **pi0.5 LIBERO40** | `gs://openpi-assets/checkpoints/pi05_libero`, config `pi05_libero`; staged receipt `ec064cce…d170`, motor parameter hash `c565…` | Flora PPO initialization ([FLORA-PPO.md](FLORA-PPO.md)), frozen-motor steering, DSRL, ICRL paper. Fine-tuned on LIBERO-Spatial/Object/Goal/10 (40 tasks, 1,693 demos). |
| **RLinf pi0.5 LIBERO-130 SFT** | `RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT@6222623f`, weights SHA `4d9089c9…6683`, own `norm_stats.json` `dae37d79…30f1`; RLinf `e1a82eb`, `rlinf-openpi 0.1.1` | Earlier source-corpus pilots only. Trained on all 130 LIBERO tasks, so no LIBERO task is unseen by it; not allowed as the Flora PPO initializer. |
| Original **Pi0** `pi0_libero` | H50 / execute 20 | DSRL replication only |

pi0.5 interface: 10-action chunks, 32 initial-noise coordinates, 10 Euler flow
steps, decoding 7 LIBERO control coordinates. Inputs are two RGB views, 8 robot
state coordinates and the instruction. The whole motor stays frozen. Check the
parameter signature before and after every run.

## Steering interface and DSRL

- **Native:** independent H10×32 Gaussian noise per chunk.
- **Tied noise:** one 32D vector per chunk, tiled over H10. Untrained tied
  noise is a required control: it often beats native sampling.
- **DSRL (PixelSAC):** the actor sees 64×64 RGB plus 8D proprioception and
  outputs the 32D tied noise. Batch 256, UTD 20, 10 critics, LR
  1e-4/3e-4/3e-4. Gaussian warmup runs until replay has more than 500 query
  records. Reward is −1 per query and 0 on success, with discount
  0.999^actual steps. Actor weights stay fixed within an episode.
- **Panels:** train on states 0–39; query fixed states 40–49 with a fixed
  per-case seed bank at every milestone. Query records never enter replay or
  source pretraining. Report initial native, tied, initialized-SAC and final
  native panels, and never select a favorable checkpoint.

## LIBERO protocol for the paper

- Source: all 40 LIBERO40 tasks. Targets: full LIBERO90, LIBERO-Plus and
  LIBERO-Pro. No target histories, labels or feedback enter source training,
  curriculum or checkpoint selection.
- Primary one-shot: one complete support episode, then a separate query
  episode, with fixed weights. Report support and query success separately,
  and count both episodes' interactions.
- Required controls for any memory claim: independently trained static and
  episode-reset controllers on identical data, reward/history interventions,
  native and tied noise, and ordinary DSRL.
- Accounting: no fixed action cap; charge source actions to the learner that
  used them. Count all qualification, settling, failures and retries. Report GPU-seconds
  and actions per arm.

Full claim boundaries are in [PAPER.md](PAPER.md#decisions-and-claim-boundaries).

Head-to-head claims require150–200 cases or several independent noise draws
per case, with paired uncertainty and cases as the resampling unit. Treat
small50-case differences as exploratory; a fixed10/50 cutoff is not a
significance test. Repeated-attempt memory comparisons use the same motor
streams with memory cleared before each attempt, and inference context must
match the training window or carry an explicit context-length control.

## Code

Evaluation readiness belongs to each model. Once its declared stopping rule
is met, freeze its selected checkpoint and dispatch its ID/OOD readout without
waiting for other learners or controls. Publish that model's results when ready;
only the final comparison waits for every method. Keep evaluation outcomes
outside the stopping and checkpoint-selection rules. History training publishes
an atomic `training-status.json` per fit; readout dispatch selects explicit
ready methods rather than admitting a whole experiment at once. Archived
combined drivers require an explicit `--legacy-combined` opt-in.

On schedulers that treat master exit as job success, keep the master alive
until every node in that allocation records its completed workers. A scheduler
success badge cannot replace the full episode and worker receipts. This
synchronization belongs to the allocation and does not gate other models.

Verify paired robot cases from native scene/state and camera observations before
actions. Qualify and record backend numerical tolerances using unchanged-input
replays; transformed feature hashes alone are not physical case identifiers.
Keep original policy observations and retain per-episode pairing evidence.

| Path | Contents |
| --- | --- |
| `src/fluffy_bassoon/meta_rl/` | Trial environments, PPO/SAC, DSRL drivers (`vla_target_online_dsrl*`, `dsrl_pi0_replication`), headroom screen (`vla_headroom_screen`), RLinf motor (`vla_rlinf_libero`), RLinf native PPO (`rlinf_native_ppo*`), source40 pipeline (`vla_source40_*`), RoboTwin (`robotwin_*`), Meta-World (`metaworld_*`) |
| `src/fluffy_bassoon/meta_rl/frozen_steering_history.py`, `history_value.py`, `history_value_train.py` | Matched AD/DPT/DIT/in-context IQL corpora; causal Q/V objectives and two-stage value/policy fits with full recovery. DPT alone changes query labels; the other three use executed steering actions. Frozen inference uses the common history controller. |
| `src/fluffy_bassoon/rl/` | pi0.5 feature bridge and residual hooks, LIBERO goal/variant helpers, RoboTwin runtime, ManiSkill motor |
| `src/lerobot_policy_ctm/` | Recurrent cores: persistent CTM, CMM, looped Transformer, Sakana CTM reference |
| `scripts/diagnostics/` | One launcher per experiment (`vla_*.sh`, `robotwin_*.py`, `dsrl_pi0_replication.sh`) |
| `scripts/setup/` | DSRL Pi0 and RoboTwin runtime setup |
| `scripts/gpu_availability.sh`, `scripts/[train]gpu_probe.py` | School-cluster availability; in-job GPU isolation probe |
| `patches/` | RLinf flow-noise key patch, applied only to an isolated copy |

The package names are kept from `tonyzhu163/fluffy-bassoon` so copied code
stays byte-identical ([PROVENANCE.md](../PROVENANCE.md)).

## Environments and tests

```bash
uv sync
uv run pytest
```

Run tests on allocated Linux resources, not macOS; `tests/test_meta_rl_cuda.py`
needs a GPU step. Robot runtimes are separate environments reused in place on
project NVMe (for example the RLinf runtime `envs/rlinf-pi05-libero-e1a82eb-20261029-r1`
on thu2); each run's plan records the exact environment it used.
Cluster access, storage and GPU rules are in [AGENTS.md](../AGENTS.md).
