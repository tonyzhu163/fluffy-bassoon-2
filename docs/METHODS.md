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
state coordinates and the instruction. During steering, the whole motor stays
frozen. Check the parameter signature before and after every run.

## Steering interface and DSRL

The primary source/student route is **full-action-expert Flow-SDE PPO →
action histories → supervised AD/DPT learned-KV bridge**. Source PPO updates
the action expert and value head with the VLM frozen. Student fitting freezes
the original VLM and action expert; only the causal history model and learned
KV adapter update. Deployment freezes all weights.

History contains actual observation/action/reward/boundary tuples. AD targets
the actual source action at each chronological decision; DPT-style fitting
queries the qualified final teacher on the same saved observations, preserving
the executed context and feedback. No historical KV codes are invented. Both
students use action-derived flow-matching supervision:

\[
x_\tau=(1-\tau)a^*+\tau\epsilon,\qquad
\mathcal L=\mathbb E\|v_{\bar\theta,\psi}(x_\tau,\tau,o,H)
                         -(\epsilon-a^*)\|^2.
\]

The action target must use the native model's qualified encoding and active
loss mask. Decoded binary commands do not uniquely recover pre-decoding
continuous model outputs. Gradients traverse the frozen suffix into the
learned adapter; training cannot wrap that suffix in `no_grad` or detach its
output. Masked, token-local low-rank KV edits have zero-initialized output
bases. Their layer/K/V coverage, rank and scale are run-bound; the earlier
fixed random rank32 map is a distinct interface. Native identity, frozen
weights, gradient flow and held-out imitation need numerical/native checks
before own-history evaluation. See the
[implementation contract](../outputs/flow-sde-kv-bridge-20261103-r1/PLAN.md).

Full-expert source learning and frozen student controllability are separate
questions. Full-expert PPO is a strong online comparison; matched online
learned-bridge PPO is needed when a claim holds trainable capacity fixed.
Shared source interactions count once across AD/DPT, with each student's
fitting and deployment cost reported separately. A direct-action student
bypasses the expert and remains a separate baseline.

The following noise-steering recipe remains the DSRL comparison:

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

New joint-steering readouts use `fixed_batches_shared_scenes_v2`: fixed-width
waves retain case order and model row positions, including after short episodes
finish. Padding rows make no simulator steps and contribute no learner samples.
Both arms share a canonical scene bank and reject unequal native geometry,
initial state or initial camera pixels before acting. Counter placement regions
are sorted geometrically before the seeded choice to remove process-dependent
set iteration. Native Gaussian draws remain keyed by case and chunk; live
observations still determine the learned steering mean at every chunk. Historical
commit-pinned runs retain their original protocol.

| Path | Contents |
| --- | --- |
| `src/fluffy_bassoon/meta_rl/` | Trial environments, PPO/SAC, DSRL drivers (`vla_target_online_dsrl*`, `dsrl_pi0_replication`), headroom screen (`vla_headroom_screen`), RLinf motor (`vla_rlinf_libero`), RLinf native PPO (`rlinf_native_ppo*`), source40 pipeline (`vla_source40_*`), RoboTwin (`robotwin_*`), Meta-World (`metaworld_*`) |
| `src/fluffy_bassoon/meta_rl/frozen_steering_history.py`, `history_value.py`, `history_value_train.py` | Matched AD/DPT/DIT/in-context IQL corpora; causal Q/V objectives and two-stage value/policy fits with full recovery. DPT alone changes query labels; the other three use executed steering actions. Frozen inference uses the common history controller. |
| `src/fluffy_bassoon/rl/` | pi0.5 feature bridge and residual hooks, LIBERO goal/variant helpers, RoboTwin runtime, ManiSkill motor |
| `src/lerobot_policy_ctm/` | Recurrent cores: persistent CTM, CMM, looped Transformer, Sakana CTM reference |
| `scripts/diagnostics/` | One launcher per experiment (`vla_*.sh`, `robotwin_*.py`, `dsrl_pi0_replication.sh`) |
| `scripts/setup/` | DSRL Pi0 and RoboTwin runtime setup |
| `scripts/gpu_availability.sh`, `scripts/[train]gpu_probe.py` | School-cluster availability; in-job GPU isolation probe |
| `patches/` | RLinf patches (flow-noise checkpoint keys; critic value at the sampled denoising step), applied only to an isolated copy |

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
