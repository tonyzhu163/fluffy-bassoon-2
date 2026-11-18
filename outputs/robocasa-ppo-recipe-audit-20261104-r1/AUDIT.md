# RoboCasa Flow-SDE PPO: literature and recipe audit

Nov 4 2026. Scope: the five-task Flow-SDE PPO sources
([10M run](../robocasa-flow-sde-10m-per-task-20261103-r1/RESULT.md),
config [`robocasa365_pi05_flow_sde_10m.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_10m.yaml))
and the checkpoint-selection curves made from them. This builds on the
[Nov 2–3 literature review](../../docs/literature/robocasa-ppo-review.md),
which already covers StructRL, PolicyWeave, SEES, Z-1, DRILL, MiDAS and RLinf.
New here: πRL's own ablation tables, RLinf's shipped π0.5 configs, DPPO's
small-σ safeguards, RLinf issue #1044, and the logged first-update optimizer
metrics. CPU and reading only: no jobs, SGD or simulator calls.

## Verdict

The PPO implementation is mostly correct. The one confirmed code bug, the
rollout/replay critic mismatch, is already patched (`70b6004b`). Earlier
audits verified the GAE boundaries, first-success reward, train-mode vision
path, RNG and likelihood parity.

The **recipe** is the bigger problem. Three settings sit on the losing side
of every published ablation and every RLinf π0.5 config. Combined, they make
the chunk-level PPO ratio about **8–10× more sensitive** to parameter changes
than in any published working recipe. The update gradient then comes mostly
from denoising steps whose noise barely changes the executed action. The
measured first update shows this: approx-KL 0.151, clip fraction 0.262 and
grad norm 65, clipped to 1.0.

The flat curves fit an under-powered, noise-dominated optimizer. They don't
show that PPO can't improve these tasks. Small sparse-reward gains also match
the RoboCasa literature: StructRL's sparse PPO gains only +1.8 pp on π0.5.

## Our settings vs published working recipes

| Knob | Ours (10M sources) | πRL π0.5 LIBERO (Table 7) | RLinf RoboCasa365 OpenDrawer | πRL ablation finding |
| --- | --- | --- | --- | --- |
| Executed chunk H | **50** | 5 (Long: 10) | 5 | H 5/10/20 → eval 94.5/95.5/89.2%; larger H hurts credit assignment and explained variance |
| Denoise steps K | **10** | 3–5 | 5 | K 1/2/4/8 → eval 88.5/97.0/94.5/86.7%; large K "complicates training" |
| Flow-SDE noise a | **0.3** | 0.5 (Object/Goal 0.3 with H5, K5) | 0.5 (pi0 default) | a 0.2/0.5/0.8 → eval 73.1/94.5/98.1%; low a "exhibits instability … significantly higher clip fraction" |
| Critic placement | **action expert** (`value_after_vlm: false`) | **VLM** (required for π0.5: state is in the prompt) | expert (π0, not π0.5) | V_vlm better EV even for π0 |
| Global minibatch | **256** | 2048 | 2048 | — |
| Episodes / update | **128** | 512 (64×8) | 128 (32×4) | — |
| Actor/critic LR, γ, λ, clip | 5e-6 / 1e-4, .99, .95, .2 | same | same | matched; not the problem |
| Update epochs | 4 | 1–4 | 4 | matched |

Sources: [πRL Tables 6–8](https://arxiv.org/html/2510.25889v2) and RLinf
`main` configs
([libero_10 π0.5](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml),
[robocasa365 OpenDrawer](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/robocasa365_opendrawer_ppo_openpi.yaml),
[model/pi0_5](https://github.com/RLinf/RLinf/blob/main/examples/embodiment/config/model/pi0_5.yaml)).
Every RLinf π0.5 PPO config uses H5 (or H10), K3–5, noise 0.5 and
`value_after_vlm: True`. We use none of these.

## Finding 1 (P1, recipe): chunk-level ratio over 600 low-σ coordinates

RLinf's Flow-SDE PPO makes one random denoising step per decision stochastic.
It sums that step's Gaussian log-prob over every executed coordinate
(`logprob_type: chunk_level`). For us that is 50 × 12 = **600 coordinates**.
πRL uses 35–70. With a = 0.3 and K = 10, the transition σ runs from 0.300 at
index 0 down to **0.032** at index 9. A mean shift δ changes the summed
log-ratio by about √(dims)·δ/σ:

| Recipe | dims | σ min / max | √dims/σ, worst step | √dims/σ, mean over steps |
| --- | ---: | --- | ---: | ---: |
| **Ours** (H50, K10, a .3) | 600 | 0.032 / 0.300 | **775** | **294** |
| RLinf RoboCasa365 (H5, K5, a .5) | 60 | 0.112 / 0.500 | 69 | 35 |
| πRL π0.5 Long (H10, K5, a .5) | 70 | 0.112 / 0.500 | 75 | 37 |
| πRL π0.5 Object (H5, K5, a .3) | 35 | 0.067 / 0.300 | 88 | 44 |

This has three consequences:

1. **The ±0.2 trust region binds after a tiny real action change.** The
   OpenDrawer first update logged approx-KL **0.151**, clip fraction
   **0.262** and mean |ratio−1| 0.47
   ([Q1-AUDIT](../robocasa-flow-sde-10m-per-task-20261103-r1/OpenDrawer/Q1-AUDIT.json)).
   Production updates stayed at approx-KL 0.05–0.10. DPPO tunes for a 10–20%
   clip fraction. πRL names high clip fraction at low noise as its
   instability symptom.
2. **The gradient is dominated by the least informative samples.** The
   score term for each sample scales as A·ε/σ. Samples drawn at late
   denoising indices perturb the final action by about 0.03 in normalized
   units, so their advantage is almost uncorrelated with ε. Yet they carry
   about 10× larger gradients than index-0 samples. With grad-norm clipping
   at 1.0 against a raw norm of 65, the normalized step direction is mostly
   this noise.
3. **The reward signal is diluted.** There are only 12 decisions per 600-step
   episode, and each 50-step chunk is one macro-action. Sparse success gives
   little credit to individual chunks; explained variance was 0.03 at the
   start and about 0.28 late.

DPPO addresses exactly this. Its released configs floor σ for sampling and
for likelihood (`min_sampling_denoising_std: 0.1`,
`min_logprob_denoising_std: 0.1`). They clip per denoising step with a
smaller ε for late steps, discount early steps, and use H4 with batch 7500
([config](https://github.com/irom-princeton/dppo/blob/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml),
[paper](https://arxiv.org/abs/2409.00588)). RLinf has an `ignore_last` flag
that excludes the final, lowest-σ step from sampling. We leave it off.

## Finding 2 (P1, recipe): wrong critic placement for π0.5

`discrete_state_input: true` puts the robot state into the VLM prompt. πRL
attaches the π0.5 critic to the VLM output for that reason, and every RLinf
π0.5 config sets `value_after_vlm: True`. Our critic is a 512-256-128 MLP on
detached action-expert features. Those features are taken at a random noisy
denoising state, so the value is a function of (s, x_k) rather than s.
Combined with the confirmed mean-vs-chosen-step bug, the critic gave
near-Monte-Carlo advantages for much of training. Logged value-clip ratio at
the first update was 0.55. The bug fix alone does not move the critic to
the published π0.5 placement.

## Finding 3 (P2, recipe): small, noisy minibatches

1,536 decisions per update are split into six minibatches of 256 for four
epochs. πRL and RLinf use 2,048 per minibatch at the same LR. With
grad-norm clipping on every step, a smaller batch at the same LR means
noisier unit-norm steps, not smaller ones. DPPO notes that denoising-MDP PPO
needs larger batches than Gaussian-policy PPO.

## Finding 4 (P3, code): likelihood omits 20 padded dimensions

`model_action_dim` is 32, but the log-prob is sliced to the 12 executed
dimensions (`[:, :, :action_chunk, :action_env_dim]`). At any stochastic
index before the last, noise in padded dimensions 12–31 feeds the remaining
ODE steps, and so the executed action. Their θ-dependent mean is left out
of the ratio. The effect is probably small because SFT data keeps the
padding at zero. The fully consistent alternative is to include all 32
dimensions or zero the padded noise. This is upstream RLinf behaviour, not
our patch.

## Checked and fine

- PPO constants (LR, γ, λ, clip, dual clip, no KL/entropy) match πRL, RLinf
  and StructRL.
- GAE boundaries, first-success latch and post-success masks were verified
  numerically on saved batches. StructRL's done-shift patch does not apply.
- The train-mode vision bug was fixed before production (`0988b5f9`). Zero
  initial likelihood gap holds on all tasks.
- Rollout RNG is distinct across ranks.
- The critic mean-vs-chosen-step mismatch is confirmed and patched
  (`70b6004b`), with a value-parity gate added. All five 10M sources predate
  the fix.
- A frozen VLM with a trainable expert matches every reference.

## Reading the checkpoint curves

- DEV uses 64 cases per point (Wilson ±~12 pp). Picking the best of 14
  checkpoints on the same bank has a winner's-curse bias, so the stars are
  optimistic. TEST128 keeps that from becoming a claim, but the existing
  power warning stands: 128 paired cases detect +9 pp only 13–34% of the time.
- The curves are flat between updates 30 and 130, and Stove and Drawer drift
  down. That matches Finding 1: updates that are clipped and noise-dominated
  move the policy without a consistent direction.
- The closest external analogue, [RLinf #1044](https://github.com/RLinf/RLinf/issues/1044),
  uses π0.5 on RoboCasa with H50 and K10, our cadence. It reports
  `success_once` oscillating around 0.5–0.6 with no upward trend. It is
  still open with no tested fix.

## Literature on PPO for VLAs and robot policies

| Work | Benchmark | Algorithm and recipe | Reported gain | Relevance |
| --- | --- | --- | --- | --- |
| [πRL](https://arxiv.org/abs/2510.25889) | LIBERO, ManiSkill (not RoboCasa) | Flow-SDE/Flow-Noise PPO; H5–10, K3–5, a .5, VLM critic for π0.5, batch 2048 | π0.5 few-shot 77.1→98.3% LIBERO | Reference recipe; its ablations directly indict our H/K/a |
| [StructRL](https://arxiv.org/html/2609.36352v1) | RoboCasa365 Composite-Seen, 16 tasks | π0.5 Flow-SDE PPO; a .5, VLM critic, 512 rollouts/iter, 100 iters | sparse 39.3→41.1%; structured 45.8% | Best RoboCasa PPO comparator; sparse PPO gains are small |
| [PolicyWeave](https://arxiv.org/html/2609.33125v1) | RoboCasa365, 18 atomic tasks | GR00T LoRA experts, Flow-SDE PPO, 2 epochs, LR 1e-5/1e-4, ~24M steps | merged 64.7→74.1% | Atomic RoboCasa PPO works, with target-task SFT and much larger budgets |
| [SEES](https://arxiv.org/html/2609.32698v1) | RoboCasa365 composite | GR00T action-head LoRA PPO, H16, K4, restored failure states | 10.2→13.1% | Shorter chunk; restarts from failure states |
| [Z-1](https://arxiv.org/html/2606.31846v1) | RoboCasa, 24 tasks | π0.5 **GRPO**, prefix branching | 67.4→80.6% | Largest RoboCasa RL gain, not PPO |
| [DRILL](https://arxiv.org/html/2610.30814v1) | RoboCasa | Sim-PPO control | 12.4% | Small-budget control |
| [RLinf #1044](https://github.com/RLinf/RLinf/issues/1044) | RoboCasa π0.5 | H50, K10 PPO | flat 0.5–0.6 | Same failure pattern as ours |
| [DPPO](https://arxiv.org/abs/2409.00588) | RoboMimic, Furniture-Bench | diffusion PPO; H4, σ floor 0.1, per-step clip, fine-tune late steps only | large | Small-σ safeguards we lack |
| [RLinf-VLA](https://arxiv.org/abs/2510.06710) | LIBERO, ManiSkill | PPO vs GRPO | PPO ≥ GRPO | Framework baseline |
| [FPO for VLAs](https://arxiv.org/abs/2510.09976) | flow VLAs | likelihood-free PPO-style ratio | — | Alternative to Gaussian step likelihoods; removing clipping hurt (45.1%) |
| [ReinFlow](https://arxiv.org/abs/2505.22094) | RoboMimic | learned, bounded noise net | — | Bounded σ, same lesson as DPPO |
| [MiDAS](https://arxiv.org/html/2608.11363v3) | RoboCasa, 3 tasks | frozen π0.5 + off-policy residual | 89.3% | Not PPO; frozen-base evidence |

No published work shows sparse-reward PPO on RoboCasa atomic tasks with a
Human300 multitask π0.5 and an H50/K10 cadence. Every RoboCasa PPO success
uses target-task SFT, shorter chunks, larger batches, structured rewards, or
all of these.

## Recommended next test

Change one family at a time on one task, such as CloseToasterOvenDoor (most
headroom, +8.9 pp TRAIN). Compare against the patched-critic baseline, and
judge each arm by TRAIN slope and clip fraction / approx-KL before spending
DEV/TEST:

1. **Noise and ratio conditioning:** a = 0.5 and `ignore_last: true`
   (or a σ floor of about 0.1 for likelihood, DPPO-style). This is cheap and
   needs no change to the action cadence.
2. **Cadence:** H10, K5, matching πRL Long and the StructRL paper. Keep
   predicted horizon 50 and execute 10, which gives 60 decisions per episode.
3. **Critic:** `value_after_vlm: true` with a new value head. This is not
   resumable from the old head.
4. **Batch:** at least 512 episodes per update, minibatch 2048, using
   execution-batch tuning only.

The recommended ordering is 1 → 2 → 3, with batch size matched to the
references throughout. Each is a recipe change for a new run identity. None
alters the active checkpoint-selection or endpoint runs.
