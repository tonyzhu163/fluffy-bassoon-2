# Per-task Flow-SDE PPO teachers, about 10M controls per task

2026-11-03. **Plan and static audit only; nothing launched.** Needs a Baige
owner after the pre-launch fixes below land on `main`. Governing decision:
[PAPER.md source campaign (2026-11-03)](../../docs/PAPER.md#agreed-design).
Live status: [coordination](../coordination.md).

## Human decisions (2026-11-03)

- Simulator first-success reward for these runs. No Flora reward here.
- One independent full action-expert Flow-SDE PPO run per task, **8 GPUs
  each**, all tasks in parallel, about **10M environment controls per task**.
- **KL off**: `kl_beta=0`, no KL early stopping. Keep logging approximate KL,
  clip fraction and the other PPO metrics.
- Collect and audit the complete PPO corpus of every task before any
  meta-learner (AD, DPT, DIT, value-based offline ICRL) or steering adapter
  is fit.
- Design and audit this before launch.

## Why this design

The only completed run of this recipe
([OpenDrawer, 18 updates](../robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/RESULT.md))
failed its gate: TRAIN 221→224/384, paired held-out 35→43/64 (p=.18). It had
288 optimizer steps. At 10M controls the same recipe gets about 3,100
optimizer steps per task. **Scale is the only intended scientific change.**
Everything else stays at the qualified recipe, so a flat curve at 10M points
at the recipe, not at a new change. The four fixes this plan adds are needed
to generalize from OpenDrawer, to keep the full data and to check the log-ratio.
They are not recipe changes.

## Tasks

Category-balanced 5 ID + 5 OOD portfolio, adopted by the human on 2026-11-03
([TASK-PORTFOLIO.json](TASK-PORTFOLIO.json)). It supersedes the
[2026-11-02 selection](../robocasa-five-task-flora-ppo-20261102-r1/TASK-PORTFOLIO.json).
Each `atomic_seen` category appears in both sets.

| Category | ID: PPO teachers (native first success) | OOD: held out (native first success, horizon) |
| --- | --- | --- |
| Open/close | OpenDrawer 29/50 (58%); CloseToasterOvenDoor 5/20 (25%) | CloseFridge 28/50 (56%, 900); OpenStandMixerHead 13/20 (65%, 450) |
| Pick-and-place | PickPlaceDrawerToCounter 23/50 (46%); CoffeeSetupMug 17/50 (34%) | PickPlaceCounterToStove 10/20 (50%, 600) |
| Buttons/knobs | TurnOffStove 7/50 (14%) | TurnOnMicrowave 6/20 (30%, 450); TurnOnSinkFaucet 10/20 (50%, 600) |

Rates come from the [five-task qualification](../../docs/FINDINGS.md), the CloseFridge
matching eval50 and the 20-case [atomic_seen sweep](../robocasa-atomic-baseline-sweep-20261103-r1/RESULT.md)
(coarse). Removed: PickPlaceCounterToCabinet (74%, fails the 5–70% admission
the runner enforces) and OpenCabinet (75%, horizon 1,050, a near copy of
OpenDrawer). Both sets now cover open and close, and turn on and turn off.
CounterToStove is the OOD set's only pick-and-place task in range. For ID,
the runner's 64-case before panel at 600 controls is the qualification;
CloseToasterOvenDoor's native horizon is 450, so its rate may rise at 600.
OOD tasks need a fresh 50-case native panel at their own horizons
(`5x0100` seeds) before any meta-learner readout.

**Horizon 600 controls for every task** (12 H50 chunks), the human's
2026-11-02 selection. Observed first-success maxima were 279/538/538/317/524
([timing](../robocasa-pi05-success-timing-20261102-r3/RESULT.md)). OpenDrawer
moves from 400 to 600, so its before panel is new rather than reused.

## Contract per task

Unchanged from [`robocasa365_opendrawer_pi05_sim_ppo.yaml`](../../configs/native_ppo/robocasa365_opendrawer_pi05_sim_ppo.yaml)
unless the row says otherwise.

| Item | Setting |
| --- | --- |
| Motor | public pi0.5 Human300 c484/75000 torch conversion, 10 flow steps, 3 views (same hashes as the prior run) |
| Trained | action expert + value head; VLM frozen, hash-checked every update |
| Exploration | `flow_sde`, noise 0.3, one stochastic denoise step |
| Reward | +1 at first success, γ 0.99, λ 0.95, chunk-level; later chunks masked |
| Loss | clip 0.2, dual clip 3.0, value clip 0.2, Huber 10, entropy 0, **kl_beta 0, no KL stop** |
| Optimizer | AdamW, actor 5e-6 / value 1e-4, betas (0.9, 0.95), wd 0.01, grad clip 1.0 |
| Episodes per iteration | 128 (scientific batch, unchanged) |
| Horizon | **600** (was 400) → 1,536 chunk samples/iteration, global batch 256, 4 epochs = **24** optimizer steps/iteration (was 16) |
| Iterations | **130** → 16,640 TRAIN episodes, 9,984,000 TRAIN controls |
| Eval | same 64 fixed cases per task before and after, flow ODE, full 600 controls (76,800 controls) |
| Execution layout | 8 env/rollout ranks, default **16 envs/GPU × 1 rollout epoch** (128 × 1); 64 × 2 and 32 × 4 are the same science (see Layout) |
| Actor ranks | 2, the qualified rank-local GAE normalization (moving to 8 changes advantage normalization; not adopted) |
| Checkpoints | full DCP + worker sidecars every **10** iterations and at 130 (~0.7 GB each) |
| Seeds | portfolio task number x (OpenDrawer 1 … CoffeeSetupMug 5): TRAIN `2,000,000 + 100,000·x + episode index`, before/after eval `600100 + 10,000·x` (`flow_seeds`); disjoint from the 1x/4x/5x/7x panels |

Per task: about 10.06M controls. At the old 4 envs/GPU layout this would be
~15 h wall / ~120 GPU-h. The 16-env layout should roughly halve that (see
Layout); Q1 measures it. Five tasks: ~300–600 GPU-h.

## Schedule at a glance (reviewed with the human, 2026-11-03)

Framework: RLinf's own embodied PPO (patched RoboCasa365 copy, `8a1233c3`)
with an FSDP actor; our wrapper only records sidecars, checkpoints and the
action journal. No other RL framework is involved.

| Unit | Per task |
| --- | --- |
| Decision (PPO sample) | one H50 chunk = 50 controls of 12-D commands; 10 flow steps, one Flow-SDE step (noise 0.3) |
| Episode | 600 controls = 12 decisions |
| Iteration | 128 episodes = 1,536 decisions = 76,800 controls, one policy version |
| Update | 4 PPO epochs × 6 minibatches of 256 decisions = 24 optimizer steps; 2 actor GPUs × 8 micro-batches of 16 |
| Run | 130 iterations = 16,640 episodes = 199,680 decisions = 9,984,000 controls; 3,120 optimizer steps; 13 checkpoints |
| Eval | 64 fixed cases before and after, 600 controls each; total 10,060,800 controls |

`rollout_epoch` and `update_epoch` are different knobs. `rollout_epoch` is
the number of back-to-back collection rounds per iteration with the same
weights. RLinf's recipe uses 32 envs × 4 rounds; we use 128 envs × 1 round.
Both give 128 episodes per update; the prior run's saved batches show all
rounds of an iteration in one batch. `update_epoch` is the number of SGD
passes over that batch and stays at the recipe's 4.

Compared with the flat OpenDrawer run (18 iterations, 1,024 decisions,
16 steps/iteration, 288 steps, ~0.97M controls), every hyperparameter is
unchanged. Only the horizon (400→600) and the scale (~11× the optimizer
steps) change.

## Layout: why 128 episodes per update, and rollout epochs 1/2/4

Episodes per policy version = total envs × rollout epochs. RLinf collects all
rollout epochs with one policy and then updates once, so 128 × 1, 64 × 2 and
32 × 4 are the same batch from the same policy; only wall time differs. The
seed formula now gives the same 16,640 TRAIN cases for all three (tested).
One round of 16 envs/GPU is fastest: measured single-GPU collection was
83/139/151/156 controls/s at 8/16/32/48 lanes, against ~35/s per GPU at the
old 4 lanes. Those numbers come from our batched collector, not the native
RLinf workers, so Q1 re-measures them.

48 envs × 8 GPUs = 384 episodes per update would be a recipe change, not a
speed-up. It gives ~12% more throughput than 16/GPU, but only 43 policy
versions at 10M instead of 130, which is coarser learning histories for AD.
It also needs 72 SGD steps per batch at the fixed minibatch of 256 (more
off-policy) or a 3× larger minibatch (one third of the optimizer steps).
And it does not fit memory: 48 lanes peaked at ~76 GiB, while the actor
colocated on GPUs 0–1 already peaks at ~35 GiB, against 96 GiB per card.
16 lanes peaked at ~29 GiB, which fits beside the actor. No action cap applies (human,
2026-11-03); the journal still charges every control.

## Data retained per task

| Needed by | Field | Current state |
| --- | --- | --- |
| all students | decision observation (3 RGB views, state, instruction) per chunk | saved, `environment/…-chunkNN.pt.gz` |
| all students | executed 50×12 command, reward, termination, truncation | saved |
| AD ordering | producing policy version per episode | saved (collection iteration) |
| DIT / IQL | next observation after the last chunk, or explicit terminal flag | **missing**, fix F5 |
| joins | explicit global episode ID in env, sampling and actor records | **missing**, fix F6 |
| DPT | final teacher checkpoint; labels queried offline later | saved at 130 |
| diagnostics | old log-prob, value, advantage, return, mask per sample | saved in actor batch |
| optional | full 11-state flow chain, actor-batch images | saved, duplicated; see storage |

## Decisions (2026-11-03, delegated by the human)

- **Tasks:** the balanced ID/OOD portfolio above (human adopted it).
- **Storage:** keep the env chunk records (the student corpus) and the full
  sampling records (chains, Gaussian parameters, RNG). Drop from the actor
  batch only what those already hold: `forward_inputs.observation/image`,
  `forward_inputs.observation/wrist_image`,
  `forward_inputs.observation/extra_view_image` and `forward_inputs.chains`.
  The batch then keeps tokens, state, actions, log-probs, values,
  advantages, returns and masks. At 199,680 samples per task the uncompressed
  actor images alone would be ~118 GB. The estimate after the drop is
  ~120 GB/task, ~600 GB for five, so Q1 must measure it. Q1 keeps everything and checks the
  dropped actor tensors byte-for-byte against the env/sampling records
  before the production segment enables the drop
  (`ROBOCASA_FLOW_ACTOR_BATCH_DROP`). Each closed task moves to BOS.
- **Time limit:** two segments per task. Q1 runs the before panel and
  iteration 1 with a pre-placed `CHECKPOINT_AND_EXIT` marker, which saves a
  full boundary and exits. Production resumes from that boundary with the
  drop enabled and `TASK_WALL_SECONDS=86400` (24 h vs ~8–15 h expected).
  That exercises the resume path (F8) before the long run. If the projected
  finish nears the deadline, the owner places the marker again and resumes
  from the next update; the 180 s deadline guard is the last resort and
  loses up to 9 iterations back to the last 10-iteration checkpoint. The
  resume needs a before-panel reference JSON bound to the resumed checkpoint
  (the prior run's `BEFORE-RECOVERY.json` format).
- **C2 settled from saved data:** the prior run's `old_logprobs` are
  `[…, 50, 12]`, so the ratio uses the 12 robot dimensions and no padded ones.

## Fix status (2026-11-03)

F1–F6 are fixed on `main` and covered by CPU tests in
`tests/test_robocasa_flow_ppo.py`. The new config is
[`robocasa365_pi05_flow_sde_10m.yaml`](../../configs/native_ppo/robocasa365_pi05_flow_sde_10m.yaml);
`flow_contract()` derives the horizon, schedule, seeds, step count and action
ceiling from it, and the driver refuses `kl_beta != 0`. For F7, the storage
switches exist (`sampling_record: full|scalars`, `actor_batch_drop`); their
defaults keep everything until Q1 measures the bytes. F8 is open: the wall
clock is set by the launch command, and the resume path is generalized but not
yet exercised. Still to do: run the RLinf-dependent tests on allocated Linux,
and the GPU checks C1–C3.

Launch per task: `ROBOCASA_FLOW_TASK=<task>`, plus
`ROBOCASA_FLOW_TRAIN_SEED` / `ROBOCASA_FLOW_EVAL_SEED` from `flow_seeds(task)`.
Use a fresh `ROBOCASA_SIM_PPO_ACCOUNTING` root per task.

## Pre-launch static audit

The audit read [`rlinf_robocasa_flow_ppo_runtime.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo_runtime.py),
[`rlinf_native_ppo_runtime.py`](../../src/fluffy_bassoon/meta_rl/rlinf_native_ppo_runtime.py),
[`rlinf_native_ppo.py`](../../src/fluffy_bassoon/meta_rl/rlinf_native_ppo.py),
[`rlinf_robocasa_flow_ppo.py`](../../src/fluffy_bassoon/meta_rl/rlinf_robocasa_flow_ppo.py),
the driver and the config. RLinf itself is only on Baige; checks C1–C2 must
read it there.

### Blockers: these would crash or corrupt a 10M multi-task run

- **F1 action ceiling.** `ActionJournal` and `GlobalActionJournal` default to
  `maximum_actions=1000000` (`rlinf_native_ppo.py:105`, split per rank;
  `rlinf_robocasa_flow_ppo.py:8`). A 10M run aborts near 1M. Pass an explicit
  per-run ceiling (10.2M) from config and record it.
- **F2 OpenDrawer/400/18 hard-coded.** `physics_tick_limit=(400+128)*128`
  (runtime `:46`); case metadata `task='OpenDrawer', horizon=400` (`:98`,
  which would mislabel every other task's history); `episodes.json` written
  and 400 controls asserted only at step 400 (`:118–120`, so a 600 horizon
  never writes outcomes); boundary every 6 (`:265`); `global_step == 18` and
  after-eval at 18 (`:314–321`). The driver asserts
  `max_epochs == 18`, 32/64 envs and world size 2/8. Read all of these from the
  config, with one config per task.
- **F3 seed formula.** `offset = iteration·128 + round·32` (`:78`) assumes
  32 envs × 4 rounds. Any other execution layout repeats or skips cases. Use
  `iteration·E·R + round·E + rank·n + i` with E envs, R rounds, and a CPU test
  that TRAIN seeds over 130 iterations are unique and disjoint from eval.
- **F4 optimizer-step assertion.** `FlowActor` asserts 16 steps per update
  (`:156`); H600 gives 24. Derive the value from the batch arithmetic.
- **F7 storage.** The prior run's PFS free space fell 805→717 GB over ~1M
  controls. That is not an exclusive measurement, but it points to ~0.9 TB per
  10M task and ~4.4 TB for five, against ~843 GiB PFS free after tiering.
  The same images are stored in three places: env chunk records, sampling
  records with full flow chains, and actor batches. Q1 measures bytes per
  component. Before launch, choose either to keep env records plus per-sample
  scalars and drop chains/images from the other two, or to keep everything and
  move each completed 10-iteration block to BOS during the run. Target ≤150 GB
  per task on PFS.
- **F8 wall clock.** `TASK_WALL_SECONDS=28200` (7.8 h) is shorter than ~15 h.
  Either request a longer allocation or run qualified resumable segments. The
  resume path asserts a 64-case before reference and copies its canonical eval;
  test it on CPU for the new configs.

### Needed for the corpus

- **F5** Save the post-chunk observation for the last chunk of each episode, or
  record that the horizon is terminal for value targets. Value-based methods
  need one or the other.
- **F6** Write a global episode ID `(task, iteration, round, rank, env)` into
  env, sampling and actor records. The prior full-corpus join is still pending
  ([history audit](../robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/HISTORY-AUDIT.md)).

### Checks that must pass on GPU in Q1

- **C1 log-ratio at the first micro-batch.** Iteration 1 of the prior run had
  clip fraction 0.238, and approx KL reached 0.13 at lr 5e-6.
  `recompute_logprobs=false`, so before any SGD the actor's log-prob must equal
  the rollout's old log-prob on the same samples (|log ratio| ≤ 1e-3). A
  mismatch is a bug and blocks launch.
- **C2 log-prob dimensions (settled, see Decisions).** Read the RLinf source to see whether the
  chunk-level log-prob sums the 50×32 padded model dims or the 50×12 env dims.
  Padded dims never reach the robot. If they enter the ratio, it gets noise
  with no reward signal. If so, report back; any change is a recipe change and
  needs the human.
- **C3** VLM hash unchanged and losses finite; 24 optimizer steps; 128
  episodes and 1,536 samples; peak memory and seconds per iteration measured.

### Kept as is, with the reason recorded

- Episodes keep stepping to the horizon after first success. That cost 23.6%
  of controls before and will cost more at 600. It is the qualified
  accounting path; changing it needs a new qualification.
- Before-panel admission 5–70% stays a hard stop.
- The eval panel is never used to select a checkpoint. Only the final teacher
  is used for DPT labels.

## Stages

1. **Q0 code (CPU).** F1–F8 on `main` and core tests on allocated Linux:
   seed uniqueness, journal ceiling, horizon-generic episode records,
   step-count derivation, resume. Write one config per task. An independent
   reviewer reads the diff.
2. **Q1 per task, on its 8 GPUs.** GPU probe in the job step; 64-case before
   panel (admission 5–70%); one training iteration with C1–C3; bytes per
   component. Use the 16 envs/GPU default; if memory or CPU fails, fall back
   to 64 × 2 (same science).
3. **Q2 audit.** Read the Q1 artifacts, fix the storage policy, then launch.
   Each Q1 iteration counts as iteration 1 of its run; there is never an extra
   iteration.
4. **Full runs.** All admitted tasks in parallel to 130 iterations. Monitor
   every 20 minutes. Stop only for faults: non-finite loss, VLM changed, C1
   regression, journal mismatch, or PFS below 200 GB. There is no scientific
   early stop.
5. **Corpus close-out.** Per-task export, a full join audit (F6), byte and hash
   manifests, then BOS tiering. Meta-learners start only after all tasks close.
   They get their own plan.

## Handoff checklist for the Baige owner

Code pin: latest `main` at or after `9b89b7d2`; record the exact commit.

1. **Q0, Linux CPU:** run the RoboCasa/RLinf core tests in the native
   environment (`tests/test_robocasa_flow_ppo.py`, `test_rlinf_native_ppo.py`,
   `test_robocasa_pairing.py`, `test_robocasa_native_clock.py`). Resolve the
   10M config with `validate_cfg` for each ID task.
2. **Launch command:** write `COMMAND.sh` in this folder, adapted from the
   [prior run's](../robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/COMMAND.sh)
   (same model/norm hashes, probe, telemetry). It needs a fresh
   output and accounting root per task and attempt, `TASK_GPUS=8`, the
   `ROBOCASA_FLOW_TASK` / seed variables from `flow_seeds`, and
   `ROBOCASA_SIM_PPO_ROLLOUT_PLACEMENT=0-7`. Before the Q1 segment, write
   `CHECKPOINT_AND_EXIT` in the output directory. The old
   `nominal_policy_controls`/1M manifest fields must come from `flow-contract.json`.
3. **Q1, one task first (OpenDrawer), then the other four in parallel:**
   check C1 (first-micro-batch log-ratio), C3, bytes per component, the
   byte-equality of actor images/chains with env/sampling records, seconds
   per iteration and peak memory at 16 envs/GPU.
4. **Resume reference:** build the before-panel reference JSON (`completed`,
   `cases`, `checkpoint` = resume dir, `metrics`, `episode_refs` with SHA-256,
   `canonical_directory`) as in the prior run's `BEFORE-RECOVERY.json`. Then
   resume with `ROBOCASA_FLOW_ACTOR_BATCH_DROP` set to the four fields in
   Decisions and `TASK_WALL_SECONDS=86400`.
5. Monitor every 20 minutes; record in this folder and the coordination row.

## Readout per task

- Log per iteration: TRAIN first success and success at end (128 episodes),
  approx KL, clip fraction, explained variance, value loss, grad norm, optimizer
  steps, seconds.
- **KL to the original policy, offline.** From the saved checkpoints, compute
  the mean log-ratio to iteration 0 on one fixed batch of saved samples. This
  is reporting only and needs no learner change.
- **Gate:** TRAIN first success over iterations 121–130 beats 1–10 by
  ≥10 points (1,280 vs 1,280 episodes). The paired 64-case after panel must
  also beat the before panel (exact McNemar, p<.05). Report success at end as
  well. A task that fails either part keeps its corpus as reference data but
  is not an admitted teacher.
- Optional, after the run: evaluate checkpoints 10…130 on the same 64 cases
  offline, to show the curve without changing the learner.

## Open decisions

None for launch. The five tasks need roughly 20 GPU-h of Q1 plus
~300–600 GPU-h of production, within the human's 8-GPUs-per-task, 10M scope.
