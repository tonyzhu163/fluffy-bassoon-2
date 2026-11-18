# RoboTwin continuation after the execution pilot

Prepared2026-11-06 during the model-free memory check. This is a CPU-only
preparation plan. It does not launch GRPO, enlarge the active PPO contract, or
admit a full six-task campaign. First finish memory qualification and the
existing two-iteration PPO pilot, with real optimizer advances and recoverable
boundaries. Keep the user's throughput choice64/GPU, accumulation8, global2048.

## Common starting point and runtime

Every independent online method starts from the same broad Sidney SFT,
`SidneyXie/pi05_robotwin@e49e2ab6c11f07511573b67261bd129e88d0a416`.
GRPO must not warm-start from a PPO-updated action expert. Preserve the native
14-joint absolute command interface, three cameras, saved mean/std transforms,
native mixed precision, frozen VLM/vision eval mode, H400/H50/K5/Flow-SDE.3,
simulator first-success reward, masks, and source/held-out split.

Use RLinf`c605b9d55ed93a180119b099e6f836abff293b08` and
RoboTwin`dca9ec682688821c42944eacd7f5dd9bfb15396f` until a separately qualified
migration. Pin the latest reviewed project main in each fresh detached run.
The published Sidney SFT scores remain reference values; no new broad SFT
benchmark panel is required. The current pilot adds no DEV/evaluation episodes.

## GRPO wiring and comparison accounting

The pinned RLinf pi0.5 GRPO example is
`examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml`.
Its actual task, H240, action chunks5, flow steps3, seed42, micro128 and one
update epoch are not the RoboTwin pilot protocol. The pinned RoboTwin GRPO
presets use other VLA families. Neither is a ready Sidney RoboTwin recipe.

Proposed local comparison overrides, to qualify after PPO advances:

| Field | Proposed GRPO | Current engineering PPO pilot |
| --- | --- | --- |
| `algorithm.adv_type` | `grpo` | `gae` |
| `algorithm.loss_type` | `actor` | `actor_critic` |
| `algorithm.group_size` |8 |1 |
| `actor.model.add_value_head` |false |true |
| Environments / rounds / horizon / chunks |256 /4 /400 /50 |same |
| Global batch / micro / accumulation / epochs |2048 /64 /8 /5 |same |
| Actor LR / clipping / entropy / reference KL |5e-6 / native /0 /0 |same |

Group8 is a local design supported by the pinned GRPO implementation, not a
claimed VLRLA Table1 setting. Five epochs deliberately match local optimization
exposure rather than copying the LIBERO example's one epoch. Remove critic
parameters/optimizer/loss, while retaining policy gradients and frozen VLM.
Use a dedicated GRPO entry/validator: the current Sidney entry unconditionally
replays `prev_values` and the current validator requires GAE plus a value head.
The original rollout writer and probability replay are reusable only after
their absent-value and GRPO-mask paths pass allocated numerical qualification.

Native RoboTwin `update_reset_state_ids` repeats each scene ID contiguously
`group_size` times. At256 environments, group8 gives32 distinct scene groups per
round and128 groups per iteration, each with8 independently sampled policy
rollouts. That is1,024 episodes and8,192 nominal chunk slots per iteration,
with409,600 nominal primitive controls. Two iterations imply40 Adam steps at
global2048/five epochs, not two Adam steps. Actual controls come from masks;
setup, failed resets and unresolved intents remain charged separately.

Equal physical budgets alone do not equalize distinct scene coverage. The
current group1 PPO pilot sees up to1,024 scene draws per iteration, whereas
group8 GRPO sees128 groups. Its engineering results are not a case-matched
PPO-versus-GRPO comparison. Before a scientific comparison, lock a common
case/replicate manifest and run a fresh PPO reference with the same32 scenes
times8 replicas per round, still using independent-trajectory GAE. Qualify the
environment grouping separately from the advantage grouping; do not edit the
active PPO pilot to create this reference.

Within each GRPO group require equal actual reset seed, scene metadata,
instruction and initial public state/RGB, plus independent recorded policy RNG
streams. Preserve requested/effective seeds and reject a group if retry
substitutions break case identity. Save group/member IDs and raw outcomes.
Check the native masked group-return/advantage calculation against a small
numeric reference, including constant-return groups. Report zero-spread groups;
an Adam step or weight-decay movement alone does not establish a useful RL
signal. Verify finite policy gradients, unchanged VLM signatures, real native
episodes and complete recovery on the actual allocated GPUs.

## Six-source readiness

[NEXT-PHASE-SEEDS.json](NEXT-PHASE-SEEDS.json) reads the actual pinned sources,
not the later upstream HEAD. All twelve task modules exist. The shipped bank
hashes are55b25866... (TRAIN) and194164f7... (evaluation), with zero integer
TRAIN/evaluation overlap for the covered tasks.

| Source task | TRAIN / evaluation seeds | Remaining qualification |
| --- | ---: | --- |
| `handover_mic` |1000 /150 |actual assets, planner, cameras, motor and runtime |
| `pick_diverse_bottles` |0 /0 |new planner-qualified disjoint banks and runtime |
| `place_a2b_left` |1000 /150 |actual assets, planner, cameras, motor and runtime |
| `place_object_stand` |0 /0 |new planner-qualified disjoint banks and runtime |
| `place_phone_stand` |1000 /150 |PPO pilot in progress; memory fix still gated |
| `scan_object` |0 /0 |new planner-qualified disjoint banks and runtime |

Never substitute `pick_dual_bottles`. Existing file loaders index the task key;
changing only the task name cannot supply missing banks. Qualify missing scenes
on allocated Linux with the same embodiment/randomization, record attempted and
rejected seeds and controls, and bind immutable bank hashes. Freeze source DEV
and final-test cases before selection, using disjoint groups from available
evaluation banks. A candidate allocation of22 DEV plus128 final cases fits each
150-seed source bank; this remains a proposal, not an active evaluation.

Held-out adaptation tasks stay out of source gradients, normalization,
histories and selection. `place_dual_shoes` and`stamp_seal` also lack shipped
banks; the other four held-out tasks have evaluation banks. Held-out task
exposure during the broad SFT must still be disclosed.

## Histories, students and deployment controls

Retain complete chronological source streams, including failures and valid
duration masks, before splitting them into student windows. Preserve producer
policy version, task/case/episode/time identity,14D executed commands, raw RGB,
joint state, prompt, rewards, boundaries and sampler metadata. Whole streams,
not adjacent windows from the same stream, determine training/development
partitions. Keep PPO and GRPO streams distinct if both become data arms.

Fit AD, DPT, DIT and IC-IQL on identical admitted source histories, with the
same frozen Sidney interface and common selection cases. AD predicts executed
behavior; DPT's later immutable query labels must bind a preselected final
source teacher, transforms and query RNG; DIT and IC-IQL retain their declared
reward/return/value objectives. No teacher-query simulator interactions are
needed for saved observations, but accelerator inference costs still count.
PPO changes the expert: its initial Gaussian is not a steering label for the
unchanged SFT motor. Use qualified direct14D/action or differentiable bridge
targets, or explicitly qualify a compatibility mapping before noise steering.

VLRLA-bridge is a separately labeled reconstruction until official artifacts
are verified. Keep its original VLM/expert frozen, preserve causal cross-episode
feedback, and train its KV bridge through the declared flow objective. The
paper's main-table history budget is unspecified; lock our actual trial length
and attempt/control budget before evaluation.

Evaluate source-task ID and six-task adaptation-held-out OOD on locked cases,
with fixed deployment weights. Pair full history with inference reset and a
separately trained episode-reset control; include reward shuffle/zero and
wrong-task history interventions in the admitted contract. Match cases, motor
randomness rules, actions and allowed attempts, and report success by attempt,
cumulative controls, ordinary online-RL curves and all training/label/evaluation
costs. Structural parity checks are distinct from an SFT success-rate panel.

## Budget and storage gates

PPO continuation has an authored20-additional-GPU-hour/1.2M-action engineering
envelope. It is not permission for silent GRPO continuation. After real PPO
update/reset/storage timing is measured, record any new method's envelope and
fresh ledger explicitly, retaining a task-level aggregate of all earlier
diagnostics, failures and controls. No GRPO GPU work before PPO real updates,
fresh capacity/health checks and storage admission. Keep total RoboTwin usage
within the user-authorized50GPU-hours unless the human approves more.

The scratch actor timing26.41s/update and observed~8.3min reset/~40s chunk are
components, not measured scientific throughput. A rough iteration forecast is
four resets plus32 chunks plus20 updates, about63min on4GPUs, before I/O and
boundary overhead. Six tasks times25 iterations would already be about630
GPU-hours on this estimate, before GRPO, students or evaluation. This is not an
admitted campaign or a validated cost quote. Use actual completed-pilot timing
to prepare a concrete staged scope and budget proposal before full collection.
Also measure retained bytes/iteration and concurrent stage-back/archive rates;
the current125GiB pilot reservation cannot fund a whole source corpus.
