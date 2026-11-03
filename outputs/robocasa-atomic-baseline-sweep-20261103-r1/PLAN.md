# RoboCasa365 atomic_seen native baseline sweep

Proposed 2026-11-03 at the human's request. Not launched; needs a Baige owner.

## Question

Which `atomic_seen` target tasks give the released pi0.5 room to improve
(native first success 5–70%), so the human can pick a representative 5 ID
(PPO) + 5 OOD (held-out) set? The current portfolio
([TASK-PORTFOLIO.json](../robocasa-five-task-flora-ppo-20261102-r1/TASK-PORTFOLIO.json))
has PickPlaceCounterToCabinet at 37/50 (74%, fails the gate; human now leans
toward replacing it) and no pick-and-place task among its five OOD tasks.

RoboCasa365 at the pinned native source `3443cdf3` registers 65 atomic tasks;
18 form `TARGET_TASKS["atomic_seen"]`. The checkpoint's model card publishes no
per-task results.

## Already measured (not re-run)

| Task | Native success | Source |
| --- | ---: | --- |
| OpenDrawer | 29/50 | [five-task qualification](../robocasa-five-task-native-qualification-20261102-r1/) |
| PickPlaceCounterToCabinet | 37/50 | same |
| PickPlaceDrawerToCounter | 23/50 | same |
| TurnOffStove | 7/50 | same |
| CoffeeSetupMug | 17/50 | same |
| CloseFridge | 28/50 | original pi0.5 matching eval50 (coordination row) |

## Sweep

- **Policy:** unchanged released pi0.5,
  `robocasa/robocasa365_checkpoints@c484448a`
  `pi05_pretrain_human300/multitask_learning/75000`; H50, 10 flow steps, three
  views (`robot0_agentview_left`, `robot0_eye_in_hand`,
  `robot0_agentview_right`). No SGD.
- **Runtime:** same native setup and worker path as the five-task
  qualification, on the latest verified `main`. Record the exact commit.
- **Tasks and native registry horizons (20 episodes each):**

| Task | Horizon | Type |
| --- | ---: | --- |
| CloseBlenderLid | 900 | open/close |
| CloseToasterOvenDoor | 450 | open/close |
| OpenCabinet | 1050 | open/close |
| OpenStandMixerHead | 450 | open/close |
| SlideDishwasherRack | 450 | open/close |
| PickPlaceCounterToStove | 600 | pick and place |
| PickPlaceSinkToCounter | 900 | pick and place |
| PickPlaceToasterToCounter | 600 | pick and place |
| TurnOnElectricKettle | 450 | buttons/knobs |
| TurnOnMicrowave | 450 | buttons/knobs |
| TurnOnSinkFaucet | 600 | buttons/knobs |
| NavigateKitchen | 450 | navigation (report; likely excluded) |

- **Seeds:** new disjoint bank, `710100 + 1000 × task_index` (index in table
  order), 20 consecutive seeds per task. Do not reuse the portfolio's
  qualification (1x0100), held-out ID (4x0100) or OOD (5x0100) ranges, or
  the PPO TRAIN range (210100+).
- **Readout:** per task, first success and success at end, with Wilson 95%
  intervals. Mark each task as below 5%, 5–70%, or above 70%.
- **Budget:** at most 147,000 controls (240 full-horizon episodes). Expected
  under one hour on 8 GPUs at the measured batched throughput. No per-learner
  action cap applies (removed 2026-11-03), but charge these to this run.

## After the sweep

This is a coarse screen (±~20 points per task). The human picks 5 ID + 5 OOD,
balanced across open/close, pick-and-place and buttons/knobs; each chosen task
not already at 50 episodes then gets a 50-episode qualification under the
usual headroom rule before PPO.
