# Native Human300 atomic reproduction

**Completed and independently audited: 363/900 successes (40.33%), versus the published39.6%.** The descriptive Wilson95% interval is37.18–43.57%. This passes the prospective ±5-point practical screen; it is an aggregate reproduction rather than exact episode/RNG replay.

All18 Atomic-Seen tasks have50 complete episodes, pretrain scenes/objects and seed7. The original JAX step75000 checkpoint uses native H50 prediction/C5 execution/K10 ODE, three224-pixel cameras and bundled mean/std normalization. No PyTorch conversion or training was used. All900 input hashes and selected video hashes passed; all three final sampler counts reconcile to96,003 predictions and479,303 policy controls. All server exits are0 and training updates0.

| Cutoff | Successes | Rate |
| --- | ---: | ---: |
| Released evaluator allowance: old registry×1.5 / new registry×1 |363/900|40.33%|
| Original registry cutoff, from the same trajectories |249/900|27.67%|

The extra allowance accounts for114 additional successes (12.67 percentage points). The maintainer independently confirms C5. Narrow1.0.1 source29f7ce8 changes horizons/version metadata, with matching effective horizons under the original evaluator. The source and asset audits therefore support keeping1.0.1 for the composite comparison.

| Task | Success /50 | Old-cutoff success /50 |
| --- | ---: | ---: |
|CloseBlenderLid|1|0|
|CloseFridge|27|13|
|CloseToasterOvenDoor|5|0|
|CoffeeSetupMug|11|1|
|NavigateKitchen|0|0|
|OpenCabinet|22|17|
|OpenDrawer|34|31|
|OpenStandMixerHead|32|24|
|PickPlaceCounterToCabinet|35|29|
|PickPlaceCounterToStove|29|20|
|PickPlaceDrawerToCounter|24|13|
|PickPlaceSinkToCounter|44|44|
|PickPlaceToasterToCounter|11|0|
|SlideDishwasherRack|40|38|
|TurnOffStove|7|3|
|TurnOnElectricKettle|10|5|
|TurnOnMicrowave|3|2|
|TurnOnSinkFaucet|28|9|

Baige Pro6K job`job-zghi39i291dx`: created13:43:53, all three pods created13:48:59, scheduler Succeeded at15:04:35 Shanghai on2026-11-09. Conservative creation-to-release charge14,526 GPU-seconds; all qualification attempts plus this stage total15,434 seconds=4.28722 GPU-hours. Active execution source329fcded121cb1982289bd6fadaf7d07d033a513 remained immutable; independent CPU audit usedf0f6ea6c.

This qualifies the native atomic evaluation path. Composite-Seen/Unseen remain unmeasured here. The existing PPO/GRPO target-scene/H600/C50 Flow-SDE protocols and the converted RLinf inference path require separate validation; this reproduction does not certify their training logic.

[Independent audit](ATOMIC-AUDIT.json) · [Gate](ATOMIC-GATE.json) · [Research and upstream links](RESEARCH.md) · [Scheduler proof](ATOMIC-SCHEDULER.jpg)
