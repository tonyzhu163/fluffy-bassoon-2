# Concrete next step toward five improving source policies

Prepared2026-11-05, after the audited512-case four-epoch result. **The user approved the70GPU-hour ceiling and the endpoint gate on2026-11-05.**
See SOURCE5-BUDGET.json. Launch preparation is checking checkpoint storage first.

OpenDrawer and CloseToasterOvenDoor already have independently confirmed
policy130 improvements. OpenDrawer's separate action-matched experiment now
also supports four PPO epochs over one:355vs315/512, paired+7.81points,
Holm-adjusted p=.000775. Those are two distinct source tasks, not three.
CoffeeSetupMug is currently exploratory; Counter-to-Cabinet and Faucet have
qualified native headroom but no trained Flow-SDE source.

## Proposed bounded source campaign

Train only PickPlaceCounterToCabinet and TurnOnSinkFaucet, each from the
original Human300 SFT actor. Use the supported four-epoch VLM-critic recipe,
40updates ×128episodes ×600controls =3,072,000 TRAIN controls per task.
Keep H50/ten flow steps, noise.3, ignore_last=true, first-success simulator
rewards, frozen VLM, trainable action expert, detached VLM-feature critic,
global minibatch256, microbatch16, two actor ranks and eight rollout/env ranks.
Actor/value rates stay5e-6/1e-4; no KL penalty/stopping or optimizer warmup.
Each actor takes960optimizer steps. Keep every chronological source chunk,
policy version and full resumable boundaries. No extension to130updates is
included in this proposal.

DEV64 at0/10/20/30/40 shows the trajectory. Primary source confirmation uses
the fixed final40 checkpoint, chosen before TEST. Do not select a DEV peak.
The existing Coffee policy130 is also fixed before its new endpoint test.
Evaluate all three against their original SFT actors on512new paired cases
per task, common draws and fixed B8. These are new endpoint panels; the stopped
selected-checkpoint sweep is not resumed or relabeled complete.

Proposed new-task TRAIN/DEV seeds preserve the registered ranges:
Counter-to-Cabinet2,200,000/620,100; Faucet2,800,000/680,100.
Fresh TEST starts:Counter73,220,000; Coffee73,250,000; Faucet73,280,000,
each512cases, disjoint from current evaluation/corpus seeds.

## Proposed source-readiness decision

For Coffee and each new task, require paired wins>losses and two-sided exact
McNemar Holm-adjusted p<.05 across these three prospectively declared tests.
Keep the independently confirmed OpenDrawer130/Toaster130 sources and all
historical failed gates/partial tests visible. This expressly replaces the
old prerequisite that all three retained sources pass the stopped five-task
DEV-selected TEST family. No scientific claim is made that that old gate passed.
If Coffee fails, report it and revise its source separately; do not declare
five effective tasks or automatically add another training arm.

## Incremental compute ceiling requiring approval

| Work | Maximum reservation |
| --- | ---: |
| Counter-to-Cabinet40,8GPUs ×225minutes |30GPU-hours |
| Faucet40,8GPUs ×225minutes |30GPU-hours |
| Three paired TEST512 panels, preparation and device checks |8GPU-hours |
| Diagnosed setup/release contingency; no automatic retry |2GPU-hours |
| **Total** |**70GPU-hours** |

The observed OpenDrawer four-epoch run cost23.69GPU-hours after startup,
so these per-task ceilings retain timing margin. Recheck whole-node capacity,
quota, actual hardware and PFS before submission;21free fragmented GPUs do
not imply two available8GPU nodes. Launch independently when each verified
allocation is ready. This is a separate budget from both the96GPU-hour current
meta campaign and the closed92GPU-hour PPO ablation.

Storage must retain a200GiB shared floor. Serialize if both full source
reservations cannot fit; move only closed, hash-verified owned outputs to the
existing private cold store, keeping active datasets/checkpoints available.
Do not move runtimes, model caches or other owners' data.

Once five sources meet the adopted decision, bind a new five-task corpus and
fit one joint frozen-base KV meta-learner per objective, context384, followed
by ID then the locked15-task one-to-five-context panel. That later fit/readout
needs its own measured reservation; this70GPU-hour request does not silently
authorize an unbounded matrix or override the current96GPU-hour ceiling.

The completed native panels now provide a concrete inference-only planning
estimate for the later50-case matrix: about238GPU-hours for four KV methods
with full/reset controls plus the original frozen anchor, before fitting or
long-task timing margin. An optional second native reference raises that to
about264GPU-hours. See TRANSFER-PLAN.md and FUTURE-READOUT-FORECAST.json.
These estimates do not expand this70GPU-hour source request.
