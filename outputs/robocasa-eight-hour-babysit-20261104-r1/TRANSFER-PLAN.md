# Frozen-base steering: prospective fifteen-task readout

2026-11-04, fixed before collecting any outcomes for this panel.

First evaluate held-out scenes of the five admitted source tasks (ID), then
evaluate these fifteen transfer tasks. The source portfolio is OpenDrawer,
CloseToasterOvenDoor, CoffeeSetupMug, PickPlaceCounterToCabinet and
TurnOnSinkFaucet, subject to the unresolved source-admission decisions. None
appears below. Task categories and native horizons were read from the actual
RoboCasa registry at `3443cdf3dd20f9ae8ed41adba442059b506891d0` on Baige.
"Seen/unseen" below is the benchmark's category relative to the original
Human300 motor, distinct from source/meta-training exposure.

| Category | Task | Native controls per episode |
| --- | --- | ---: |
| Atomic seen | CloseFridge | 900 |
| Atomic seen | OpenCabinet | 1050 |
| Atomic seen | OpenStandMixerHead | 450 |
| Atomic seen | SlideDishwasherRack | 450 |
| Atomic seen | TurnOnMicrowave | 450 |
| Composite seen | KettleBoiling | 1500 |
| Composite seen | LoadDishwasher | 1800 |
| Composite seen | PrepareCoffee | 1800 |
| Composite seen | RinseSinkBasin | 1350 |
| Composite seen | ScrubCuttingBoard | 1200 |
| Composite unseen | ArrangeTea | 2250 |
| Composite unseen | CuttingToolSelection | 1200 |
| Composite unseen | GatherTableware | 2250 |
| Composite unseen | PanTransfer | 1800 |
| Composite unseen | WashFruitColander | 3150 |

These choices cover doors/appliance controls, boiling/coffee/dishwashing/
cleaning, and novel selection/arrangement/manipulation compositions. They are
a fixed development subset, not a representative random benchmark sample.
CloseFridge and TurnOnMicrowave also belong to the current three-task
exploratory readout. Keep that prior evaluation exposure visible; none of its
outcomes may select the later model or alter this panel.

Use fifty fresh cases per task, six complete episodes per case, common
case/chunk draws and fixed-width, fixed-order waves. Episode 1 is the no-context
anchor. Query episodes 2–6 have one through five actual complete episodes in
context; report their individual first-success and final-success rates, plus
paired uncertainty against the episode-reset control. Report cumulative
best-of-N separately. Frozen original-motor anchors and zero-init bridge
parity protect competence. Deployment updates history only; both the base
policy and learned steering weights remain frozen.

## Context capacity required before retraining

At H50, the longest declared task needs 63 decisions per episode and
**378 tokens for six complete episodes**. The current context192 models cannot
retain that complete history. Use at least context384 for the later fit and
qualify its actual cache, episode boundaries and loss windows. A proposed
complete-episode training window of up to32 source episodes uses at most384
H50 decisions with the H600 source corpus; preserve global batch8 and objective
weights while measuring the execution batch on the allocated GPU. This is a
new immutable fit contract, not a modification of the current context192 run.
Its exact training cost and learned-KV native qualification remain to be
measured and included in a concrete budget before a large allocation.

The unmodified native horizons sum to21,600 controls across these15 tasks.
At50 cases ×6 episodes, one model/control variant requires **6,480,000 policy
controls**, plus at most576,000 reserved reset controls. Full-history and
episode-reset controls double this per learner. Current three-task ID/OOD
evaluation uses its own shorter task contract and the existing96GPU-hour cap;
this larger panel has no invented budget or implicit source admission.
