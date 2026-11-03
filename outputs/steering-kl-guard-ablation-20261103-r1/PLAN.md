# KL guard diagnostic from the actual first PPO batches

Human requested this test on 2026-11-03 and explicitly instructed: “Hold
recovery until the KL test is reviewed.” The production and coordination
owners were messaged with that hold. The two source runs remain separate,
immutable experiments; this diagnostic never advances or overwrites them.

## Question and paired comparison

Does target_kl=0.02 stop useful first-batch optimization too early, and does
removing that stop improve native success after one complete optimization
epoch? Use both actual source interfaces, velocity32 and independent
all-layer K/V1152. This is a first-update diagnostic, not a convergence result.

For each interface load its actual round-001-before-sgd.pt and immediately
preceding event-003-dev-000.pt. Use the bank's normalized pre-SGD policy,
actual rewards/observations/actions/old Gaussian parameters, and the boundary's
unchanged Adam and sampler state. Do not use the already-trained update-2
policy as an initializer. Compare:

| Candidate | Epoch limit | KL stop | Other settings |
| --- | ---: | ---: | --- |
| guarded1 | 1 | 0.02, full-bank joint Gaussian KL after each minibatch | Original LR, Adam, minibatch32, clipping and GAE |
| unguarded1 | 1 | Disabled; KL remains recorded | Identical to guarded1 |
| guarded10_reference | 10 | Original0.02 stop | Replays the production configuration on CPU |

Restore identical initial model, Adam, minibatch-generator and CPU RNG state
for every candidate. The guarded1 and guarded10 reference should coincide
when stopping during epoch1. CPU versus historical GPU replay is compared
numerically, without claiming bitwise device equivalence. Save independent
candidate weights, configuration and recovery state. Hash source files before
and after. Model computation runs only on allocated Baige Linux resources.

## Readout

Record samples used for gradients, optimizer steps, full/partial epochs,
initial/final exact joint KL and sampled KL, clip fraction, actor/critic
losses, explained variance, mean/std change and wall time. Numerical finiteness
and more processed samples alone do not establish better robot behavior.

Then evaluate initial/guarded1/unguarded1 deterministically with the frozen
native motor on the same fresh diagnostic cases and native-noise schedules.
Use50 new target-split cases per each of the five existing TRAIN tasks,
600physical controls each, synchronized48-lane collection, identical camera
and physical reset checks. Diagnostic seeds are8,900,000+100,000*task_index+
slot, slot0..49; these do not overlap the source TRAIN or existing DEV/TEST
seed banks. Report paired per-task and macro first-success/end-success,
wins/losses and uncertainty. Do not select or inspect the production final
test panel. Retain successes, failures and every physical call.

The native panel has1500episodes/900,000policy controls across the two
interfaces and three policies. Reset/qualification calls are additional and
journaled. Baselines are recollected under the diagnostic execution batch;
old eight-GPU baselines are descriptive only.

## Resource and interpretation boundaries

Use existing allocated development CPUs for optimizer replay, with CUDA hidden
and four Torch threads. Native evaluation has a separate cumulative ceiling
of21,600allocated GPU-seconds (6GPU-hours), at most4GPUs, including startup,
cleanup and any failed attempt. User/repository authority permits incremental
compute below8GPU-hours. Check queue, actual devices and resource fit before
launch; run the required probe inside the actual allocation. No automatic
retry. Stop at a complete case/bank boundary with recoverable artifacts when
the forecast cannot fit. Record actual costs and partial results.

A preference supported by this small first-update diagnostic still needs a
declared prospective training run. Changing the production optimizer contract
requires a fresh output identity and clean scientific initializer; preserve
the old update-2 checkpoints, bank files, baseline panels and accounting.
The human hold remains in force until review.

Execution choice before any GPU launch: use four GPUs, two case shards per
interface, with the same48lanes/GPU and exactly the same1500episode panel.
This shortens elapsed time without changing data volume, policy weights,
pairing, or the6GPU-hour cumulative ceiling. Request84CPU/832GiB/192GiBSHM
on one node; the inspected queue has RPBZZZ6 nodes with175CPU/1800GiB limits.

## Diagnosed launcher correction before the second GPU attempt

R1job-voaelvgv4mqi failed the visibility probe before loading any model or
dispatching any reset/control. Its exclusive scheduler container exposed
exactly four GPUs through NVML but did not set CUDA_VISIBLE_DEVICES. The
launcher omitted the default namespace binding used by the qualified source
launcher. Correct the omission only when NVML exposes exactly four assigned
devices; preserve a nonempty scheduler CUDA setting. This adds no device
namespace and does not change the six candidates or case/noise schedules.

The scheduler pod was created14:46:51 and the job was terminal14:47:40,
49seconds x4=196GPU-seconds conservative upper; process clock was4GPU-seconds.
Charge200GPU-seconds once to the6GPU-hour campaign ceiling. Use freshgpu-r2,
an89-minute scheduler limit, and a process deadline derived from the remaining
budget with180seconds reserved. R1artifacts stay immutable. The successor is
a diagnosed operational repair, not an automatic retry or budget renewal.
