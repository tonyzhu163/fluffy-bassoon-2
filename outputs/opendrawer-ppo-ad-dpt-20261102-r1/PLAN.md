# OpenDrawer AD/DPT trained to convergence

Human authorization,2026-11-02: train AD and DPT to convergence and compare
them against initial and final PPO on the same held-out cases. This
supersedes the provisional500-update/6GPU-hour/200k-action proposal.
This is a simulator-reward development experiment, separate from the other
owner's active Flora PPO. No new PPO training or source collection.

## Data

The immutable S8-r2 source contains384 episodes,six behavior versions,
5760 decisions,2088 features and six pre-tanh residual controls. Its hashes
and the original/final PPO hashes are pinned in [assembly.json](assembly.json).
Group64 collection slots across six source versions (different scenes,not
same-scene retries). Eight VAL slots are selected by PCG64 seed90211 before
windows:56 TRAIN streams/336 episodes/5040 positions;8 VAL streams/48 episodes/
720 positions. Source PPO saw VAL states; this is held-out student imitation,
not independent teacher training. Source successes35,37,26,32,34,33 per64
show no clear improvement. Retain every episode and report that limitation.

AD labels are actual u. DPT-inspired labels are the pinned final PPO actor's
deterministic means at the original features,with original normalization.
Every non-target context array is bitwise identical. These are candidate
teacher labels,not asserted optimal actions. Rewards stay actual simulator
terminal success. Preserve all failures and chronological version identities.

## Fit and convergence

AD/full,AD/static,DPT/full,DPT/static use the existing plain causal decoder:
seed0,width128,two layers,four heads,context128,window64,batch8,AdamW LR3e-4,
Gaussian NLL,TRAIN-only normalization. Match initialization,sampling,optimizer
and checkpoint-selection rules; report differing convergence endpoints/costs.
No architecture or hyperparameter sweep.

Validate on16 fixed batches every100 updates. Require at least1000 updates,
then10 checks without a material decrease in validation NLL. Material means
more than0.001*max(1,abs(reference NLL)); update the reference only on material
improvement. Save the candidate at ANY lower validation NLL. Select that best
validation checkpoint. There is no fixed total-update or GPU-hour cutoff
that declares convergence. Only the saved validation plateau admits evaluation.

An invocation checkpoints after at most12000 seconds,an operational boundary.
Resume identical weights,Adam,sampler,Python/NumPy/Torch/CUDA RNG and plateau
state under the same source/data/fit contract. Scheduler limits trigger
checkpoint/resume; diagnose failures before another fresh allocation attempt.

## Physical comparison

50 prospectively fixed fresh constructor seeds880110000..880110049 from the
unchanged native target split,paired per-chunk motor draws,full H750 episodes.
Compare initial PPO,final PPO,and all four students with deterministic means,
identical frozen motor/norm/residual bounds. Primary cases begin with empty
student memory; full retains history within the episode. Require exact initial
public features and base chunk hashes across all six policies. Evaluation
cannot select checkpoints,change cases,or tune hyperparameters.

Report success counts,paired wins/losses and exact paired tests against BOTH
PPO baselines,plus full/static differences and source/deployment costs. One
task,one source seed and50 cases remain preliminary.

Secondary check:first eight cases,each full student gets one actual support
on the same reset,then a separate query with the primary query's motor draws.
Install actual terminal simulator reward before query. Report support and
query success separately. Matched context-reset query is exactly primary K0:
same weights,empty memory,episode-start token,reset,noise,no episode-index
input. Reuse those results,charging support to both interpretations. Source
histories did not train same-scene retries; persistence alone proves no ICRL.

## Execution

One Baige container,two GPUs,32 CPUs,128GiB RAM,16GiB shared memory. Each GPU
fits one objective's two modes,then25 primary and four support/query cases.
Scheduler chunks at most479 minutes; resume if more is needed. Aggregate
compute is reported,not cut at the abandoned six-hour estimate.

The fixed cohort contains332 physical episodes at most,with conservative
332*(750+128)=291496 controls including constructor controls. This is cohort
accounting,not a fitting limit. Verify at least32GiB free output space and a
write probe; record quota as unavailable if no quota query exists. Preserve
source data/checkpoints. Probe actual allocated GPUs inside the job. First
retained initializer episode verifies zero-residual parity on all15 chunks
before each executes. Keep records,three-view clips,journals and recoveries.
Monitor every20 minutes; report cluster,job,progress,last check,next action.
