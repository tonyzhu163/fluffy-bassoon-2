# Sidney RLinf integration and PPO pilot

User selected Sidney and asked to continue toward PPO on2026-11-05, then explicitly confirmed broad multi-task SFT with no RL tuning as the intended clean checkpoint. Keep Sidney; do not switch to task-specific RLinf adjust_bottle SFT. Use
Sidney's published SFT scores as the baseline reference; no broad SFT rerun.
The earlier local screen is not admitted because reset instructions described
stale scene metadata. Prior scheduler usage:2561GPU-seconds(0.7114GPUh).

Fix reset language using the actual constructed task.get_info() and protect
changed-seed/unstable-retry behavior with core environment tests. Keep external
RLinf c605b9d and RoboTwin dca9ec6 checkouts immutable. Record actual object IDs
and effective instructions at reset. Never replace a selected bank seed silently.

Convert pinned Sidney e49e2ab6 with RLinf's released key/layout conversion,
overriding its tied-embedding assumption with Sidney's actual input embedding.
Retain saved mean/std,14 absolute joints and native Hugging Face tokenization
through a named checkpoint-specific model-boundary adapter. No Motus delta,
calibration or quantile transforms. Native H50/K10 first; compare saved pixels,
state and token IDs plus same-noise actions in native/RLinf. Check FP32 reference
arithmetic (decoded maxabs<=5e-4, normalized maxabs<=1e-3) and report BF16 drift
separately. These are integration checks, not SFT benchmark estimates.

One1GPU/30min bridge allocation runs the core reset tests, conversion, three
aligned source-observation replays and one RLinf episode. A full-FP32 comparison
is diagnostic only, invoked if mixed-BF16 outputs need numerical localization;
training retains Sidney's native mixed BF16 with full-FP32 vision and norms.
Then migrate explicitly to the
VLRLA K5/.3 Flow-SDE sampler for a two-iteration action-expert PPO pilot on the
source task place_phone_stand, simulator reward, frozen VLM. Reuse the native
RLinf learner/recovery wrappers; keep global batch2048, microbatch32, five
epochs,256envs x4rounds and H400. Verify rollout/actor replay before SGD,
measure capacity/time, retain complete optimizer/sampler/RNG boundaries, and
check advancing updates. This establishes execution, not convergence. No
held-out-task selection. Scheduler resource limits and actual GPU type are
checked before launch. Budget up to4 additional GPUh for bridge+pilot, within
the repository-authorized50GPUh task scope; record all attempts and migrations.

Execution migration: the eight-GPU group was not schedulable. Use four ranks
with microbatch32 and accumulation16, retaining global2048,256 environments
and five epochs. A50-minute four-GPU allocation replaces25 minutes on eight.
The model bridge is complete; train01 failed before model initialization on
Hydra inheritance. The retry uses the same recipe as a standalone primary
configuration. Completed scheduler charge through train01:3761GPU-seconds
including the earlier qualification;920 bridge plus280 failed-PPO seconds
are part of the current additional envelope. No SGD has run at submission.

Measured migration after the first live source chunks:256-scene reset costs
about7 minutes and a50-control vector chunk about40 seconds on four GPUs.
The50-minute allocation cannot reliably complete two full iterations. Preserve
the complete step0 native actor/Adam/sampler/RNG boundary, stop only this owned
job, and resume it in a fresh output with a3-hour maximum allocation. Keep
the same four ranks, batch, source seeds, sampler and learner. Charge the old
partial prefix in the same action journal under its original attempt; it is
not an extra PPO update or admitted complete source history. Increase the
self-imposed bridge+pilot envelope from4 to14 additional GPUh, including all
attempts; this is within AGENTS.md authorization up to50GPUh for the task.
No user hard cap changes. Preserve all checkpoint/cache inputs.

BF16 execution migration: actual64-environment rollout logprobs replay exactly
at batch64, but batch32/16/8 replays differ by up to0.04197 per scalar.
Keep BF16 and the original1e-3 replay guard. Qualify a full64-example backward
pass, then use actor micro64 and accumulation8 on four ranks; global2048,
five epochs, sampler, loss and optimizer schedule stay unchanged. The earlier
64-to32 continuation was stopped before any SGD. Retain its restored-state
receipt and charge its prefix; resume the complete original step0 boundary.
The next four-GPU allocation is170 minutes so all attempts stay within the
14 additionalGPUh envelope. No optimizer updates have occurred yet.

Matching-batch64 training qualification passes after keeping frozen SigLIP
in eval mode during policy training (PyTorch MHA otherwise switches kernels).
Training-mode logprobs replay exactly, full backward peaks at19.05GB and takes
2.74s, with190 policy and8 critic gradient tensors finite and no frozen VLM
gradients. No SGD in this qualification. Retain native mixed BF16; no tolerance
relaxation or FP32 training fallback.

User steering: maximize feasible6000-series memory use or measured throughput,
not merely the previously qualified micro64. Benchmark native4-GPU FSDP/PPO
at micro64/128/256/512 with global2048. Warm Adam once, time three full updates
per candidate, include raw camera preprocessing and record old-probability
recompute overhead (amortized over five epochs). Synthetic advantages/returns
and scratch updates are performance-only and never become the PPO initializer.
Select fastest qualified memory-safe batch, record capacity and actual update
time, and use native algorithm.recompute_logprobs if batch-dependent BF16
numerics require it. Scientific batch, optimizer schedule and loss remain
native. Extend the self-imposed envelope to18 additionalGPUh to include this
user-requested throughput qualification; stay within AGENTS.md50GPUh scope.
Train05 stopped before SGD; original full step0 boundary retained. Throughput01
hit a state-selector API mismatch before timing; corrected and restarted in
fresh throughput02, aborting any further non-memory harness error immediately.


Host-memory retry migration after train06: Ray killed a worker at488.18/512GiB
(95% host-memory threshold), before any scientific SGD. The four simulator
workers each occupied102–107GiB. Request1024GiB hostRAM on the same four GPUs
and require at least896GiB actual cgroup memory before model loading. Keep
Ray's existing protection; record cgroup memory every10s and validate all768
first-reset RGB views plus finite states before native action collection.
The user accepted micro64/accum8/global2048; five epochs and all scientific
settings stay unchanged. This is the same two-iteration phone-stand pilot.

The18 additionalGPUh and1M charged-action values were our engineering
envelopes, not user hard caps. Explicitly migrate to20 additionalGPUh and
1.2M cumulative charged actions, still inside user AGENTS.md authorization
up to50GPUh for the unchanged task. Preserve every earlier journal file and
charge, including unresolved intents; new attempts may explicitly read the
old1M ceiling while writing the new1.2M ceiling. Never reset the shared ledger.
Train06 charged5876 allocatedGPU-seconds (5944 conservative creation-to-release);
completed additional usage is22993GPU-seconds, cumulative25554 including legacy
qualification. Conservative additional charge23088 leaves48912GPU-seconds
inside the new20h envelope. A4GPU170min retry plus launch margin fits.

User RAM-debug steering holds the production retry. Qualify oneGPU/16 real
scenes without VLA/SGD, with at most7200 physical hold controls and480 setup/reset
allowances, all charged in rank0 of the same original journal. Record
RSS/PSS, cgroup anon/file and glibc live/free arenas around scene construction,
resets, chunks, GC and malloc_trim. Check retained RGB/state bytes remain
identical across allocator diagnostics; preserve native shader/camera settings.
The20additionalGPUh planning envelope includes this≤0.3334GPUh diagnostic.
The explicit1.2M action migration remains sufficient including all old charges
and the unequal rank0 diagnostic plus two new scientific iterations.


The16-scene diagnostic completed: freed native allocations retained in glibc,
rather than growing Python/rollout history, explain the dominant second-reset
increase (27.93GiB RSS to15.13GiB after trim, live13.79GiB unchanged).
Production remains held for `memory02`, a model-free64-scene comparison of
baseline, native batch-end trim and per-scene-close trim. It records reset peaks
every200ms, retained observation/RNG/scene/motor identity, and a matched second
reset plus50-control hold. Maximum9,600 physical controls and2,560 constructor/
reset allowances are charged to originalrank0 under a fresh attempt. Its
30min/1GPU bound remains inside20additionalGPUh.

The earlier proposed1024GiB allocation is provisional and superseded if this
qualification supports512GiB. Prepared retry code requires all paired checks,
unchanged RGB bytes, and a measured per-rank peak bound before model loading.
The conservative full-phase bound is four times measured64-scene peak plus
40GiB environment runtime overhead,80GiB other workers, and64GiB pending
rollout/checkpoint buffers. It must stay below the actual unchanged95% Ray
threshold. Only then enable the opt-in close-time trim and native batch-end
trim in the fresh Sidney run. Active worktrees remain immutable.


Memory02 reached its worker timeout during the final per-scene-mode camera
read. Its closed baseline/batch-end results are retained; no per-scene peak or
paired motor result is admitted. Scheduler charge1,750GPU-seconds (creation to
release1,762), worker1,717;6,400physical controls plus2,560conservative allowances
remain charged. Rank0now80,640. A freshmemory03 runs only the two paired resets,
not the completed baseline. It uses original source seed partition0 but
explicitly charges diagnostic allowance to original ledger rank1. Maximum
6,400physical+1,920setup/reset=8,320; rank1then69,120, leaving room for212,480
freshpilot controls/allowances below its300,000 partition. Runtime40min/1GPU;
conservativecompleted25,274+2,440diagnostic+41,040production=68,754GPU-seconds
within20additionalGPUh. This is a diagnostic continuation, not scientific scope
expansion. Active memory02 worktree and journals are immutable.


Memory03 lowered64-scene reset peak to56.120GiB and had bit-identical initial
RGB/14Dstates, but its post50hold assertion failed before candidate error/reward
was persisted. No production admission follows. Memory04 adds native-native
repeat control, exposed qvel/root/object pose/velocity/contact and bookkeeping
probes, exact actual action hashes/raw outcomes, and per-case differences saved
before every gate. Restore identical Python/NumPy/Torch/CUDA hold RNG as a
model-free diagnostic control; this does not change production RNG behavior.
Keep1e-6state and exact reward/termination/truncation/mask gates unchanged.

Freshmemory04 uses1GPU/64CPU/256GiB/40min and journalrank2, originalsource seed
partition0, at most9,600physical controls+2,560constructor/reset allowances.
Rank2wouldbe72,960plus212,480freshPPO=285,440<300,000. Completedconservative
additional26,602+2,440diagnostic+41,040PPO=70,082GPU-seconds<72,000(20h); keep
all failed/unresolved charges. Scientificupdatesremain0. Never modify prior
active/terminal worktrees or silently relax a tolerance to obtain admission.
