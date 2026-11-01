# First real native AD/DPT fit: prepared, not launched

The sole implementation owner is exporter/trainer thread `01a18acc`. This
artifact only invokes its qualified code at exact commit
`1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; it does not implement another trainer.
Root must review the resource/scope and submit the actual allocation.

The previous CPU qualification passed: job `job-tmocet3s5ur4`, exit0,
46 focused tests in9.30seconds, the same immutable bank re-exports, four tiny
2-update fits and frozen-wrapper checks. It used0GPU/110scheduler seconds.
Those results qualify interfaces, not the real100-update configuration below.

## One bounded production fit allocation

Request **one GPU,8CPU,32GiB RAM,8GiB shared memory,20minutes**. Use the existing
Python environment, exact fresh detached worktree at the qualified pin, and
fresh runtime output `robotwin-native-firstfit-20261030-r1`. The actual
allocation/CUDA probe must pass before training. There are no VLA loads,
simulator calls, new source actions, target interactions or deployment SGD.

Run four independent **sequential processes** on that one GPU:

| Objective | Mode | Seed | Updates | Milestones |
| --- | --- | ---: | ---: | --- |
|AD|full|0|100|25,100|
|AD|static|0|100|25,100|
|DPT|full|0|100|25,100|
|DPT|static|0|100|25,100|

All retain width128/layers2/heads4, memory capacity300, maximum training
window128, batch8, AdamW3e-4, weight decay0 and gradient clipping1. The Gaussian
physical-noise NLL and bound1.5 deployment family are unchanged. Native input
is3090=2062public features+1024previous noise+4feedback/clocks; output1024.
Validation uses4fixed sampled batches at updates1/25/50/75/100 and any bounded
terminal update. Normalization is fitted on source training only.

Sequential execution preserves batch8 and gives measured process timing and
recovery evidence. We have no throughput measurement showing that packing four
long-window models improves this runtime. Separate processes reclaim allocator
state, isolate model/optimizer/RNG state, and avoid changing the global batch
or weighting to fit concurrent processes. Packing can be reviewed after these
measurements; no claimed speedup is attached to this first fit.

## Data and provenance

Training index:
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-history-20261030-r1/train/index.json`

SHA256:`4e8d4e64eb1845a234f710342f168d3a4cdb86366d64dd6c7cef6c7c5895a4af`.

Validation index:
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-history-20261030-r1/validation/index.json`

SHA256:`194ae0fd4588e7bd55c3cd7a0fc6eedfb010bcb8043e311eeedd8adff0b2472b`.

Both use SAC5 DPT teacher SHA256
`f2fb39571cc3d0e809dfdc454f7f6df7f48e585fc972fd4c2fb8c83d36e63a42`.
The validation bank is **PPO10 acting behavior**,24complete source-validation
episodes,10,552selected validation actions, anchored to a93,851-action source
snapshot. It is not SAC10 acting data. Explicit BC/DPT selection, source-only
membership, compatible native interface and disjoint whole cases passed
qualification. AD validates against BC; DPT uses the same pinned SAC5 teacher
for train and validation labels. SAC5 remains a source-trained candidate,
not a certified expert.

All four models reuse these same unique source interactions. Repeated windows,
fit positions and100optimizer updates are training computation, not new robot
data. Keep unique source/deployment costs separate from offline fit compute.

## Time bounds and recovery

Each trainer retains a240-second fitting-clock bound and100-update endpoint.
That clock starts after loading, so the launcher separately applies an
approximately970-second **global** budget covering probe, loading, all fits and
artifact auditing, with a70-second final audit reserve. A990-second TERM plus10-second KILL emergency
outer timeout remains inside the20-minute scheduler ceiling.

A model process is bounded by its remaining global allowance and at most270
seconds, leaving loading/exit time around the240-second fitting clock. If the
wall budget binds, preserve its last **complete committed** `latest.pt` and any
uncommitted partial files. A timeout can lose unsaved optimizer steps after the
last checkpoint; report committed counters, not presumed executed updates.
Models that cannot start within the remaining allowance are explicitly marked
not started. `exit0` can mean a controlled partial fit: only `completed_all_four`
and four audited100-update recoveries establish completion of this allocation.

The launcher freezes copied dataset indexes, per-model commands/logs/metrics,
best-validation and milestone portable exports, complete policy/AdamW/sampler/
Python/NumPy/Torch/CUDA RNG recovery, checksum/CRC/counter audits and every
artifact's SHA256. The candidate best checkpoint and final milestone checkpoint
have different selection meanings and retain their own identities.

A future resumption must load the exact `latest.pt` SHA with the same100-update
contract, configuration, data indexes, mode,seed and240-second invocation bound.
Do not restart completed models or enlarge this allocation automatically. The
fresh-output `job.sh` deliberately does not resume an existing entire job; root
can issue a separately reviewed allocation for remaining models/recoveries.

## Deployment and command

Root creates a clean detached worktree from Git at the exact qualified pin.
Transfer this reviewed launch artifact to the project's `inputs/` directory,
not into that candidate source worktree. Preserve its own SHA256 receipt.
Do not precreate the runtime output. Then, **inside the actual allocation**:

```bash
bash /mowenjie/fluffy-bassoon-2/inputs/robotwin-native-firstfit-20261030-r1/job.sh \
  /mowenjie/fluffy-bassoon-2/worktrees/robotwin-native-firstfit-20261030-r1 \
  /mowenjie/fluffy-bassoon-2/outputs/robotwin-native-firstfit-20261030-r1
```

No live source job, corpus, teacher checkpoint, source-action ledger or
controller contract is edited. Source production failure/recovery remains with
its existing owners. This is fitting evidence only: finite loss, improving NLL
or a portable export does not demonstrate robot success, recovery or in-context
reinforcement learning. Memory capacity300 also does not mean gradients were
trained across300decisions; windows remain at most128.
