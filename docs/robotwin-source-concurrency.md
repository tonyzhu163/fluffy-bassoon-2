# Concurrent six-lineage source collection

`python -m fluffy_bassoon.meta_rl.robotwin_source_parallel` is a prospective
execution format. The legacy `robotwin_steering_source` entry remains serial
and unchanged. Both require a scientifically eligible frontend under the
existing source-selection contract. Scheduling or capacity qualification
cannot make the currently ineligible frontend eligible.

The new entry runs all six original `sac_calibration_{minus,plus}_{train0,train1,validation2}`
learners in spawned processes, with a qualified `max_concurrency` from2 through6.
If capacity is2, for example, each of three waves has two independent active
learners. There is no automatic one-worker fallback. A slot is reused only
after its process exits, so the previous process's model memory is released.
Each worker constructs its own actor, critics, targets, temperature, optimizer,
replay, sampler and private/global RNG streams. Learner seeds and complete
episode specs come from the existing `lineage_specs`/`episode_spec` functions.

The original initial24/max48, H32/native400, batch32, strict-after32-valid-chunk
warmup, UTD4 and complete-episode update boundary remain unchanged. This first
entry executes initial24 only; upper48 requires a separate prospective
admission. Validation is still the original two fixed cases at0/12/24, with
later cases paired to the actual baseline instruction/reset. Training or
validation failures remain in each worker's chronology and full recoveries.

## One action ledger

Only the parent opens `CampaignBudget`; it retains the canonical lifetime lock
and original phase/method caps. Workers use separate local multiprocessing
Pipes. There is no network listener, per-worker campaign, uncharged allowance,
second physical loop or new action-counting convention.

`SourceBudgetBroker` admits a complete source episode400 or the complete
two-case validation panel800 atomically before reset. Prospective holds
subtract from total, phase and method capacity while canonical pending grants
replace the corresponding part of the hold. Resolved grants retain actual
charges and restore only the unexecuted part of the hold. A proven complete
receipt releases the remaining unissued capacity. The initial per-lineage
execution quotas are9600 source actions and2400 validation actions, within
the original method/phase/campaign caps; they are not new allowances.

The broker's separate SHA-chained append-only ownership journal records
admission and chunk ownership before the canonical grant, and records the
grant before replying. Canonical decision IDs include parent run ID, worker
method, original episode ID and chunk ordinal. Repeated or nonsequential IDs
never execute. Ledger mutation and ownership-journal mutation are not a
cross-file transaction: every interruption between them is deliberately
conservative. An intent with unknown status, an uncertain canonical charge,
or a dead process with an active group blocks fresh episode admission.
Healthy workers can finish their already admitted episodes.

The original complete-episode gate sees global charges unchanged. Its
worker-facing unresolved-intent view hides only a foreign **pending** intent
whose exact durable ownership and active group are known and whose process
the parent observes alive. Own pending, unknown and uncertain intents remain
visible. Parent recovery/final manifests always retain the unfiltered ledger.

Recovering an existing broker journal requires the explicit `recover=True`
audit path with its original ledger/run/code contract. Recovery marks all
previous processes dead, retains holds and charges, and authorizes no replay.
Parallel-v1 has no automatic resume CLI. Physical partial episodes and partial
optimizer cycles require the owner's explicit reconciliation/migration; this
entry never clears them to obtain a new reset.

## Canonical shards and data gates

The immutable `frozen_motus_source_shards_v1` `shards.json` phase index binds
exactly six roots at `shards/<original-method>`. Each root has exactly one
canonical `RobotwinCorpusWriter`. No process appends to another writer's
mutable manifests or statistics. Every worker has its own actors, rolling
recoveries, immutable initial/milestone/prefix/final recoveries, episode
receipts, validation receipts, cleanup and lineage-state output.

The parent saves `parallel-source-recovery.json` before optional prefix
assembly, preserving full recovery references and the unfiltered action tip
even if export fails. The existing prefix builder already accepts a different
canonical root per whole lineage. The parent assembles balanced12/24 only
from six immutable publications with successful frozen-motor/cleanup checks;
it checks the exact expected shard root for every episode. No exporter,
TRAIN/VAL assignment, window boundary, AD label or DPT label is changed.
DPT still needs a qualified same-mapping TRAIN1 teacher; AD improvement and
ICRL admission remain separate empirical gates. Synthetic CPU fixtures admit
no scientific data. Condition-cross remains a subsequent owner-controlled
phase under the existing canonical action ledger.

## Prospective allocation and qualification

The CLI uses the existing bound file arguments, with a new allocation schema
`frozen_motus_parallel_source_allocation_v1`. Bind the original scope/profile,
source selection, interface, motor, joint conventions, control qualification,
student profiles12/24, original `action_ledger`, one fresh job output and one
fresh canonical phase shard root. Retain original historical cost fields.
The explicit new `authorization_basis` is `human_each_job_under_8h`;
`reserved_gpu_seconds` is this single parent job's integer bound strictly
below28800. The old scope's cumulative8h field is historical, not a newly
invented aggregate human limit. Existing action/storage/CPU caps are preserved.

Add `max_concurrency`2..6, `episode_target:24`, exact `worker_method_limits`
from `method_limits(profile)`, `worker_devices` mapping all six methods to
`cuda:0`, positive `worker_cpu_threads`, and `allocated_cpu_count` covering all
concurrent workers. No GPU visibility variable or device namespace is changed.
The current backend and renderer still require one actual scheduler-visible
GPU. Multi-GPU placement requires separate backend qualification.

Bind `CPU_qualification_receipt` and `capacity_qualification_receipt`. The
capacity format is `frozen_motus_parallel_source_capacity_v1`, status passed,
and must bind exact source commit, hostname, GPU UUID/device, concurrency,
CPU/thread counts, mapping/interface/motor identity, frozen tensor hash,
H32/native400, `max_full_recovery_bytes` and `max_chunk_output_bytes`.
It contains exactly concurrency distinct process/method complete episode
receipts, each with canonical manifest, ordered `decision_ids`, positive actual
actions, frozen-before/after and renderer reference. The gate independently
checks raw complete chunk identities, actual overlapping physical intervals,
one renderer PCI, and resolved matching charges in the SAME campaign. A
closed `scheduler_cost_receipt` records all actual qualification GPU seconds.
Qualification runs and their cost are separate paid owner experiments.

The source allocation reserves at least `(60+max_concurrency)` full-recovery
versions plus concurrent raw-write headroom within the original64GiB outputs.
Each worker checks shared headroom before writes, verifies actual full-state
size, and audits each reset/chunk's isolated raw-write growth against its
qualified bound. The actual allocated GPU probe is still mandatory before
launch. No capacity or source experiment is authorized by this documentation.

Prepared qualification, to run on allocated Linux CPU, not macOS:

```sh
uv run pytest -q tests/test_robotwin_source_broker.py tests/test_robotwin_source_parallel.py
```

Only these14 new prepared cases are in the qualification target. Thirteen
legacy dependencies and fixture helpers are byte-identical to the author base
and reviewed integration main; their previous validation carries, and their
old suites were not rerun for this integration.

The new cases include real spawned CPU process overlap and waves, crash-boundary
accounting, unknown/uncertain refusal, per-phase/method caps, fixed validation,
full state retention, and64 actual tiny-network optimizer cycles after the
original strict32-chunk warmup. Runtime capacity/model/simulator qualification
is still pending and cannot run until the exact selected frontend is eligible.
