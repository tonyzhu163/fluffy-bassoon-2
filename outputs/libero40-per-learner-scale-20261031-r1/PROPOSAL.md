# LIBERO: a cumulative 1M source-action envelope per learner

Prepared 2026-10-31 from saved receipts and read-only Baige metadata. **Proposed,
not approved or launched.** The human source/target design stays all40 source
goals → full LIBERO90 + Plus + Pro. No target trajectory enters pretraining;
no goal filtering or outcome-selected source cases. The original32-GPU-hour
pool now has3,489 seconds unassigned after terminal reconciliation and the
new1,400-second PPO warm-start /1,000-second SAC continuation qualifications.
Those two continuation checks are funded from the original pool. The24h/122h
figures below remain proposed ceilings, not approvals. Component margins and
actual terminal qualifications must be aliased once rather than charged twice.

The recommended meaning of1M is **cumulative per production learner**, including
source training, source validation, settling, qualifications, failed prefixes
and retries. Keep old source data/recoveries and count inherited interactions
once. SAC produces one corpus reused by AD/DPT full/static; those four offline
fits do not each require another1M interactions. Shared PPO and its separately
trained reset control each need their own fresh on-policy data.

| Learner | Known completed source floor | Maximum additional actions to cumulative1M | Conservative new compute ceiling | New peak storage ceiling |
| --- | ---: | ---: | ---: | ---: |
|SAC teacher bank → AD/DPT corpus |411,578 |≤588,422 |24 GPUh |120 GB |
|Shared recurrent PPO |171,166 |≤828,834 |44 GPUh |64 GB |
|Separately trained reset PPO |171,707 |≤828,293 |44 GPUh |64 GB |
|Offline full/static fits and exports |Reuses SAC data |0 |4 GPUh |Within32 GB joint derivative/evaluation allowance |
|One fixed target comparison wave |Separate from source |≤250,000 target actions |6 GPUh |Within that same32 GB allowance |
|**Reviewable full envelope** | |**≤2,245,549 new source +250,000 target** |**122 additional GPUh** |**280 GB new peak** |

These production floors are deliberately lower bounds: existing separately
metered source qualifications/validation/control costs must also be aliased
once into the appropriate ledger before launch, reducing the additional action
allowance. For example, the earlier SAC technical qualification used12,944
actions; it is not erased or automatically charged to every learner. No old
9500-action stream is edited in place. Continuation needs a fresh identity and
explicit hash-bound recovery/cost import. If the scientific choice is a fresh
source distribution instead, that is a separately declared new1M generation;
its old development cost stays reported. A fresh full three-learner generation
would need roughly136 source GPUh rather than112, plus the same10 GPUh fitting/
evaluation allowance. A cumulative R3→batch14 PPO continuation is an explicit
training-phase change with inherited weights, not batch14 training from random
initialization. A fresh batch14 comparison must use the fresh-generation budget
and still report old development costs. Neither option implies a scaling benefit.

## Source allocation and cases

For each learner's cumulative1M envelope, propose880,000 actions for training,
80,000 for source validation and40,000 for qualification/retries/uncertain
intents. All three numbers include settling. Within the total, allow at most
72,000 settling actions; this is a nested allowance, not an extra72k. Inherited
validation already above a proposed subcap is retained and the other subcaps
shrink; completed receipts are never rewritten. A complete-case/batch stop may
leave unused allowance. Do not manufacture extra short episodes to spend it.

SAC:40 persistent per-goal actor/replay/optimizer streams;22k all-in training
allowance per goal, with source goals visited in an outcome-independent
Spatial/Object/Goal/10 round robin. Cycle real train-state IDs0–39 before
repeating one with another seed; report actual nominal and post-settle hashes
separately from seeds. Each case contains four complete episodes. Existing
qualified SAC collection uses400-action horizons, H10, batch256, UTD20 and
warmup>500 real steering decisions. Native220/280/300/520 SAC horizons would
require a declared small implementation/qualification change, not an unrecorded
launch flag. Keep this horizon difference explicit versus new native PPO.

The completed SAC collection has527 train cases /2,108 episodes /31,351 query
chunks:783.8 chunks per goal, only272.9 chunks/goal acted by learned SAC after
warmup (20,436 Gaussian chunks;10,915 learned chunks). At the old mean rate,
880k training actions would give about84.6k chunks,2.11k/goal and roughly1.6k
postwarmup chunks/goal—an estimate, not a quota or proof of teacher quality.
That is about5.9× the old learned-behavior exposure. Aggregate case count would
be about1,421, or35.5/goal; actual per-goal counts differ with episode length.
Representative old rates imply roughly56 cases for Spatial00,32 for Object00
and19 for LIBERO10:00 at22k/goal. Thus full40-state coverage is not guaranteed
for every goal; additional seeds never substitute for that claim. Fixed H400
permits at least13 worst-case K4 cases per22k stream before overhead, rather
than a guarantee of35 or40 successful cases.

SAC validation can retain the existing eight predeclared source goals
(tasks0/1 in each suite), states40–49, with independent teacher/replay and no
training-case overlap. The old selected validation already costs85,301 actions,
so the nominal80k subcap is already exceeded: retain it, and reduce new training
or reserve by5,301 before other historical validation/qualification charges.
Do not cold-start40 tiny validation teachers that cannot clear warmup. Fixed
final-teacher validation alone cannot be presented as an improving AD history.

PPO:14 complete K4 trials per global batch, two token-weighted epochs/LR1e-4,
one shared policy, native horizons, no updates during collection. The fixed
all40 cycle balances complete-trial counts to within one; per-goal actions are
reported separately. State IDs cycle0–39 per goal.100 updates is a target:
1400 trials/35 visits per goal would cost1,904,000 actions at maximum lengths.
At the old successful-trial mix,880k training actions is about87 batches and
30–31 nominal IDs per goal; new horizons/policy outcomes can change that.
The remaining training allowance must fit a full14-trial worst-case batch;
retained partial batches survive time stops and do not receive smaller updates.
One frozen K4 validation case on each of40 source goals at state40, plus one
native episode per same case, costs at most68k native-horizon actions, fitting
80k before retries. Fixed parameters throughout; native costs must be counted
once if shared across learners. Full90/Plus/Pro remain target evaluation only.

## Compute and storage evidence

Measured SAC cost is30,886 GPU-seconds /411,578 selected source actions:
**20.85 GPUh/M** including the existing cold starts. PPO full is21,095 seconds /
171,166 actions: **34.23 GPUh/M** for the old one-trial recipe. Simple top-up
extrapolations are12.3 and28.4 GPUh, respectively. SAC's larger learned share
adds optimizer work: the old mixed17.8 ms/update reference suggests about4–6
extra GPUh for roughly1M additional gradient steps. The24h SAC and44h PPO caps
include uncertainty/setup/retries; they stop early if exhausted. Fresh batch14 qualification is now complete:651.24s collection,4.09s replay,
13.46s update,8,313 all-in actions and852 charged GPU-seconds. This is about
22.4 GPUh/M at measured steady collection/replay/update rate, or28.5 GPUh/M
including the cold qualification. One all-success batch cannot establish
steady-state throughput or scaling benefit. Sampled device peak was10,804MiB;
collection was serial, so14 means the gradient batch rather than parallel
simulator lanes. The old one-trial figures remain historical evidence. Both exact
individual and merged replay add computation and are explicitly metered.

| Completed storage example | Raw / action | Journal / action | Full latest recovery | Final paired AD/DPT or candidate | Whole retained layout / action |
| --- | ---: | ---: | ---: | ---: | ---: |
|SAC Spatial00,7,892 actions |19.87 kB |31.57 kB |55.55 MB |22.00 MB paired |77.55 kB |
|SAC Object00,8,230 actions |22.80 kB |38.22 kB |56.84 MB |23.46 MB paired |77.90 kB |
|SAC LIBERO10:00,8,275 actions |22.36 kB |38.43 kB |57.20 MB |23.87 MB paired |77.71 kB |
|PPO r3 only,144,157 actions |21.88 kB |0.043 kB |16.58 MB |5.65 MB candidate |31.69 kB |

MB/GB/kB are decimal. Recovery/candidate sizes are fixed/per-stream overhead,
not linearly growing bytes per action; their current PPO ratios are115 and39
bytes/action only at this snapshot. The PPO numerator excludes inherited
records; dividing by cumulative171,166 would understate its storage rate.
Old three SAC streams publish full recoveries/paired cumulative exports every
16 completed cases plus final; atomic latest is refreshed every episode. PPO
archives one5.65 MB behavior policy and one rollout per old trial. Its r3-only
behavior archive is1.119 GB and rollout archive0.238 GB. Batch14 would archive
one behavior policy per14 trials; the same100 policies are about0.565 GB, with
full raw episodes/likelihoods and pending-batch recovery retained.

The existing layout therefore projects roughly78 GB/M SAC and32 GB/M PPO,
before workload variation/atomic copies. Proposed120/64/64 GB caps provide
headroom, and32 GB covers new fits/exports and target traces. The completed
four-candidate25/100/500 artifact tree is only0.365 GB. No1TB source dataset is
required by these measurements. Raw compression varies across tasks and these
four examples are not a worst-case proof; enforce free-space/storage stops.

For NEW runs only, preserve atomic latest full recovery plus sparse immutable
full milestones (existing16-case cadence plus final is already usable), and
archive policy-only versions at declared publications. Retain raw public RGB/
proprioception, noise, executed actions/rewards/boundaries, policy clocks and
behavior likelihoods/caches. Emit one final hash-bound AD/DPT history per
stream rather than concatenate overlapping cumulative exports. Chronological
AD subsets preserve their actual policy clock; DPT subsets relabeled by the
final teacher retain that teacher's full source cost. A different prefix-
teacher claim needs its own preserved actor/checkpoint and cost. Do not delete
or repack old recoveries. Any more compact journal/content-addressed frame
format needs qualification before being used as the storage premise.

PFS check:2,199,023,255,552 bytes total; about1.332 TB free (1.211 TiB) at the
recorded time. Current RoboTwin/LIBERO/other usage is already included.280 GB
new LIBERO peak would leave about1.052 TB before other growth. RoboTwin's
existing owner audit recorded16.663 GB raw frame/decision blobs across1.137M
source actions, excluding JSON/policy/checkpoint archives. Its ongoing growth
is not reserved or bounded by this plan; no separate private2TB is assumed.
Reprobe free space/quota and coordinate the owner's growth before launch;
retain at least500 GB shared free as a planning floor, not an exclusive claim.

An optional two-stream-per-GPU packing qualification could reduce GPU-hours,
but is unproven: measure real throughput, memory, CPU/rendering interference
and exact per-stream recovery before changing these ceilings. Keep each
scientific batch, loss weighting and physical ledger unchanged. Do not assume
2× speedup or silently multiply processes inside the current allocation.

**Next decision:** after the currently funded baseline/batch14 diagnostics,
select and predeclare the source-history distribution, then approve either
the24h/120 GB SAC top-up first or the full122h/280 GB envelope. Existing data
are mostly already successful and contain only10 learned-behavior
fail→later-learned-success cases; more data is not assumed to repair transfer.
No approval, allocation, budget mutation or launch was performed here.

Evidence: [compact measurements](metadata.json),
[production receipts](../libero40-bulk-approved-20261030-r1/terminal-source-receipts.json),
[original compute ledger](../libero40-bulk-approved-20261030-r1/compute-ledger.json),
[batch14 contract](../../docs/vla-source40-batch14-ppo.md), and RoboTwin owner's
`outputs/robotwin-native-failure-20261030-r1/RECOVERY-COST-AUDIT.md`.

## Executable cumulative continuation refinement, 2026-10-31

The source-SAC continuation pipeline is implemented in e7900a7, with its
1,000s/5,000-new-action qualification funded inside the original32h. The prior
source floor is432,831 actions after including12,944 technical qualification,
10 failed-reset settling,7,540 frozen source controls and759 speed-qualification
actions once. Keep cumulative per-train-stream21,800 actions (872,000 total),
import actual85,301 validation actions with no new tiny teacher warmups, retain
21,253 historical extras and reserve20,000 for NEW final-teacher competence.
Maximum cumulative total is998,554, with1,446 unassigned. NEW training suffixes
are at most545,723 actions, including the qualification prefix. See
[executable contract](../../docs/vla-source40-sac-topup.md). Old source actors
remain40 separately trained per-task actors; this is not a shared all40 SAC
baseline. Source competence remains a separate measured check.

R3→batch14 PPO warm start is an explicit phase change, not fresh training.
Cumulative historical source charges including validation/qualifications are
195,964 full and196,776 reset; new phase ceilings therefore shrink to804,036
and803,224 before any new qualification is added. Shared qualifications are
deduplicated in joint campaign costs. The active warm qualifier preserves old
238/476 update/Adam clocks, starts new phase counters separately, and does
zero restore SGD. A larger phase must bind its own whole14-K4 boundary and
retain partial batches at the time/action stop. Old and new actions are both
reported; no claim of1M newly collected actions per learner is implied.

Human cadence correction: overlap collection and offline fitting from complete
source snapshots and target evaluation. Optional repeated-query/cache work is
deferred and is not an admission gate for this resource request. The remaining
required continuation qualification evidence must precede a larger allocation;
no24h source-bank or122h campaign has been launched.
