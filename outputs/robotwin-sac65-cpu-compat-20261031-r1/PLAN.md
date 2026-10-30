# Closed SAC65 CPU compatibility gate

**PREPARED FOR OWNER STATIC REVIEW. Not launched or qualified.** The execution
owner retains resource review, admission, deployment and UI launch. This is
one fresh run, `robotwin-sac65-cpu-compat-20261031-r1`, with exactly0GPU/8CPU/
64GiB and a1,800-scheduler-second ceiling. No source controls, ledger access,
physical actions, VLA queries, optimizer steps, fits or robot teacher panel.

## Code and scope

The candidate is the exact Git commit supplied with this handoff. Deploy it
to a fresh detached CPU worktree, not an active GPU worktree. The original
qualified algorithm is `1aeea4f9fcc307bb147def92374bb81a3e2cb50e`; this does
not automatically qualify the changed reader. Latest main was refreshed and
reviewed at `acca706d5e75d1da099ff7ce6fddd727dace0fba` during preparation.
[PROVENANCE.json](PROVENANCE.json) records thirteen static AST comparisons:
label/token/mask/counter mathematics and general corpus-path containment are
unchanged; source-recovery identity, projection provenance and acting-policy
path binding are the explicit compatibility delta. Native trainer code is
unchanged and is not invoked in this gate.

Production `BehaviorPolicyArchive` stores absolute policy paths. A flat
byte-identical projection by itself cannot satisfy the old reader's root
containment. The correction accepts an alias **only in `acting_weights`**.
It requires the exact canonical lexical origin in a protected root's policies
directory, a contained projected policies path and identical full-file SHA.
Unlisted symlinks that resolve to an admitted origin are rejected before
normalization. Other RGB/native/metadata paths keep the original containment
logic. Actor kind, finite tensors, tensor digest, version and context SHA
checks still run on the projected archive. Original raw records are not
rewritten, and original archive bytes are rechecked and retained as provenance.

The reader independently checks the projection's bound ancestry/tip, every
ancestor manifest/root/contract/checkpoint-reference tuple, canonical and
backup contract bytes, each predecessor migration and original ordered roots.
The ancestry must be nonempty and end at the exact bound tip manifest path/SHA;
an internally consistent shorter chain cannot borrow a separate full tip.
The first selected-trial ancestor is retained rather than rounded to whole
rounds. All producer scientific/runtime contracts must agree. The closed tip
checkpoint is fully hashed. A differently serialized stage file additionally
needs actual source/task counters, trial count, update count and actor tensors
to match the closed coordinator; an owner boolean cannot supply that proof.

The assembly helper still reuses the exact qualified c62 `corpus_chain` reader
and hardlinks/copies no altered record. This gate permits **hardlinks only**;
no automatic copy fallback. The c62 Git object must already be deployed and
is extracted/hash-checked by the wrapper; no source control/PID function is
called. The candidate reader must pass its new exact-pin qualification.

## Exact immutable input

Use only the pinned26-root SAC65 chain from
`outputs/robotwin-finalbank-offline-plan-20261031-r1/SAC65-REFERENCE.json`:

- Ancestry SHA `5e65bc308c8d73ab21ff80452c39fec98b873171e49a1432312a8645bd07d861`.
- Tip `/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-continuation-sac-20261030-r1/slices/0025/experiment/continuation-manifest.json`,
  SHA `8c429b7d7a66b1374ad95a1136b783717509eef0651f457d775c39b59d5f75b6`.
- Its immutable `coordinator-recovery.pt`, SHA
  `b09c13f54e4db8994c9a5d94cd6f18b6af3514d4d5ef4e63f1f91e482578b0e0`,
  owner-measured684,183,761 bytes.
-65 complete source rounds /390 K4 /1,560 episodes /651,638 source actions.
  SAC stays on e825/DB; do not attribute PPO's570 migration to this stream.

The first ancestor contains39 completed rounds/236 complete K4 and protected
selected-trial prefix actions; the next root reaches41 rounds/246 K4. There
is no one-round-per-root assumption, reward filtering or prefix deletion.
No moving checkpoint, newer source slice, target clone or live ledger is read.
This gate never routes65 through the source100 command runner and never sets
the real `closed_source100_authorized` or `assembly_cpu_gate_passed` flags true.

## Actual allocation proof and import limits

Before hiding CUDA, [gate_job.py](gate_job.py) requires an owner-hashed receipt
of the **running allocation** and a hash-bound UI proof: exact job/hostname,
0GPU/8CPU/64GiB. The [resource template](resource-receipt.template.json) is
deliberately unissued and fails admission. The wrapper records existing GPU
environment values, `nvidia-smi -L` and `/dev/nvidia[0-9]*` visibility. A GPU
visible before masking stops the gate; it is not disguised by an empty CUDA
variable. Receipt semantics are the owner's actual-allocation attestation;
the file/UI SHA checks establish immutability, not independent screenshot OCR.

The parent starts Python with `-I -B -S` and remains stdlib-only. Only after
the zero-GPU receipt does it mask CUDA. Phase workers also use `-I -B -S`,
install the import firewall before adding the existing environment's
site-packages/source paths, and do not execute `.pth` or sitecustomize files.
JAX/JAXlib, OpenPI, RLinf, SAPIEN, MuJoCo/ManiSkill, native RoboTwin/motor
backends and full VLA policies are forbidden. The existing CPU Torch/NumPy
feature-steering heads (`vla_noise_sac/ppo`) and pure recurrent-core definitions
are necessary for the qualified converter; no frozen VLA is loaded. CUDA
initialization and Adam/AdamW steps are guarded against. Each phase retains
its import/CUDA receipt. No trainer/fit is invoked.

## One bounded job and I/O assumptions

[job.sh](job.sh) starts an external timer before the Python interpreter,
resource checks, imports, tests, projection or export: TERM at1,770s, KILL
after10s, below the1,800s scheduler limit. The parent has a1,700s deadline
starting before its own preflight. All startup, inventory and failure time
belongs to this allocation. Scheduler terminal cost is still reconciled by
the owner; process elapsed time is not billed time.

Before any bulk SHA/model import, stat-only inventory records exact per-root
file/byte counts and counts below4KiB/64KiB. This preparation has **not**
enumerated production payloads. Conservative eligibility caps are500,000
regular files and256GiB total referenced-directory bytes across26 roots;
the I/O planning estimate is up to8× that byte count (at the cap,2TiB logical
reads), plus tiny-file open/stat overhead. This is a conservative bound,
not a measured throughput or a promise the gate will finish. Actual counts
and elapsed time govern residual admission. A large or slow corpus can make
this one job incomplete; no second1,800s allowance follows automatically.

All source files must have the destination filesystem's device ID. No source
symlinks are followed. A same-filesystem hardlink may still fail due to mount
policy; such failure is retained rather than converted to a copy. Require
at least16GiB free for metadata/indexes/temp files and an owned output write/
fsync probe. No raw-payload capacity is consumed by copies; metadata and
export NPZs do consume new space. No chmod/rename/delete touches original files.

Phases share the same remaining deadline:

| Phase | Maximum | Remaining time protected for later phases |
| --- | ---: | ---: |
| Resource/Git/ancestry/stat inventory | inventory at most180s | never outside global deadline |
| Focused fixtures/import checks | 180s | 900s |
| Full byte-SHA hardlink projection | 900s, clipped by remaining time | 660s |
| Paired saved CPU AD/DPT export | 900s, clipped by remaining time | 90s; require at least540s before starting |
| Array/count/import audit | 90s | global deadline remains binding |

If the protected remaining allowance is insufficient, stop before starting
that phase. The owned guardian handles phase workers and descendants on
timeout, nonzero, caller signals or caller death; it never controls source
sessions or constructs a ledger. The one outer timer bounds the caller, and
the independent guardian shares its absolute deadline. No retry, changed
deadline or copy fallback occurs.

## Meaningful fixtures and real admission

The fixture phase requires **75 tests, zero skips/failures/errors** in JUnit:
17 existing native-corpus cases plus58 new focused compatibility cases.
These are static discovery counts, not a claim they have run. New fixtures cover:

- A normal original single-root corpus; a selected K4 prefix split between
  roots; every AD/DPT array equal to the unsplit baseline, including actual
  histories/noise/rewards/boundaries/durations/teacher labels.
- Absolute production-shaped policy refs; unlisted/identical foreign archives,
  file and ancestor-directory symlinks, byte/context-SHA tamper and incomplete
  projection rejection; generic frame/native path containment stays closed.
- Hash-rebound ancestry reorder/drop/duplication, manifest root/contract/lineage/
  count mismatches; actual stage source/task/update/trial/actor mismatches.
- Identical-only file collision deduplication; conflicting collision, runtime
  change, top/nested/canonical symlinks, partial files,65-as-final rejection and
  retained failed hardlink without copy fallback.
- Production dictionary runtime schema and current65 index-source/teacher/
  projection/count/partition bindings; an old bank cannot be substituted.
  The contract's planned goal remains100 while the immutable admitted cursor
  is65. Nine optional final100 index-metadata fixtures are deferred in a
  separate file and are not collected by this gate. No real PPO/reset vector
  fragment, final100 runner or finalbank/teacher/fit qualification is included.
- CPU import firewall and blocked optimizer step. No fitting is permitted,
  including fixture Adam steps; head forwards/labels are CPU-only checks.
- Shared-deadline bootstrap, atomic ticket arrival and stale scope rejection;
  ancestry terminal-tip closure; real tiny worker/descendant lifecycle checks
  for timeout/TERM/INT/SIGKILL, actual reaping and unrelated-session survival.

## Presubmittable bootstrap and owned cleanup

The launch command binds a fixed inbox path and the reviewed code/output,
not an invented future hostname or proof SHA. The parent derives its actual
hostname, records bootstrap intent and waits at most120s for the owner's
atomic final ticket. The ticket binds that hostname, exact code pin/output
and actual resource-receipt path/SHA. The owner first uploads the actual
running-allocation UI proof and resource JSON, then atomically renames an
issued ticket to the inbox. A partial upload is not a ticket; a present stale,
malformed or mismatched ticket fails. The [ticket template](bootstrap-ticket.template.json)
is unissued and rejected. No model/test import occurs while waiting.

Both timers start before this wait. The same absolute1,700s deadline is passed
to every later operation; a120s wait leaves at most1,580s, never a fresh1,700s.
The one outer GNU timer remains TERM1,770/KILL1,780. There is no nested GNU
timeout wrapper or new operational allowance.

[owned_cpu_runtime.py](owned_cpu_runtime.py) dispatches CPU commands using
only the qualified ownership primitives at
`0c16c2b794f8de5b78603206917e72195083c265`:

| Predeployed source file | SHA256 |
| --- | --- |
| `robotwin_trained_pair_runtime.py` | `116b5251b43454a4c61fe0bd4faeb3d1cbf6d7a2dd08356241599cbd2f32827c` |
| `robotwin_trained_pair_continuation.py` | `173774c46c177548750c78e1e09aa5c8caf583d265be8ecb5517757077274c9b` |
| `robotwin_trained_pair.py` | `d8ee934c580cb1d1ab3991f446618af2bd9e06a6ece97cf0fe83aa828b882dc0` |

Owner stages exactly these three verbatim files in a private helper directory
before allocation; no pycache or unrelated modules belong there. No code or
test executes during deployment. Their GPU continuation APIs/minimum budgets
are not called. The CPU-only adapter reuses `ContinuationGuardian.launch`
(including corrected spawn registration with normal child signal masks),
captured SID/UID/leader-start identity checks, parent pipe/PPID watcher and
bounded `cleanup(graceful=False)`. The new adapter still needs qualification.

An independent stdlib guardian owns every phase session and its descendants,
including Git subprocesses and changed-PGID children. It watches caller loss
and shares the original absolute deadline, even if the caller is SIGKILLed.
A Linux subreaper in this new guardian adopts/reaps its own orphan descendants;
it cannot adopt unrelated source jobs. TERM/KILL and wait results are retained
per cleanup. Parent TERM/INT closes the owner pipe and waits; repeated signals
cannot interrupt final bounded cleanup. Only captured owned identities are
signaled. The original source controllers/ledgers are never invoked.

Thirty seconds of cleanup are **inside**1,700s, so actual work stops by1,670s
or earlier after bootstrap/preflight. Visibility/Git checks also use this
ownership layer. No completion marker is written until phase descendants and
guardian are cleanly reaped. Kernel-unreapable work produces failed cleanup
evidence and no qualification; the scheduler's original1,800s boundary remains
required. Tiny CPU regressions exercise timeout/TERM/INT/SIGKILL, normal child
signal masks, a changed-PGID descendant, actual waits and an unrelated session
left alive. No source PID is discovered or controlled by these tests.

After fixtures, the real helper creates `private/projection` by full byte SHA
and same-device hardlinks. The existing converter then exports `ad dpt` from
the exact closed source65 coordinator, using that same source-only checkpoint
as a deterministic-mean CPU compatibility teacher. This is a saved-data
artifact, not competence measurement, source100 export, fit or robot action.
No validation bank is fabricated at65; fixed-policy validation semantics are
covered by the existing CPU fixtures as explicit BC/DPT.

The converter must admit all390 complete global K4 and reconcile651,638
physical actions with each task counter and every native valid-action mask.
It must preserve actual actor updates/identities and fail fixed-policy false
AD. The final audit independently compares all paired fields except targets,
checks AD targets equal executed noise, DPT labels finite/bounded1.5, exported
NPZ hashes and mask-derived durations summing651,638. Native masks are checked
by the existing converter, not falsely described as a second raw-mask audit.

## Evidence and publication boundary

Fresh runtime output is exactly
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-sac65-cpu-compat-20261031-r1`.
It retains intent, actual pre-mask resource/UI proof, parent and raw phase logs,
JUnit/fixture summary, exact I/O inventory, pinned reader/context/admission
hashes, phase status/timing/import receipts, private projection and paired
NPZ/index, paired audit, terminal summary and shell exit code. Original source
roots/checkpoints remain protected in place.

Only all phases finishing within the global deadline may write
`COMPATIBILITY-PASS.json`. A failure writes `INCOMPLETE.json` when possible
and retains private artifacts; an external kill can leave only intent/phase
logs and a nonzero shell exit code. Missing final receipts never imply pass.
No partial projection/export is published as a qualified corpus or finalbank
admission. The owner must retain actual terminal scheduler/resource proof
before declaring this gate qualified. It cannot qualify teacher competence,
GPU/fitting performance, source100, a new reservation or active-source migration.

## Owner deployment and launch command

Before allocation, review this candidate and its narrow reader delta, confirm
the existing Python environment, deploy its exact Git pin and the c62 object,
stage the three0c16 files above and choose an initially unissued bootstrap
inbox. Once UI Running, upload the actual0GPU receipt/proof and issued ticket.
No setup command
executes tests or model work on development SSH. Inside the admitted allocation:

```bash
bash <candidate-worktree>/outputs/robotwin-sac65-cpu-compat-20261031-r1/job.sh \
  <candidate-worktree> <exact-candidate-commit> \
  /mowenjie/fluffy-bassoon-2/outputs/robotwin-sac65-cpu-compat-20261031-r1 \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r1/actual-bootstrap.json \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r1/owned-primitives
```

This file submits no job. The handoff message provides the actual candidate
pin after commit; placeholders are not a request to use moving latest/HEAD.
