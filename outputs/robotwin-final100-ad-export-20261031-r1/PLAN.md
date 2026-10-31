# Final100 AD-only CPU profile — prepared, not admitted

Prepared Oct31 after the original RoboTwin campaign closed. Human approval of
the separate AD-first phase is pending elsewhere. This preparation performs no
allocation, test, checkpoint/model load, data inventory, projection or export.
The original campaign and its deleted monitor remain closed. Future host,
resource, projection, index and model hashes in `profile.template.json` are null.

The profile uses the existing CPU gate, phase worker, local Git staging,
proof dispatcher and owned runtime under
`outputs/robotwin-sac65-cpu-compat-20261031-r6/`. It adds a named final100 route;
it never invokes that run's fixed65 projection/export sequence. Old pinned run
worktrees and receipts are unchanged. The new route has static validation only;
prior software passes are not a claim that this final100 payload exists.

## Inputs and existing consumers

Use exporter commit `7401cd50c038a4cb35149eceeb72128e9a368cc3` and exact c62
`control.py` SHA `67db52fa8f6fa4e8bb56bc6e5cea9f531b5fe41716845350f3372226c5de77a0`.
Copy the already closed62-root ancestry verbatim to the prepared input path;
its SHA is `0af5082349ec1a032eff6b752e3b91fd473b8c922a394c496b4f75bceff68c7e`.
It includes the actual0061 source-validation root and ends at tip
`5b978a88c423ccdc7e1921f309ec047d0fc6cfe034d07e1f9e5523e6cc2f25d4` /
final coordinator `f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8`.
Both AD and BC use the fixed source100 stage
`59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90`.
The source-only61-root chain and historical pre-evaluation coordinator are not
substitutes for this validation-inclusive final chain.

The unchanged projector runs with `--purpose finalbank`, exact admission and
c62 hashes, a fresh output and actual hostname. There is no cross-device copy
flag. Its full original/projected hash, ancestry-end, containment, collision,
contract, inventory and alias checks remain intact.

The unchanged module CLI `fluffy_bassoon.meta_rl.robotwin_history_corpus`
exports train `--objectives ad --partition train` and validation
`--objectives behavior_cloning --partition validation --validation-iteration100`.
Both use source method SAC, CPU, four threads, exact7401/hostname and no teacher
arguments. The existing reader binds the final coordinator/projection to the
stage actor, contract and counters. It does not newly compare final private RNG;
AD/BC does not use that RNG. There is no teacher competence prerequisite.

After export, rehash actual indices, invoke existing
`run_offline.verify_final_indexes(...,objectives=('ad',))`, then
`robotwin_history.load_dataset` for AD/BC and `validate_partitions`.
Require train600K4/2400episodes/1003790 selected actions and
VAL6K4/24episodes/10180 selected actions, actual duration sums, disjoint actual
task/seed and K4 IDs, unchanged causal tokens, finite bounded executed-noise
labels, index/dataset SHA and ZIP CRCs. The unchanged readers enforce masks,
policy versions, prefix coverage and source/validation selection. No target or
native data becomes a training array. The earliest239-action prefix is retained.
Only a complete payload receipt plus all clean phase/final cleanup receipts can
support a future fit admission. Fits and physical readout remain separate.

## One conditional clock and cost

Aggregate completed software cost is170 CPU-worker-seconds, counted once inside
the proposed7200. The remaining7030 exists only after direct human approval.
The UI preparation rounds down to117minutes=7020;10 seconds are unrequested.
Neither the closed461 nor269 unused software seconds is a credit.

| Bound | Seconds | Scope |
| --- | ---: | --- |
| Scheduler request | 7020 | 0GPU,8CPU,64GiB,shm2,auto-fault-tolerance off |
| Parent absolute deadline | 6720 | min(6900,7020-300), before resource/Git/data/model work |
| External TERM / KILL | 6790 / 6800 | One timer starts before Python/bootstrap |
| Bootstrap wait | 120 | Inside parent, same original origin |
| Owned cleanup reserve | 30 | Inside parent, unchanged |
| Visibility / two local stages / c62 extraction | 5 / 30+30 / 10 | Inside parent |
| Inventory / projection | 180 / 2800 | Owned phases, no clock reset |
| AD train / BC validation / final admission | 2400 / 600 / 400 | Owned phases, no clock reset |

Each phase ends at the earlier of its cap and the shared deadline minus30.
The existing adapter's closed `final100-ad-v1` profile permits6720 globally and
2800 per phase; its compatibility default remains1700/900. The selected profile
is independently checked by parent and guardian and recorded in ownership
metadata. Exact0c16 helper hashes, launch registration, signals, subreaping and
owned cleanup algorithms are unchanged. No arbitrary caller cap is accepted.
Failure/timeout preserves partial data and stops; there is no automatic retry,
deadline renewal or old-envelope borrowing. Terminal scheduler duration is
charged once; parent/pod durations are separate measurements.

Measured fixed65 reference:495.045s projection,683.673s paired AD+DPT export,
9.888s audit,143628 input files/56.812GB. Linear unique-byte scaling from390 to
600 source trials suggests about87.38GB, but final62-root I/O and AD-only export
times have not been measured. These phase caps give finite operational headroom,
not an ETA or a promised pass. No preparatory bulk scan has been performed.

## Storage and I/O allowance

The pending new logical storage allowance is128GiB. Before any bulk hashing or
export, the owned180s inventory requires at most500000 summed source files,
at most88GiB summed existing raw payloads, no links/escapes and one filesystem
device shared with the output. Require16GiB free and a write/fsync probe. Reserve
40GiB logical headroom for exports, indices, metadata and temporary output.
Hardlinked raw payloads count conservatively as new logical bytes even though
their physical blocks are shared. Check the complete output logical size after
each phase against128GiB; retain partial output on failure. Existing source
blobs are never copied, moved or deleted.

The finite read costing allowance is `8 * actual inventoried bytes +64GiB`,
at most768GiB. It covers repeat byte hashes and archive/feature passes; it is a
conservative cost estimate, not a syscall-byte measurement. Final actual counts
and free-space receipts must be published. A failed eligibility or capacity
check is not permission to enlarge storage or add time.

## Conditional launch bindings

Before deployment/submission the execution owner must verify the pending direct
human phase authorization, create an exact hashed authority receipt and a new
hashed profile copy, and review fresh main compatibility. The authority receipt
schema is `robotwin_ad_first_human_phase_authority_v1`, with verified direct
human message reference, `direct_human_authorization:true`,
`CPU_worker_seconds:7200`, `new_storage_bytes:137438953472`. These fields record
the actual human message; they cannot manufacture approval. The prepared null
receipt makes this template reject launch. No fresh question is requested here.

Git-deploy the exact auxiliary and7401 commits to their fresh worktrees, compare
all prepared file hashes, extract the three exact0c16 helpers and c62 object,
and transfer only the bound ancestry and receipts. Do not mutate earlier runs.
Use the existing modern Python3.12 artifact dispatcher and unchanged explicit
CUA proof functions. Actual Running Worker1x0GPU/8CPU/64GiB proof, screenshot,
AX, hostname and nonce must arrive before masking/imports. The same single
bounded Submit-to-proof call and original bootstrap deadline apply. No browser
backend polling or idle allocation waiting for funding.

The scheduled command is the existing entrypoint, with all values resolved from
the future owner-hashed deployment receipt (no invented pin/host):

```sh
bash "$AUX/outputs/robotwin-sac65-cpu-compat-20261031-r6/job-software.sh" \
  "$AUX" "$AUXILIARY_PIN" "$OUT" "$INPUT/actual-bootstrap.json" \
  "$INPUT/owned-primitives" "$INPUT/profile.json" "$PROFILE_SHA" final100-ad
```

The current template deliberately contains no actual allocation or fitted model
binding. `payload-admission.json` will be created only from actual completed
exports and uses actual index/dataset hashes. No model is constructed or trained
by this CPU profile, and no new robot/VLA/CUDA/optimizer operation is permitted.

Static review accepted auxiliary53f5361abf1bbfcd8cb586b4de764dfe6869922f,
profile SHA `1a10e71d2f2049be122168d7e846817f0e0341eedb9eb8bbb41ff3c2ddcb2f8e`. Optimization checked clocks, ownership, resources and every10 auxiliary/4 AD
Git bootstrap hash; the exporter owner checked the existing API calls and
AD/BC admission sequence. The only initial staging blocker was an unused
untracked Mac-only helper entry, removed without changing runtime code.
[HANDOFF.json](HANDOFF.json) binds these reviews. No runtime qualification or
resource authority follows from static acceptance.
