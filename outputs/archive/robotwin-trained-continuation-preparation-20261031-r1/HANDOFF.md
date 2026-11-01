# Original100-round paired continuation — CPU qualified

Final supervisor `0c16c2b794f8de5b78603206917e72195083c265` passed its allocated
Linux CPU gate on Baige `job-q6dhvcewwwx1`: 67 tests, including all three real
spawn/cleanup regressions, with zero failures/errors/skips. Both source checks
were clean at the exact pin. Independent hash/XML/allocation readback is in
`linux-cpu-gate/LOCAL-READBACK.json`; see `CPU-QUALIFICATION.md`. This removes
the software-gate prerequisite. The execution owner still binds the latest
released heads and operational authority before launching the continuous run.
The qualified deployed checkout stays immutable at0c16; later evidence-only
commits do not replace its execution pin.

The first trained paired run passed at unchanged runtime57016/c353:18,879new
source actions, one real update/five SGD steps per arm, all12replay error pairs0,
frozen motor unchanged, clean sessions and released GPU contexts. The source
owner's terminal UI reports450GPU-seconds; peak65.5537GiB/70.3878GB. This is an
actual trained result, not a matched scalar speedup ratio. Protected thin
readback is under `outputs/robotwin-trained-pair-20261031-r1/`.

The source owner continued short, already-qualified real training pairs while
this supervisor received review and its allocated CPU gate. Its
initial inputs must be the **latest owner-issued** final manifests at launch,
not hardcoded first-run53/55 or historical52/54 boundaries. The completed R2
boundary was54/56; further bridge rounds change it normally. No current
allocation, worktree or launcher is modified by this preparation.

## Execution contract

One future exclusive96GB GPU/32CPU/192GiB/shared16GiB allocation owns two
independent source streams. Runtime570 remains unchanged. Each fresh child
still collects at most six complete K4 source trials, uses the original
global batch6/five epochs/LR1e-6/KL0.1, commits raw/native counts and group ACK,
and exits at a complete updated round. A new whole-process output identity is
used every time. SAC remains scalar under its owner.

The supervisor performs full owner-release/canonical-head admission and one
idle GPU probe initially. It then audits and respawns each arm independently;
one arm's completed child/audit never signals the other arm's live session.
Every child still loads its frozen motor and performs its own native parity
check. There is no claim of a persistent warm model or elimination of that
per-child cost. The growing canonical journal is verified at post-child
audits, which remain externally time bounded.

Unequal source-round offsets are expected. An arm reaching100source rounds
needs its original final evaluation in a fresh child; after all original
10/25/50/100 units are complete it retires, while the other arm continues.
Input-due evaluation preserves completed-unit cursors and may make progress
without new source trials. Only complete native evaluation units, identical
source learner/sampler/coordinator RNG and closed raw/physical accounting may
continue automatically. A timeout, native abort, unclosed episode, missing ACK,
unknown charge or nonzero exit stops automatic continuation and requires owner
audit. No failed physics is replayed blindly.

A fresh allocation can also begin with an arm already at100. The owner must
issue a **current post-evaluation** boundary manifest from that arm's final
coordinator/canonical head and completion proof; do not reuse a stale
source100/pre-evaluation manifest. Production-only admission permits this
boundary and subtracts exact completed evaluation units. An already complete
arm retires before model work; if both are complete, no GPU probe/model is
needed. The original one-round launcher's default admission remains strict.

For evaluation-only children, compare the authoritative coordinator RNG,
private learner sampler, task schedule and method cursor. The saved learner
`rng` field is another capture of global state, not an independent acting RNG:
it may reflect evaluation while the coordinator's stage override preserves
source state. Record both digests rather than falsely requiring those redundant
snapshots to match across partial evaluation units.

## Resource authority and planning estimate

The human-authorized goal remains100rounds per learner; the old overall compute
ceiling was explicitly superseded. Future allocations are operational slices,
not a newly invented global GPU pool. The owner must reserve and meter the
exact future slice, with at most46,800schedulerGPU-seconds/43,200driver seconds
and the same one-GPU resource dimensions. No LIBERO budget or sealed
qualification hold is borrowed. This preparation reserves nothing.

Use the latest actual source counters to calculate the remaining whole rounds.
For example, after R2 at54/56,46persistent and44reset source rounds remain:
maximum972,000additional source actions, plus at most48,800original final-stage
evaluation actions before subtracting completed units. Those are inside the
unchanged original worker caps, not additional caps. The actual early-success
source count is lower; never cut episodes/trials to meet an action target.

Using the measured first450seconds as a conservative same-configuration
planning basis gives46paired intervals≈20,700seconds/5.75GPU-hours, before
allowing for final evaluation, differing remaining arm lengths, journal-audit
growth, filesystem/model cold loads and failures. This is a planning estimate,
not a completion guarantee or a scalar speedup. Recompute from actual later
results and charge the complete allocation-active duration exactly once.

## New owner inputs

Create a fresh c353-format compiled initial plan from the latest released
manifests and frozen-motor receipts, with a new terminal owner release and
stable original canonical heads. Preserve the fixed parent SHA, original
source/total caps and all scientific arguments. Its initial2700-second
subunit is explicitly included in the longer operational authority; it is
not an extra reservation and cannot reuse the spent first-pair plan.

The long authority has schema
`robotwin_trained_pair_continuation_authority_v1`, runtime570, DBlineage,
fixed `parent_manifest_sha256`, hash-bound `initial_admission_plan`,
`source_goal_rounds:100`, `original_total_compute_ceiling_seconds:null`,
`qualification_capacity_reused:false`, `automatic_failed_retries:false`,
`resources` with the same dimensions and `scheduler_cap_seconds<=46800`,
`maximum_driver_seconds<=43200` and at least180seconds below the scheduler
cap, `child_wall_seconds:3600`, `initial_subslice_seconds:2700`,
`initial_subslice_included:true`, plus hash-bound original `authority` and
`prior_charge_and_physical_snapshot`. The full allocation is funded/metered
once under this authority; overlapping arm work is not charged twice to the
included initial subunit.

After the owner reviews the final exact code and its allocated Linux CPU gate,
the single future invocation is:

```bash
bash "$LAUNCHER/scripts/diagnostics/robotwin_trained_pair_continuation.sh" \
  "$LAUNCHER_SHA" "$RUN_ID" "$INITIAL_PLAN" "$INITIAL_PLAN_SHA" \
  "$CONTINUATION_AUTHORITY" "$CONTINUATION_AUTHORITY_SHA"
```

The launcher checkout and separate runtime570 checkout must both be clean and
commit-pinned. Review current main before the future submission. The wrapper
submits no job and controls no predecessor allocation. A hard scheduler limit
and outer timer remain necessary for kernel/filesystem failures; software
ownership guards do not establish whole-container teardown under every fault.

The passed CPU tests cover independent offsets/retirement, final evaluation,
closed-unit progress, authority/cap changes, isolated session cleanup and the
real tiny-CPU-Adam driver-to-audit path. Rerun relevant prior launcher tests for
any changed admission helper. No tests/model/GPU work ran on macOS or through
the development SSH host. No future launch is authorized by a prepared test
file alone.
