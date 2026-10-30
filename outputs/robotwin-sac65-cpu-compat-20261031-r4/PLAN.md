# SAC65 full CPU gate: local exact-code I/O correction

**PREPARED ONLY. No r4 allocation or qualification.** The only new execution
change materializes reviewed Git code locally so full code-identity checks do
not stat the shared checkout. Scientific checks, guardians and source data stay
unchanged. The execution owner retains sole deployment/admission/UI launch.

## Actual diagnosis and remaining envelope

The three immutable attempts cost 149 + 150 + 150 = **449 scheduler seconds**
(3,592 CPU core-seconds), with no fixture, inventory, projection or export phase.
R3 `job-0mb2yh59u58h` ran 05:54:48–05:57:18 Asia/Shanghai on 2026-10-31.
Its live `git-status` process exceeded its ten-second limit, received TERM and
was reaped cleanly. This was an actual command timeout, not missed completion.
Cold shared-filesystem I/O is an inference, not a measured throughput result.

R3 did establish an actual complete single-CUA proof capture in 22.143 seconds
and successful pre-mask visibility with the real empty `nvidia-smi` placeholder
recorded as ENOEXEC. Automatic artifact delivery failed under macOS Python3.9;
manual same-job delivery took 104.1057 seconds. Neither automatic end-to-end
delivery nor 148-test execution was established. All raw deviations remain in
the owner's r3 evidence. The original runtimes and code remain immutable.

[budget.json](budget.json) binds all three exact accounting snapshots. Of the
original 1,800 seconds, 449 are charged and the prior 181 remain sealed. At most
1,170 are admissible. Request **19 minutes / 1,140 scheduler seconds**, with
0GPU / 8CPU / 64GiB. Seal another 30, for 211 total:
`449 + 1140 + 211 = 1800`. Maximum cumulative CPU core-seconds are 12,712.

The parent and independent guardian share **1,090 seconds**, starting before
bootstrap; cleanup30 and bootstrap at most120 are inside it. One outer timer
uses TERM1,110 / KILL1,120, below the 1,140 scheduler request. Fifty seconds lie
outside the parent clock. R3's observed scheduler duration minus parent elapsed
was 35.30 seconds; those clocks are separate and that observation is not a hard
overhead bound. The scheduler remains authoritative and incomplete cleanup
cannot produce a qualification pass. No timer resets on staging or proof arrival.

## One necessary I/O correction

[stage_local_code.py](stage_local_code.py) runs in the actual allocated job as
an owned stdlib worker, capped at **30 seconds total**, clipped further to retain
the test floor. It does not run a separate pilot or modify the source checkout.

1. Verify `/tmp` resolves to a local filesystem, differs from the shared source
   device, has at least512MiB free and passes an owned write/fsync probe. Use a
   fresh private scratch directory. The measured Git tree before this small
   delta was1,405 blobs /19,703,968 bytes; no model/data copy is required.
2. Read the original exact HEAD. `git clone --shared --no-checkout` borrows the
   existing source Git objects read-only, then checks out the exact reviewed
   commit detached into the job-local directory. No source reset, rewrite,
   cache move, credential transfer or source-controller action occurs.
3. Require the effective HEAD to equal the reviewed pin and run the **same full
   `git status --porcelain`**, including untracked files. There is no tracked-only
   shortcut. Compare original bootstrap files against their reviewed local Git
   blobs; a dirty deployed bootstrap is rejected without overwriting it.
   The shared environment filter removes inherited repository selectors and
   `GIT_CONFIG*` override channels from staging and all later Git consumers.
4. Persist original/effective paths, both HEADs, full clean-status output/SHA,
   filesystem/capacity evidence and bootstrap hashes in `local-code.json` and
   persistent logs. All consumers import/run from the exact local checkout.

Keep durable ancestry/input references on the **original immutable PFS paths**.
The effective temporary code path is execution provenance, not an exported input
hash path. Raw corpus/checkpoints, projection/output storage and the exact three
ownership helper files remain in their existing locations. Data hardlinks still
require source/output to share a device; moving code changes no data semantics.

The guardian, transfer guard, visibility worker and CUA capture helper are
byte-identical to r3. The local bridge now refuses macOS Python<3.10 before it
arms, and records its executable/version. Use the explicitly verified bundled
**Python3.12.14**, not an unqualified generic `python3`. Python documents the
macOS cross-process monotonic-clock change in3.10 in its
[time documentation](https://docs.python.org/3/library/time.html#time.monotonic).
No clock protocol or generic guardian redesign is introduced.

## Actual phase admission

The unchanged floors are 960 seconds remaining before tests, 750 before
projection, 660 before export and45 before audit. Setup must therefore finish
within **130 seconds** of the same1,090-second start. A30-second local-code cap
plus the observed22-second capture leaves useful conditional setup room, but
automatic transfer and inventory elapsed time are not yet measured successes.
Bootstrap120 plus inventory180 cannot both reach their maxima and admit tests.
Every admission uses actual remaining time; failure stops with incomplete
evidence. No completion promise, retry, new allowance or sealed-budget borrowing.

Keep all original file/byte/free-space/same-device eligibility checks and the
full byte-SHA projection and paired AD/DPT audit. Source remains SAC65 /26 roots /
390 K4 /1,560 episodes /651,638 actions with the same source/checkpoint/ancestry
SHAs. No teacher competence, fit, source100, robot, VLA or SGD authority is added.

Require **150 fixtures, zero failures/errors/skips**:148 unchanged plus2 meaningful
Git-linked-worktree regressions for exact full-clean local execution and dirty
bootstrap rejection without source overwrite. None has run. Static checks are
syntax, source-byte/AST comparisons, metadata hashes and budget arithmetic only.

## Owner launch handoff

After one narrow exact-pin review, follow [CUA-HANDOFF.md](CUA-HANDOFF.md): fresh
source code worktree, existing exact helper staging, modern local Python, fresh
live bridge, installed unchanged complete-detail CUA helper, and one actual
Submit/detail/proof invocation with hard tool timeout<=60s / original55s inner
deadline. Only the owner submits. A full pass requires all phases, clean owned
teardown and terminal scheduler-cost reconciliation; no earlier preflight-only
attempt is reported as corpus qualification.
