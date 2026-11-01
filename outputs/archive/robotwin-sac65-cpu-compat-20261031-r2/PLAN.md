# SAC65 residual CPU compatibility recovery

**Superseded by [r6](../robotwin-sac65-cpu-compat-20261031-r6/RESULTS.md), the passing qualification.**

**PREPARED, UNLAUNCHED, UNQUALIFIED.** A new exact-pin static review and owner
admission remain required. This is the recovery of the original necessary
CPU qualification, with unchanged scientific scope and immutable r1 retained.

Baige r1 `job-eahrz09v6kkk`, at `29df647fd30469fc04743baf47ba331718c9d1a2`,
failed its 120-second owner-ticket wait before any phase ran. Scheduler cost
was 149 seconds / 1,192 CPU core-seconds / zero GPU-seconds. Evidence commit
`083f570` contains the actual terminal UI and accounting. The exact accounting
bytes are retained in [predecessor-scheduler-accounting.json](predecessor-scheduler-accounting.json),
SHA `8e26ce12a16cc63a8636c345e74c9e453a153945050a447d6ae29e74eebf3260`.

## Residual admission and phase floors

[budget.json](budget.json) binds this diagnosis and the remaining envelope:

- Original 1,800 scheduler seconds minus 149 charged leaves 1,651.
- Request **1,500 scheduler seconds / 25 minutes, 0 GPU / 8 CPU / 64 GiB**.
- Seal 151 unused seconds. Maximum combined time is 1,649 seconds, or 13,192
  CPU core-seconds. No additional attempt is authorized automatically.
- One outer timer: TERM 1,470, KILL 1,480. The absolute parent/guardian limit is
  1,400 from before bootstrap/preflight, with cleanup 30 inside it. There is no
  new 1,700/1,800-second allowance. Bootstrap remains at most 120 inside 1,400.

The parent checks exact budget and predecessor bytes before resource admission.
Its exact clean Git pin binds those artifacts; the runtime intent records that
pin and budget hashes. All original eligibility caps and phase reserves remain.
The minimum floors are tests 30 + projection 60 + export 540 + audit 15 + cleanup
30 = **675 seconds**, excluding bootstrap/preflight. Tests additionally protect
900 seconds for later work, projection protects 660 and export protects 90.
Before tests, at least 960 seconds must remain (30 test + 900 reserve + 30 cleanup).
Thus a maximum 120 bootstrap + 180 inventory leaves at most 1,100 seconds before
other setup; tests can start only if actual remaining time passes their check.
After minimum tests, projection needs 750 remaining, export needs 660 and audit
needs 45. Actual work is dynamically clipped, so a slower corpus can fail closed.
These floors establish admissibility, not measured throughput or a finish promise.

## Unchanged scientific/runtime core

Input is the exact SAC65 chain: 26 roots, 390 complete K4 / 1,560 episodes /
651,638 actions, ancestry SHA `5e65bc308c8d73ab21ff80452c39fec98b873171e49a1432312a8645bd07d861`,
closed checkpoint SHA `b09c13f54e4db8994c9a5d94cd6f18b6af3514d4d5ef4e63f1f91e482578b0e0`.
Reader, teacher/token/mask/counter math, hardlink-only projection, original
import firewall and exact three `0c16c2` helper hashes are unchanged. The copied
CPU adapter is byte-identical to r1. The phase worker changes only the focused
test-file list; no new training/projection/export behavior is introduced.

Require **125 fixtures, zero failures/errors/skips**: the original 75 plus 50
residual/proof/transfer fixtures. This is static discovery, not execution.
The new transfer guard uses a live session-holder through TERM grace, followed
by group KILL and wait. Its independent pipe/PPID watch bounds a transfer even
if the dispatcher dies. Tests cover timeout, dispatcher TERM and SIGKILL,
TERM-ignoring changed-PGID descendants, normal success/nonzero head exits and unrelated-session survival. Status-publication I/O failure keeps the holder resident; transient enumeration
errors retry, and persistent errors kill the known owned members/original group
while marking unresolved SID cleanup as a failure. All real process
fixtures execute only in the allocated Linux qualification.

There is no optimizer step, robot action, VLA query, fit, source100 admission,
PPO/reset extension or teacher competence claim. No source process or ledger
is controlled. Dev SSH performs artifact I/O only.

## Pre-armed actual-proof handoff

[dispatch_bootstrap.py](dispatch_bootstrap.py) is a one-shot local artifact
bridge. It never submits a job or reads a browser backend. It waits for actual
CUA proof and real gate intent, validates their scopes, uploads proof/resource
bytes, verifies remote hashes, and publishes the ticket last using atomic
no-clobber hardlink creation. Existing files are never overwritten. The local
bridge expires after 300 seconds, so owner form preparation/deployment must be
finished before arming it. Each transfer has an independent bounded guard and
remote commands have their own short I/O timer. None wraps or extends the CPU gate.

The bridge alone does not remove the UI-capture dependency. The execution owner
must follow [CUA-HANDOFF.md](CUA-HANDOFF.md): **one bounded CUA invocation includes
Submit, actual Running observation and atomic publication of screenshot/AX plus
the nonce marker**. No model/compaction boundary may separate these operations.
This uses CUA-visible actual job/host/resource data, never guessed allocation
IDs or a prefilled resource template. Cgroup CPU/RAM and GPU invisibility are
diagnostics; they cannot replace scheduler proof of an actual zero-GPU allocation.

The owner reviews the exact CUA sequence against freshly observed controls
before submission. If that complete sequence cannot be prepared, do not launch.
If it fails or cannot capture Running in its bounded watch, fail closed and
reconcile the actual allocation; no automatic retry or resource reset follows.

## Exact execution handoff

After independent exact-pin review, deploy a fresh detached worktree and stage
the same exact three helper files in the fresh r2 input directory. Verify the
25-minute scheduler setting, actual 0/8/64 resources and autofault OFF. Then arm
the local bridge (configured SSH target/port supplied locally):

```text
python3 -I -B -S <local-code>/outputs/robotwin-sac65-cpu-compat-20261031-r2/dispatch_bootstrap.py \
  --inbox <fresh-local-inbox> --code-commit <reviewed-full-pin> \
  --ssh-target <configured-target> --ssh-port <configured-port>
```

Check its live readiness before the single CUA submission/proof step. The exact
five-argument CPU command remains:

```text
bash <fresh-remote-code>/outputs/robotwin-sac65-cpu-compat-20261031-r2/job.sh \
  <fresh-remote-code> <reviewed-full-pin> \
  /mowenjie/fluffy-bassoon-2/outputs/robotwin-sac65-cpu-compat-20261031-r2 \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r2/actual-bootstrap.json \
  /mowenjie/fluffy-bassoon-2/inputs/robotwin-sac65-cpu-compat-20261031-r2/owned-primitives
```

Retain dispatcher/transfer receipts, actual UI proof, bootstrap receipt, complete
phase/import/JUnit/owned-cleanup logs and scheduler terminal accounting. Only a
full `COMPATIBILITY-PASS.json` after clean teardown establishes SAC65 CPU
compatibility. It establishes no scientific performance result.
