# Actual r2 preflight corrections and remaining SAC65 CPU gate

**Superseded by [r6](../robotwin-sac65-cpu-compat-20261031-r6/RESULTS.md), the passing qualification.**

**PREPARED ONLY. No r3 allocation, no runtime qualification.** Same scientific
scope; no new GPU, robot, VLA, SGD, fitting or source100 authority. The execution
owner retains exact-pin admission, deployment, proof capture, launch and cost.

## Actual failure, not a converter result

R1 `job-eahrz09v6kkk` cost 149 scheduler seconds and timed out before work.
R2 `job-wcfjtype0eec` at immutable `16d8af8c6a1a6c7cd6980cbbc848deabafa1a6e6`
cost 150 scheduler seconds, Running 05:32:37 to Abnormal 05:35:07 on
2026-10-31 Asia/Shanghai. It also reached no tests, inventory, projection or
export. Actual Chrome bytes were JPEG despite the `.png` filename, so the
bridge rejected them. The first CUA call also stopped at the Running header
before the hostname rendered; later calls completed the proof, so its one-call
contract was **not** satisfied. The owner retained that deviation and manually
published the authentic proof in the same job. Bootstrap wait was 118.11069
seconds. The owned visibility command then failed with ENOEXEC for `nvidia-smi`.
Owned cleanup receipts were clean, with empty sessions/survivors/adopted lists.

Exact owner accounting bytes are retained in [r1](predecessor-r1-accounting.json)
and [r2](predecessor-r2-accounting.json), with SHA bindings in [budget.json](budget.json).
Their 299 total scheduler seconds / 2,392 CPU core-seconds are charged once.
The r2 process elapsed time 118.44084 is not scheduler billing time. Neither
attempt supplies evidence about corpus conversion or teacher competence.

## Narrow code changes

- Save the native CUA screenshot to `actual-running.image`. Detect PNG/JPEG
  from bytes, retain those bytes and SHA exactly, and publish a truthful
  `.png`/`image/png` or `.jpg`/`image/jpeg` proof. No image conversion occurs.
  The remote publisher validates encoding, MIME and bytes before any final file.
- [capture_complete_proof.js](capture_complete_proof.js) waits for the actual
  Running task, job ID, hostname **and complete single Worker resource table**.
  It continues while the hostname/table are missing; a Running header alone
  never releases the loop. The same pre-Submit overall deadline bounds it.
- Run [visibility_probe.py](visibility_probe.py) as a valid stdlib Python worker
  inside the existing owned runtime. Only the utility's `OSError(ENOEXEC)` is
  caught and retained with its path/header/bytes/SHA. Missing or unusable utility
  is diagnostic, never proof of zero allocated GPUs. Unexpected exec errors,
  utility timeouts, worker nonzero and all guardian/cleanup failures propagate.
  Actual authoritative zero-GPU UI proof plus pre-mask device-node, GPU-output
  and NVIDIA namespace checks remain mandatory before masking/model imports.

The guardian and transfer guard are byte-identical to r2. Reader, teacher,
history, masks, counters, projection and export algorithms are unchanged.
Input remains SAC65 / 26 roots / 390 K4 / 1,560 episodes / 651,638 source actions,
with the same ancestry and closed-checkpoint SHAs. No extra generic hardening
or separate diagnostic branch is included.

## Residual cap and honest viability

Original 1,800 minus 299 charged leaves 1,501. Preserve the already sealed 151,
leaving at most 1,350 admissible. Request **22 minutes / 1,320 scheduler seconds,
0 GPU / 8 CPU / 64 GiB**, and seal another 30, for 181 sealed total:
`149 + 150 + 1320 + 181 = 1800`. Maximum cumulative CPU core-seconds are 12,952.

One outer timer: TERM 1,290 / KILL 1,300. The same absolute work deadline is
1,220 from before bootstrap; cleanup 30 and bootstrap at most 120 are inside
it. There is no fresh 1,400/1,500/1,700/1,800-second allowance. Both exact cost
receipts and all timer/sealed values are checked before resource admission.

The unchanged dynamic phase floors require remaining work time of at least
960 seconds before tests, 750 before projection, 660 before export and 45 before
audit. **Setup including proof/inventory must therefore use at most 260 seconds
before tests.** The independent maxima 120 bootstrap + 180 inventory already
sum to 300, so worst-case setup cannot admit tests. This proposal relies on
actual elapsed time passing the check, not a completion promise. If slow,
retain incomplete evidence and stop before the phase. No automatic retry,
copy fallback, new allowance or borrowing the sealed 181 follows.

Keep the 500,000-file / 256-GiB stat inventory bounds, 16-GiB free minimum,
same-device hardlinks only and conservative unmeasured 8x logical-read estimate.
Phase caps remain tests180 / projection900 / export900 / audit90, clipped by
the same remaining deadline and later-phase reserves. No raw payload has been
scanned in this preparation.

## Review and execution prerequisites

Require **148 fixtures, zero failures/errors/skips**: the previous 125 plus 23
focused actual-format, ENOEXEC, unexpected-error/timeout, pre-mask visibility,
parent ownership-failure and cumulative-budget cases. These are static counts;
none has executed. Python AST, embedded remote Python, shell syntax and Node
syntax checks are static only. Actual CUA proof delivery is an operational
prerequisite, not established by those checks.

Use [CUA-HANDOFF.md](CUA-HANDOFF.md) after changed-area exact-pin review. Prepare
the fresh form/command, deploy a fresh detached code directory, stage the same
three exact helpers, install the CUA helper and arm a fresh local proof bridge.
The owner must await that helper in the **same single invocation** as Submit and
actual-detail navigation. No allocation ID is guessed and no browser backend
is polled. Failure to capture complete proof produces no marker; do not repeat
r2's manual workaround as evidence that automation passed.

Only a complete compatibility artifact plus phase/import/JUnit/owned-cleanup
receipts and reconciled terminal scheduler cost establishes the CPU gate pass.
The code owner prepares the patch; the execution owner is the sole launcher.
