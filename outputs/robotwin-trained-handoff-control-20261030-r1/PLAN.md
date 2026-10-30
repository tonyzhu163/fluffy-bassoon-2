# Owner handoff controller: preparation only

This is a separate Git-pinned control artifact, not a source-driver change.
No real source process has been paused, continued, stopped or computed against
by this preparation. Root must qualify the fake-process regressions on allocated
Linux CPU resources and independently review this controller before actual use.

## Exact scope

Run only **inside the exact owned Baige allocation through its WebShell**. The
production CLI additionally requires an actual`job-*`Linux hostname.
Development SSH remains artifact/read-only access. The production CLI fixes the
old CPU supervisor's c715 artifact SHA
`492efd191e0dd1d45433d290a92c6bcd766593bfd8783e97321e7705909413bf`, actual executor
e825821 and original DB scientific lineage. Root supplies the full hostname,
PID/start ticks, exact supervisor script, runID/method, code-root and output.

Before every signal, the controller verifies Linux hostname, same UID, unique
full supervisor command identity, exact Python/script path and script SHA,
source pin/code-root/output/runID/method, current PID start ticks, and the clean
executor Git pin. Waiting uses one-second cheap fresh /proc identity/StateT
checks; it does not repeat Git, script hashing or uniqueness scans on each poll.
Full checks repeat before signals, before artifact audit and before the final
receipt. Linux pidfd signalling is mandatory to eliminate the PID-reuse
window; there is no unsafe `kill(PID)` fallback. An already stopped parent is
rejected rather than claimed as our own pause. The controller itself must run
from an exact clean Git pin supplied by root.

The only permitted signals are **SIGSTOP and optional SIGCONT to that CPU
supervisor**. No source learner, native simulator, CUDA process, process group,
other job or allocation receives a controller signal. The source child's direct
PPID, interpreter/driver/source/lineage/method/output and finite six-trial round
flags are verified. Unknown children, ambiguous/zombie identities and externally
stopped source children fail closed.

## Handoff sequence

1. Verify ownership; remember the currently owned direct source child.
2. SIGSTOP only the CPU supervisor through its checked pidfd. Confirm StateT.
3. Let its finite source child keep running to its existing round guard. Poll
   until all owned direct source children have exited or become zombies. Kernel
   wait status or the exact slice's exit receipt establishes the exit code.
4. Scan slice paths newest first. Do not trust stale `cursor.json` or select an
   older boundary when a newer potentially executed child lacks proof. A
   directory with no console and no experiment is provably before c715's Popen
   and may be noted as inert. An empty console is ambiguous and fails closed.
5. Independently verify the newest wallstop summary, bound continuation manifest,
   full checkpoint SHA/ZIP CRC, original lineage contract, empty pending PPO,
   trial count divisible by six, iteration/update counts, unchanged frozen motor,
   and the existing canonical ledger snapshot/head with zero pending/uncertain
   actions. The sole qualified source validator also checks every manifest binding.
6. Follow hash-bound corpus ancestry, sum closed raw episode action counts by
   phase/method across all roots, and require equality with actual ledger counts.
   No frames/model data are copied. Read the ledger twice across the audit;
   no live flock or Budget constructor is used, and the header is not edited.
7. Recheck StateT and no live source child, write a fsynced receipt under the
   active output's `control/` directory, and leave the CPU supervisor stopped.

Native cleanup134/139 is accepted only with those durable wallstop proofs.
A failed child, missing/non-round manifest, incomplete raw data, ledger change,
CRC/hash mismatch or unknown exit halts for root audit. ParentStateT by itself
is not a source recovery guarantee.

**Root must then audit and stop the exact old allocation through the UI before
any new writer starts.** The controller receipt explicitly sets
`permission_to_start_new_writer=false`; it neither declares the old allocation
released nor authorizes a replacement writer. An owner-release receipt can bind
this handoff and the subsequent verified UI stop separately.

Default timeout behavior leaves the parent stopped for audit. Optional
`--rollback-on-timeout` resumes only the same fully revalidated CPU supervisor
that this invocation stopped. A stale identity prevents rollback; no child is
continued or killed. Do not request rollback after starting another writer.

## Actual-use command template (root only, after qualification)

Invoke the file directly from its clean, pinned auxiliary Git worktree. Do not
copy an unpinned script into an active code checkout. The controller pin below
is the new control-artifact commit; source code-root remains e825821.

```bash
PYTHONDONTWRITEBYTECODE=1 \
/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python \
  /mowenjie/fluffy-bassoon-2/aux/CONTROL_PIN/outputs/robotwin-trained-handoff-control-20261030-r1/control.py \
  --controller-commit CONTROL_PIN \
  --expected-hostname job-zgdkrn0qqueh-master-0 \
  --supervisor-pid 452 --supervisor-start-ticks 491975770 \
  --supervisor-script EXACT_OWNED_C715_SUPERVISE_PATH \
  --run-id EXACT_OWNED_RUN_ID --method ppo \
  --code-root EXACT_OWNED_E825_WORKTREE --output EXACT_OWNED_ACTIVE_OUTPUT \
  --wait-seconds 4000
```

The example host/PID/start were a read-only observation, not authority to reuse
a stale process identity. Root rechecks each actual method's current allocation
and command immediately before invoking control. Run asynchronously in that
allocation's WebShell if needed; retain controller stdout and its receipt.
No actual-use command is run by this preparation.

## CPU-only qualification

Use a **fresh** qualification output identity such as
`robotwin-trained-handoff-control-qual-20261031-r1` after the date rollover.
`qualify.sh` requests no allocation itself and can batch root's trained-launcher
wrapper guard tests in the same actual0GPU CPU job.

The regressions spawn a fake supervisor in a new session with a finite fake
source child in its own session. They demonstrate real SIGSTOP lets that child
finish, leaves the supervisorT and prevents the next child; wrong hostname,
command fields, PID/start reject without signalling another process; timeout
rollback safely resumes only the owned parent; stale start identity prevents
SIGCONT; newest missing proof cannot fall back to the first stale boundary.
All signalled processes in these tests are owned temporary fakes. Tests do not
load a VLA, simulator, source learner, real ledger or model checkpoint.

Production hash/CRC/raw-action checks remain independently reviewed and are
not replaced by fake checkpoint fixtures. No production source control is
part of the qualification job.
