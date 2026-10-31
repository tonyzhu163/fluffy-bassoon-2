# Production-native RoboTwin vector candidate

This integration starts from the scalar contracts at `db0184089fa88a5f96c153f9b80029bb91b77def`.
It does not edit `robotwin_noise.py`, replace its capture helpers, load another
motor, change precision, or authorize migration of an active run.

`ProductionRobotwinVectorEnv(scalar_env, budget, run_id, lanes, corpus_writer,
output_dir=run_output)` runs process-isolated native simulator lanes around the
existing parent motor. Every motor encode/decode uses the original scalar B1
method and its saved lane cache. Calling-thread scene ownership, reset retry
limits, optional qualified final capture, sparse feedback, actions, rewards and
per-task native episode limits remain the existing scalar implementation.
The subprocesses contain no motor and exchange CPU NumPy arrays only.
Before an admitted group starts, an idle parent scene left by preflight or
scalar evaluation is closed and its stale prefix/raw cache released. This
requires no pending scalar/lane corpus decision. The exact shared motor stays
loaded; later scalar evaluation can reset a fresh scene normally. Idle parent
render systems are not retained just to increase device memory occupancy.

The driver chooses physical lane count within its unchanged global batch. The
backend accepts six concurrent trials, and does not prefetch future policies or
expand the training batch. Multiple independent driver/model instances require
their own ledgers, corpus handles and outputs. Actual whole-device memory is
recorded at construction and group boundaries; those samples include other
processes and are not a peak or a claim that a selected lane count is optimal.

## Group lifecycle

1. Set each lane's original behavior/corpus context, including immutable policy
   checkpoint identity. Call `admit_group(tasks, episode_seed_lists)` before
   reset. It checks the live exclusively locked ledger, complete-trial native
   worst-case allowance and one behavior version. The durable group plan and
   ordinary-resume fence precede all native reset/step RPCs. No nested escrow
   double-charges actions; individual calls use the existing campaign ledger.
2. `reset_batch` and `reset_lanes` accept only the admitted seed/task sequence.
   A durable zero-policy-command reset intent precedes dispatch. Native reset
   receipts must confirm zero policy commands; internal settling physics is
   recorded as not separately instrumented, exactly as in the scalar runtime.
3. Supply policy metadata with `record_decision_metadata(lane, metadata)`.
   Before any active-lane step is dispatched, every lane has an exact upper
   reservation and a durable original corpus intent. Complete returned counts
   all resolve before any motor encoding or corpus serialization can fail.
   Native arrays and flags are saved before outer learning flags are finalized.
4. `finish_group()` requires every planned episode and corpus decision to be
   complete. It leaves `physical_complete_uncheckpointed` and the resume fence
   intact. The driver must first fsync its coordinator with pending rollouts,
   learner/sampler/RNG/schedule/counters, `parallel_collection_group`, and the
   exact `campaign_journal_fence` (canonical root, sequence/hash, charge/count).
5. `acknowledge_checkpoint(path)` checks that payload against the live group
   and campaign. It fsyncs the checkpoint, records its hash in the canonical
   group journal, then durably clears the fence. Only then may another group
   or policy update begin. An arbitrary older checkpoint cannot acknowledge.

`cancel_group(reason)` clears a fence only with proof of zero reset/step RPCs.
`can_restore_unstarted_group()` provides the caller's explicit rollback proof.
After dispatch, failures remain unresolved and block ordinary resume. Returned
CPU replies are durably retained for forensics; unanswered reservations remain
charged. Driver checkpoints must embed the failure metadata and reject copied
stale checkpoints using the canonical campaign/group journals before GPU work.
Neither this backend nor its process pool automatically resumes or retries.
Cleanup affects only its own child processes; the parent owns motor and ledger.

`group_recovery_metadata()` exposes the canonical audit path, state, physical
counts and timings. Group wall time includes native resets, parallel RPCs,
unchanged B1 motor work, synchronous raw corpus persistence and budget/journal
I/O. CPU fakes establish interface and failure semantics; allocated GPU checks
must separately exercise the actual production capture, end-to-end timings,
device memory, exact PPO replay and unchanged optimizer update before adoption.
