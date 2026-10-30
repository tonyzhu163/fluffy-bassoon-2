# Production SAC worker budget and exact resume

Prepared only: no GPU availability claim, allocation, canonical action/time hold
or24h approval is created here. The source collector, `vla_source40_topup.py`
and all active qualification worktrees remain unchanged.

The proposed NEW component is at most **85,880 GPU-seconds**:
eight workers×(10,500s execution+60s startup allowance), plus a final source
teacher check≤1,400s. This leaves520s under86,400s. Historical migration
qualification time remains charged once to the original32h pool, not requested
again as NEW compute. The existing canonical nested child-hold sum still must
fit86,400s including historical qualification; it overlaps macro time and must
not be added to macro charges. No final check is launched by the worker.

Before launch, root must bind the exact clean source, production plan content
SHA and fresh prefix/root in `budget-template.json`, then set its status to
`root_reconciled` after approval and reconciliation. The budget remains
immutable during execution. The production plan must retain all40 train
streams; each worker owns the original fixed stride of five streams. Nothing
is reassigned using observed success or update counts.

In an already funded allocation, request at most2h55m for a fresh worker:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_topup_worker.sh" \
  "$SOURCE" "$PREFIX" "$PRODUCTION_PLAN" "$CANONICAL_LEDGER" "$WORKER" "$MACRO_BUDGET"
```

`WORKER` is0–7. Source/plan/funding/assignment/ledger identity is pinned in
`outputs/$PREFIX/worker-$WORKER/worker-time.json`. The clock starts before
deployment checks and includes setup, failed preflight, child execution,
between-child gaps/idle and teardown. Each attempt adds its own60s conservative
scheduler allowance; the original10,560s worker ceiling never increases.
Time outside an allocation, between ended and later resumed attempts, is not
invented as billed GPU time.

Read this before requesting a retry allocation:

```bash
PYTHONPATH="$PINNED/src" "$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker remaining \
  --root "/mowenjie/fluffy-bassoon-2/outputs/$PREFIX/worker-$WORKER"
```

Request no more than `maximum_new_scheduler_seconds` and add `--resume` to the
same worker invocation. An active/unclosed attempt blocks automatic restart;
its full original reservation remains charged until explicit terminal
reconciliation. A partially spent worker never receives a fresh2h55m clock.

The next child invocation uses the minimum of its remaining original2,100s
clock and the macro's remaining time, preserving startup/cleanup headroom.
For example, a child already charged560s can reserve at most1,480s execution
plus60s allowance, not a fresh2,040s. Its existing full partial-case learner,
replay, optimizer, sampler/global/motor RNG are consumed by the existing
`--resume` path. No collector/RNG/data-order rule is changed.

Only a successful terminal fulfilled stream is skipped. Proof includes the
same source/plan/canonical leaf, a complete-case boundary, qualified export,
actual quota/bank exhaustion and exact full-recovery/candidate hashes. A
`completed=true` flag alone does not qualify a wall stop. An exhausted or
failed partial stream stays unfulfilled and stops that worker for diagnosis;
there is no hidden per-stream budget reset or blind next-goal launch.

Reviewable receipts:

- `assignment.json` and `assigned.txt` are immutable across resumes.
- Every invocation has a fresh `attempts/<time>-<ordinal>/` directory.
- `decision-*.json` records fulfilled/launch/blocked and both remaining clocks.
- `child-outcome-*.json` preserves each actual child result independently.
- `macro-outcome.json` records cumulative macro GPU charge, its ceiling and
  fulfilled/unfulfilled/unattempted assignments. Old outcome bytes are retained.

A five-stream assignment or finished allocation is not an all40 completion
claim or a512-case-bank exhaustion claim. The canonical source ledger and
actual per-stream indices remain authoritative for physical actions and
training/exposure. Focused metadata tests are prepared and included in the
allocated preflight; none were run on Mac/dev.
