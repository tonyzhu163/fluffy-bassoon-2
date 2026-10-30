# Cumulative source40 SAC top-up

Prepared2026-10-31. **Executable preparation, not a launch approval.** Full
top-up remains proposed at24 NEW GPUh /120 GB new storage. The proposed one-goal
migration qualification is at most1,000 cumulative GPU-seconds and5,000 NEW
simulator actions, unfunded until the owner assigns reclaimed resources.

The protocol retains frozen official OpenPI40 pi0.5, H400 / execute10 /32D
tied noise, published SAC warmup>500 chunks, batch256 and UTD20. All40 source
train learners and the fixed eight independent source-validation learners keep
their own complete replay/optimizer/RNG/case cursor. No90/Plus/Pro trajectory
enters source pretraining. No source goal or case is selected by its outcome.

| Cumulative source SAC cost | Actions |
| --- | ---: |
|Selected all40 train + fixed8 validation |411,578 |
|Earlier technical source train/validation qualification |12,944 |
|Failed typed-key qualification settling |10 |
|Frozen source-SAC native/mean/sample controls |7,540 |
|Collector speed qualification, terminal-capacity alias counted once |759 |
|**Already charged source floor** |**432,831** |
|All40 cumulative train ceilings:40×21.8k |872,000 |
|Fixed8 validation:import actual terminal cost, no new K4 |85,301 |
|Historical extras above, counted once |21,253 |
|NEW final-teacher competence:48 fixed native/mean/sample queries |20,000 |
|**Worst-case cumulative total / unassigned** |**998,554 /1,446** |
|Maximum new train-stream actions, qualification prefix inside |545,723 |

The failed competence-r1 produced no reset/action receipts:zero actions; its
78 GPU-seconds remain in compute accounting. Native PPO's22,753 source-validation
actions and target episodes are different components and are not recharged to
this SAC learner. Qualifications/controls are cost evidence, not training data.
The code binds the48 terminal indices/contracts/recoveries and these cost
receipts; mutable compute-ledger provenance is snapshotted by component.

All original512-case banks, case IDs, scene seeds, split and chronological
cursors remain intact: train nominal states0–39, validation40–49.512 is a
candidate bank, not requested/exhausted collection. A new case starts only if
its full K4 worst-case1,640 actions fit the stream's cumulative quota. Case and
episode/RPC costs include unresolved intents, repeat qualification and settling.
All40 train streams have substantial headroom. The eight terminal validation
streams are import-only, charged at actual85,301 rather than treating unused
96k reservation as spendable. Retain their existing histories instead of
spending pointless qualification resets. Metadata reports
admitted streams, actual complete cases and remaining bank entries separately.

Plans have no resource authority. A single canonical action/compute ledger
owns one current leaf per task/partition. A new leaf replaces its exact terminal
parent; copied prefixes are not added as new interactions. A competing leaf is
rejected. A production plan following qualification must override Spatial00's
parent with the terminal qualification output, importing its cost/state once.

Migration copies the parent experiment byte-for-byte to a fresh output. Original
raw files/full recoveries/milestones remain untouched. Parent mutable metadata
and full latest recovery are also preserved under an ancestry SHA directory.
Only the new recovery's scientific contract/cap changes. A fingerprint across
every other serialized leaf must match before any SAC interaction/update. Allocated
restore then independently compares actor/critic/target/temperature parameters,
all optimizers, full replay, update counter, replay sampler, Python/NumPy RNG,
and typed motor RNG. Every new episode must use learned SAC throughout with
fixed update counter/stage; no new Gaussian warmup is inserted. New resets
require both nominal-state and measured post-settle simulator-state hashes.
Seeds alone are not reported as new physical states.

The existing collector owns full physical journaling and atomic full recovery
after every completed episode. Partial cases resume their original case/seed;
failed prefixes remain charged and retained. Time/storage stops happen before
a new whole case, and final complete prefixes are exported even at a time stop.
Sparse immutable full checkpoints/AD+DPT prefixes remain every16 cases plus
final. `candidate.json` points to a hash-bound full source-actor checkpoint;
it does not label the actor an expert. New storage has a3 GB/stream cap with
1 GB whole-case/atomic headroom and a500 GB shared-PFS free-space floor. No
old artifact is removed or repacked to obtain that headroom.

AD targets are the actual inherited+new chronological noises and original
teacher clocks. DPT relabels the same full history with the current final
**source-only** actor. Each export records cumulative stream action cost and
global-plan ancestry; the21,253 shared overhead is counted once globally.
Never concatenate overlapping cumulative prefixes. A DPT subset using the
final actor retains that actor's full training cost. Qualification additionally
requires a new complete K4, actual optimizer advancement/finite metrics, final
paired export and an independent post-update full recovery/motor-RNG restore.
Frozen motor parameter bytes are checked before and after collection.

Preparation commands, on Baige with the reviewed clean source in `$PINNED`:

```bash
PYTHONPATH="$PINNED/src" "$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_topup plan \
  --source-commit "$COMMIT" --mode qualification \
  --project-root /mowenjie/fluffy-bassoon-2 --evidence-root "$PINNED/outputs" \
  --receipts "$PINNED/outputs/libero40-bulk-approved-20261030-r1/terminal-source-receipts.json" \
  --output "$NEW_PLAN_DIR/qualification.json"
```

After the owner funds one allocation and pins the same commit, the bounded
qualification invocation is:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_topup.sh" "$COMMIT" \
  libero40-sac-topup-migration-20261031-r1 "$NEW_PLAN_DIR/qualification.json" \
  libero_spatial:00/train "$CANONICAL_TOPUP_LEDGER" 940 1000
```

The940-second slice plus60-second startup allowance shares one1,000-second
cumulative clock including setup/tests/retries. The driver deadline precedes
outer timeout. Unclosed attempts retain their full hold; actual overruns are
reported, never truncated. There is no scheduler submission in these scripts.
Focused tests run inside that actual allocation; none ran on macOS.

For a later approved production phase, write an overrides JSON mapping
`libero_spatial:00/train` to the terminal qualification output root, then repeat
`plan --mode production --overrides ...` with a fresh plan filename. The NEW
production request consists of eight workers, each at most2h55m actual execution
and a60s startup allowance, plus a final check of at most1,400s:
**8×(10,500+60)+1,400=85,880 GPU-seconds**, leaving520s unassigned under24h.
The earlier qualification is charged once to its original32h pool; it is not
another NEW GPU-time request. This describes a ceiling, not eight available
GPUs or a promise that every source quota finishes within it.

After approval, bind the exact production plan/pin/prefix in a root-funded
`source40_sac_production_macro_budget_v1` budget. An unfunded template is in
`outputs/libero40-sac-production-worker-budget-20261031-r1/budget-template.json`.
For worker0–7, in an already approved allocation:

```bash
bash "$PINNED/scripts/diagnostics/vla_source40_topup_worker.sh" \
  "$COMMIT" "$PREFIX" "$PRODUCTION_PLAN" "$CANONICAL_TOPUP_LEDGER" "$WORKER" "$MACRO_BUDGET"
```

The immutable macro clock includes deployment checks, setup, child work,
inter-child gaps/idle, failures and teardown. Each resume gets a fresh attempt
directory while preserving assignment bytes and prior outcome receipts. Before
requesting a retry allocation, read its **remaining** scheduler duration:

```bash
PYTHONPATH="$PINNED/src" "$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker remaining \
  --root "/mowenjie/fluffy-bassoon-2/outputs/$PREFIX/worker-$WORKER"
```

Request at most the returned `maximum_new_scheduler_seconds`, then add
`--resume` to the same worker command. It does not reset a10,500s deadline.
Unclosed attempts remain fully charged and require explicit terminal
reconciliation, rather than assuming they consumed no GPU time.

Every train stream retains its own original2,100s cumulative child clock and
full partial-case recovery. The wrapper computes the next slice from **both**
remaining child and macro time; it never hands a closed child a fresh2,040s.
Only a successful terminal, quota-fulfilled or bank-exhausted stream with exact
plan/contract/full-recovery/candidate evidence is skipped. `completed=true`
alone, a wall stop, exhausted child time or a partial case is insufficient.
Failed/unfulfilled streams stop the worker for diagnosis and remain visible
alongside unattempted assignments. A macro receipt is not an all40 completion
or512-case-bank exhaustion claim.

The canonical child `time_hold` aggregate still has its independent86,400s
ceiling:40×2,100 + up to1,000 historical qualification +1,400 final check.
Those conservative nested child charges overlap their allocated macro time;
**never add them to the macro GPU bill**. Both bounds must pass. Macro setup/
idle is billed even when it produced no child action. Historical qualification
cost is imported once and not re-reserved as a new1,000s grant. No packing
speedup or automatic scientific/resource expansion is assumed.

NEW final-teacher competence is reserved inside this same1M cap:the same eight
predeclared source goals, states40/41, native/mean/sample,48 physical queries,
19,680 nominal worst-case actions and a20k envelope. It must use the new final
train actors with zero SGD/replay insertion; old16/16 controls do not establish
new actor competence or expert quality. The existing qualified per-task SAC
diagnostic performs the queries; a top-up alias binds its20k to this canonical
ledger instead of borrowing the old PPO-validation remainder again.

After all40 new train actors are terminal, prepare that alias with:

```bash
PYTHONPATH="$PINNED/src" "$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_topup_competence prepare \
  --plan "$PRODUCTION_PLAN" --ledger "$CANONICAL_TOPUP_LEDGER" \
  --output "$NEW_FINAL_ALIAS_DIR" --run-id "$NEW_FINAL_RUN" --source-commit "$FINAL_COMMIT"
```

Prepare a clean worktree for that final run at the same reviewed commit, then
the existing qualified launcher accepts the new budget alias:

```bash
bash "$FINAL_PINNED/scripts/diagnostics/vla_source40_sac_diagnostic.sh" \
  "$FINAL_COMMIT" "$NEW_FINAL_RUN" "$NEW_FINAL_ALIAS_DIR/alias.json" \
  "$FINAL_ALIAS_SHA256" 1340 '' topup
```

The1340-second slice plus60 startup seconds shares the same1,400-second final
check hold, inside24h. The query implementation, fixed cases, raw/recovery
format, no-SGD checks and physical reset comparison are unchanged.
The final execution pin is separate from each immutable producer pin, so the
owner can review the latest main before launching the final evaluation.

Read-only progress/cost command:

```bash
PYTHONPATH="$PINNED/src" "$MODEL_PYTHON" -m fluffy_bassoon.meta_rl.vla_source40_topup status \
  --ledger "$CANONICAL_TOPUP_LEDGER"
```

Passing the bounded qualification is a runtime gate, not evidence that the
current source distribution teaches adaptation. Baseline/teacher/distribution
results still determine whether to approve the24h source top-up. The broader
122h proposal remains separate and unapproved.
