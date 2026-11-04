# 512-case paired endpoint test: policy 0 vs policy 130

**Status 2026-11-04: original jobs stopped/released for a user-authorized eight-GPU continuation.** Toaster
`job-anm7svcgngnn` and OpenDrawer `job-895ee4k7j85c` have exact step0 likelihood
parity and independently verified first paired eight-case banks/common RNG.
Owner: Codex chat `01a1a4de`; the human requested execution of this prepared plan.
A duplicate Toaster submit failed at the existing-output guard; its conservative
41GPU-s charge stays inside the same27600GPU-s all-attempt cap. Internal process
timeouts plus recorded startup and release allowances reserve27559GPU-s,
including that failure; actual scheduler release costs remain controlling.
[Execution, first-bank evidence and budget](EXECUTION.json), [first check](FIRST-CHECK.json).
The user requested acceleration in chat `01a1a512`; that chat owns the
[continuation](../robocasa-endpoint512-acceleration-20261104-r1/PLAN.md). It retains
896 complete paired policy episodes and preserves all512 cases per task.
The original jobs and incomplete-bank evidence remain immutable; do not
restart them or run the original two-directory combine against partial data.
The original all-attempt ceiling and statistical comparison remain binding.
No training or checkpoint selection.

## Why

The checkpoint-selection sweep's TEST uses 128 paired cases with Holm across
five tasks. I simulated its power from the original 64-case panels'
discordance. At the TRAIN-curve effect sizes it has little chance of detecting
a real gain:

| Task, assumed true gain | 128 cases (p<.01 / p<.05) | 512 cases |
| --- | --- | --- |
| Toaster, +8.9 pp | 13% / 34% | 81% / 94% |
| OpenDrawer, +10.9 pp (64-case size) | 31% / 58% | 99% / 100% |
| OpenDrawer, +4.6 pp (TRAIN size) | 2% / 9% | 21% / 45% |

A null at 128 would not show that PPO failed. This run fixes one comparison
per task in advance (update 130 vs policy 0), so there is no selection
penalty, and uses 4× the cases.

## Pre-declared analysis (fixed before any data)

- **Tasks:** CloseToasterOvenDoor and OpenDrawer, the two with the largest
  observed gains. They were chosen from TRAIN curves and the original panel.
  All cases here are fresh, so the choice does not bias this test.
- **Comparison:** first success within 600 controls for update 130 vs policy 0,
  paired by case seed.
- **Test:** exact two-sided McNemar. Holm across the two tasks at family
  α = .05: the smaller p must be < .025, the other < .05. An improvement claim
  also needs wins > losses.
- **Reporting:** paired case-bootstrap 95% intervals (descriptive) and success
  at end.
- **Admission:** a task that passes is a defensible teacher under the
  2026-11-04 human admission rule (coordination). The original 64-case gates
  stay as recorded. The sweep's own TEST and selection rules are unchanged.

## Design

[`PREPARED.json`](PREPARED.json) was built from the adopted sweep manifest
`SWEEP-PREPARED-R2.json` by `endpoint.py prepare`. It records the manifest
hash, the step-0 and step-130 checkpoints with hashes, the source configs and
the initial-parity fixtures.

- **Cases:** 512 per task in 64 banks of 8. Seeds are
  `11,000,000 + 10,000·ordinal + i` (Toaster 11,020,000–11,020,511; OpenDrawer
  11,010,000–11,010,511). `prepare` asserts they are disjoint from the sweep's
  DEV, TEST and qualification seeds.
- **Common noise:** both policies run each bank under the same sampler seed,
  `17,000,000 + 10,000·ordinal + bank`. Every bank checks that the
  sampler-RNG hashes are identical across the two policies.
- **Order:** ABBA, with 32 banks run 0→130 and 32 run 130→0. Adjacent banks
  share a policy, so there is one model reload per bank boundary (~64 per
  task) instead of 128.
- **Runtime:** unchanged source runtime `0988b5f9` and RLinf `8a1233c3`. The
  run imports the sweep's qualified eval-only workers (`native_eval.py`) and
  helpers without modifying them. Each run uses fixed 8-lane batches with
  strict model loads. Step-0 likelihood parity must pass before any episode.
  Each bank asserts exact initial physical and public state, 12 H50 chunks
  and 600 controls per episode. The run stops at the 200 GiB PFS floor.
  Rendering variability is allowed, as in the sweep's qualification.
- **Staging:** step 130 is on PFS. Step 0 is restored from BOS to one private
  copy, which is removed only after the task's RESULT exists.

## Budget

| | |
| --- | --- |
| Episodes | 2 tasks × 512 cases × 2 policies = 2,048 |
| Controls | 1,228,800 |
| Per task | ~2.8 h at the slowest qualified rate (70.6 controls/s) plus reloads |
| Jobs | one 1-GPU job per task, run in parallel, 230-minute limit, no automatic retry |
| Ceiling | 7.67 GPU-h |
| Expected | ~5.6 GPU-h |

This is a separate ledger from the sweep's 22 GPU-h.

Order of work: build or verify `PREPARED.json`, start `stage` on the dev host,
submit [`COMMAND.sh`](COMMAND.sh) once per task, then run `endpoint.py combine`
for the Holm decision.

## CPU checks done (macOS)

- The manifest builds from the committed sweep manifest.
- Seeds are disjoint, and ABBA gives a 32/32 balance with one reload per bank.
- `paired()` reproduces the exact McNemar on synthetic outcomes; identical
  outcomes give p = 1.
- `holm()` handles three cases correctly: both pass, step-down stops at the
  first failure, and a significant regression is rejected but is not counted
  as an improvement.

The GPU path is untested until the first bank's receipts are checked.

## Independent raw-receipt and statistics audit

Both jobs' first-bank audits and exact initial likelihood replays have passed;
the historical preparation checks above are preserved. The separate
[FINAL-AUDIT.py](FINAL-AUDIT.py), source `6c95116e`, was Git-deployed beside this
run as `FINAL-AUDIT-6c95116e.py`; it does not change the active executable
checkout. It recomputes physical/public snapshot equality, renderer diagnostic
differences, case/control identities, all 12 H50 sampler records, paired RNG
identity and actual ABBA order from the raw files. Its final mode independently
reproduces McNemar with SciPy's binomial test and verifies the bootstrap/Holm
decisions.

After both task clocks exit successfully and their staging closeouts exist,
run on allocated Linux CPU, using a new destination:

```bash
TASK_ROOT=/mowenjie/fluffy-bassoon-2
TASK_RUN=$TASK_ROOT/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1
TASK_META=$TASK_ROOT/sources/stove-drift-endpoint512-d13dc18d
TASK_PY=$TASK_ROOT/setups/robocasa365-native-20261101-r1/env/bin/python
TASK_PREPARED=$TASK_META/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1/PREPARED.json
"$TASK_PY" -B "$TASK_META/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1/endpoint.py" \
  combine --prepared "$TASK_PREPARED" --outputs "$TASK_RUN" --destination "$TASK_RUN/HOLM.json"
"$TASK_PY" -B "$TASK_RUN/FINAL-AUDIT-6c95116e.py" --root "$TASK_RUN" \
  --prepared "$TASK_PREPARED" \
  --source-script "$TASK_META/outputs/robocasa-flow-sde-endpoint-test512-20261104-r1/endpoint.py" \
  --destination "$TASK_RUN/FINAL-AUDIT.json"
```

Use `--allow-incomplete` and a distinct destination only for completed-bank
receipt checks while jobs advance; that mode makes no final statistical or
whole-run claim. Scheduler release and all-attempt cost closeout remain
required after the independent artifact audit.
