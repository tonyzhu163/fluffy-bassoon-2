# Approve25 additional GPU-hours and130GB additional PFS storage

**PROPOSED; no budget authority or launch approval.** Request only this source
phase: finish the remaining38 source-SAC streams, retain the two filled bridge
leaves and fixed8 validation streams, measure final source-actor competence,
and fit all4 AD/DPT full/static controllers at25/100/500 updates. Reviewed code
base: `3b35ade21a847c6fb63e0865714439e0622decd0`; execution uses the latest
reviewed main and an exact clean deployed pin. Existing34 allocated CPU tests,
qualified SAC continuation and two production bridges suffice; no optional
diagnostic or new qualification branch is a prerequisite.

| Additional GPU component | Maximum seconds |
|---|---:|
|8 source macro workers, each10,500 execution +60 startup |84,480|
|Fixed final source-actor checks |1,400|
|All4 offline AD/DPT fits,25→100→500 |3,000|
|Planned total |**88,880**|
|Contingency within25h/90,000s |**1,120**|

Every GPU worker requests1 GPU/16 CPU/96GiB RAM/8GiB shared memory. Eight is a
packing maximum, not an availability claim. Dispatch as allocations become
available. Macro clocks include setup, idle, failed attempts and retries;
nested child clocks overlap and are never added to the macro GPU bill. Prior
qualification, the two bridges and the **current1,800s interim25/100 fit** are
funded from the original32h and are not charged again. PPO/reset1M scaling,
all LIBERO90/Plus/Pro target reevaluations and RoboTwin are separate, unfunded
components outside this request. No further scope approval is requested here.

**130GB means ADDITIONAL PFS STORAGE, not RAM or VRAM:**120GB source suffixes
plus10GB immutable snapshots, fits and source-query artifacts. Root's03:40
Shanghai `df` measurement was1,264,536,518,656 free bytes; after130GB,
1,134,536,518,656 would remain, above the existing500GB free-space floor.
Concurrent usage is still checked by the existing collector. Preserve old raw
data, recoveries and immutable runs.

## What the existing data and new ceiling mean

TRAIN currently has40 streams,33,729 decision rows,2,332 episodes/583 K4 cases,
327,547 committed primitives and351,297 all-in stream actions. Fixed validation
has8 streams,8,202 rows,584 episodes/146 K4 cases,79,381 primitives and85,301
all-in actions; it remains import-only. Historical source overhead21,253 is
counted once. Current source **resource charge is457,851**, not dataset size.

Keep the cumulative train ceiling872,000=40×21,800, validation85,301,
historical overhead21,253 and final source checks≤20,000: cumulative resource
ceiling **998,554**, under the existing1M action cap. Remaining train-all-in
envelope is≤520,703; actual new training primitives are observed later and
cannot exceed that. At the current93.24% primitive/all-in ratio, an illustrative
estimate is≈485,500 new /≈813,000 cumulative training primitives. This estimate
is not a gate or promise. Complete K4 stops leave unused quota.

Retain CF49 bridge roots `libero40-sac-source-bridge-20261031-r1-s00-train`
(20,465 actions) and `...-o00-train` (20,677), source
`cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b`. Their1,335/1,123 remaining actions
cannot fit another K4; never cold-relaunch them. Thus38 newly packed streams
have at most518,245 usable headroom;520,703 is the conservative global envelope.
All40 goals remain admitted. This is quota preservation, not semantic filtering.

Use frozen official OpenPI215 pi0.5 LIBERO40, receipt
`ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`;
40 independent per-task SAC learners, published warmup>500 query chunks,
batch256/UTD20, H400/execute10, persistent replay/optimizer/sampler/global/motor
RNG. Complete K4 histories retain real observations, executed noise/motor
actions, rewards and boundaries.512 is the candidate case schedule, not a
completed-case claim. Report nominal and actual post-settle state hashes;
a seed alone is not a new physical case. Train states0–39 and validation40–49
stay disjoint. No90/Plus/Pro trajectories enter source pretraining.

Collection overlaps the already-funded interim fit now. Freeze one immutable
complete prefix per40 train/8 validation streams for each fitting generation;
AD uses actual chronological noise, DPT uses the current source-only actor's
labels with its full training cost. The new final generation fits AD/DPT ×
full/static, seed0,128D/2 layers/4 heads/context300/window128/batch8/LR3e-4.
25/100/500 are optimizer checkpoints of500 cumulative updates per controller,
not625 or data-size scales. A changed dataset gets a fresh immutable generation.

## Exact preparation and execution after approval

The owner reviews/deploys an exact clean pin and funds only the table above.
Use the qualified actual CPU base Python3.12/Torch2.7.1+cu128 for metadata and
capture; OpenPI motor uses its existing Python3.11 runtime, and student fitting
uses the qualified cp311 cu128 overlay. Never inject that overlay into the CPU
3.12 runtime. GPU probe/model work occurs only inside actual allocations.

```bash
TASK_PROJECT=/mowenjie/fluffy-bassoon-2
TASK_PIN=EXACT_REVIEWED_DEPLOYED_SHA
TASK_PINNED=$TASK_PROJECT/worktrees/libero40-source-scale-25h-20261031-r1
TASK_CPU=/mowenjie/fluffy-bassoon/envs/lerobot-v044-8fff0fde/bin/python
TASK_MODEL=/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python
TASK_DIR=$TASK_PROJECT/outputs/libero40-sac-production-plan-20261031-r1
TASK_PLAN=$TASK_DIR/production.json
TASK_LEDGER=$TASK_PROJECT/outputs/libero40-sac-topup-preparation-20261031-r1/ledger.json
TASK_PREFIX=libero40-sac-topup-20261031-production-r1
TASK_BUDGET=$TASK_DIR/macro-budget.json
mkdir -p "$TASK_DIR"
"$TASK_CPU" - "$TASK_DIR/overrides.json" <<'PY'
import json,sys
from pathlib import Path
base='/mowenjie/fluffy-bassoon-2/outputs/libero40-sac-source-bridge-20261031-r1'
with Path(sys.argv[1]).open('x') as f:
 json.dump({'libero_spatial:00/train':base+'-s00-train','libero_object:00/train':base+'-o00-train'},f,indent=2)
PY
PYTHONPATH="$TASK_PINNED/src" "$TASK_CPU" -m fluffy_bassoon.meta_rl.vla_source40_topup plan \
  --source-commit "$TASK_PIN" --mode production --project-root "$TASK_PROJECT" \
  --evidence-root "$TASK_PINNED/outputs" \
  --receipts "$TASK_PINNED/outputs/libero40-bulk-approved-20261030-r1/terminal-source-receipts.json" \
  --overrides "$TASK_DIR/overrides.json" --output "$TASK_PLAN"
TASK_PLAN_SHA=$(sha256sum "$TASK_PLAN" | cut -d' ' -f1)
```

After human approval, root records the90,000s phase components and creates the
funded macro record from the existing template; this proposal does not do so:

```bash
"$TASK_CPU" - "$TASK_PINNED" "$TASK_PLAN" "$TASK_PIN" "$TASK_PREFIX" "$TASK_BUDGET" <<'PY'
import json,sys
from pathlib import Path
code,plan,pin,prefix,out=sys.argv[1:]
value=json.loads((Path(code)/'outputs/libero40-sac-production-worker-budget-20261031-r1/budget-template.json').read_text())
value.update(funding_status='root_reconciled',source_commit=pin,plan_sha256=json.loads(Path(plan).read_text())['plan_sha256'],
 campaign_prefix=prefix,campaign_root='/mowenjie/fluffy-bassoon-2/outputs/'+prefix)
with Path(out).open('x') as f:json.dump(value,f,indent=2)
PY
# One command per approved worker allocation, WORKER=0..7; fresh limit2h55m.
bash "$TASK_PINNED/scripts/diagnostics/vla_source40_topup_worker.sh" \
  "$TASK_PIN" "$TASK_PREFIX" "$TASK_PLAN" "$TASK_LEDGER" "$WORKER" "$TASK_BUDGET"
# Before any retry allocation, read the remaining scheduler ceiling; then use --resume.
PYTHONPATH="$TASK_PINNED/src" "$TASK_CPU" -m fluffy_bassoon.meta_rl.vla_source40_topup_worker remaining \
  --root "$TASK_PROJECT/outputs/$TASK_PREFIX/worker-$WORKER"
```

Capture/assemble complete prefixes on allocated CPU while collection proceeds,
and freeze the final generation before its GPU fit. These are existing data
pipeline steps, not additional diagnostic prerequisites. Precreate the fit
run's snapshot/assembly so their copy cost does not consume its GPU window:

```bash
TASK_FIT_RUN=libero40-source-topup-final500-20261031-r1
TASK_FIT_ROOT=$TASK_PROJECT/outputs/$TASK_FIT_RUN
PYTHONPATH="$TASK_PINNED/src" "$TASK_CPU" -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly capture \
  --plan "$TASK_PLAN" --plan-sha256 "$TASK_PLAN_SHA" --ledger "$TASK_LEDGER" --output "$TASK_FIT_ROOT/snapshot"
TASK_ADMISSION=$TASK_FIT_ROOT/snapshot/admission.json
TASK_ADMISSION_SHA=$(sha256sum "$TASK_ADMISSION" | cut -d' ' -f1)
PYTHONPATH="$TASK_PINNED/src" "$TASK_CPU" -m fluffy_bassoon.meta_rl.vla_source40_topup_assembly assemble \
  --admission "$TASK_ADMISSION" --admission-sha256 "$TASK_ADMISSION_SHA" --output "$TASK_FIT_ROOT/assembly"
# One approved GPU allocation,49min;3000s cumulative incl60s startup/retries.
bash "$TASK_PINNED/scripts/diagnostics/vla_source40_topup_training.sh" \
  "$TASK_PIN" "$TASK_FIT_RUN" "$TASK_PLAN" "$TASK_PLAN_SHA" "$TASK_LEDGER" 2940 3000
```

After all38 source suffixes are terminal, the final source-only check admits
the exact two retained producer lineages plus38 current producers. It uses
the same fixed8 source goals × states40/41 × native/mean/sample =48 fresh
queries,≤20k actions, zero SGD/replay insertion; actors remain candidates.

```bash
TASK_FINAL_RUN=libero40-sac-topup-final-competence-20261031-r1
TASK_ALIAS_DIR=$TASK_DIR/final-competence-alias
PYTHONPATH="$TASK_PINNED/src" "$TASK_CPU" -m fluffy_bassoon.meta_rl.vla_source40_topup_competence prepare \
  --plan "$TASK_PLAN" --ledger "$TASK_LEDGER" --output "$TASK_ALIAS_DIR" \
  --run-id "$TASK_FINAL_RUN" --source-commit "$TASK_PIN" --allow-retained-parents
TASK_ALIAS_SHA=$(sha256sum "$TASK_ALIAS_DIR/alias.json" | cut -d' ' -f1)
# Exact-pin worktree at $TASK_PROJECT/worktrees/$TASK_FINAL_RUN; approved22min GPU allocation.
bash "$TASK_PROJECT/worktrees/$TASK_FINAL_RUN/scripts/diagnostics/vla_source40_sac_diagnostic.sh" \
  "$TASK_PIN" "$TASK_FINAL_RUN" "$TASK_ALIAS_DIR/alias.json" "$TASK_ALIAS_SHA" 1340 '' topup
```

All resumes retain original source/manifest identities, full recovery and
cumulative time/action charges; never grant another full slice after failure.
Limits take precedence. Report actual completed source quotas, unused case
bank/headroom, query outcomes and exact policy/full-recovery hashes; retain any
partial work rather than silently expanding. This phase supplies source data
and frozen controllers. High sample efficiency and topline robotic performance
remain questions for the separately budgeted target comparison.
