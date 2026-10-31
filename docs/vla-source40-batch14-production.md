# Proposed cumulative PPO source continuation

Executable preparation only. Neither the source-SAC proposal nor an old32h
reservation authorizes this separate proposed44h/64GB per-PPO-learner phase.
The launcher rejects a proposal without an exact approved funding record.
Existing source runs, raw histories and full recoveries remain immutable.

## Exact lineage and cumulative source actions

Read-only Baige metadata checks confirmed the historical action receipts:

| Learner | Legacy train/validation/qualification prefix | Qualified native prefix | Cumulative prior | Native stream ceiling including its old prefix | Maximum NEW production actions | Final source-validation reserve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL |195,964|9,272|205,236|774,036|764,764|30,000|
| EPISODE/reset |196,776|0|196,776|773,224|773,224|30,000|

Each row satisfies prior + NEW production + final source validation =1,000,000.
All settling, qualifications, uncertain RPC intents and retries count. Shared
historical qualification IDs remain deduplicable across a joint resource report;
they are conservatively charged to each learner's individual ceiling. No
LIBERO90/Plus/Pro data enters optimizer input.

FULL starts from the explicit qualified native recovery at
`/mowenjie/fluffy-bassoon-2/outputs/libero40-batch14-warm-qualification-20261031-r1`,
producer `376415e6915399436cb3fea214c50bd3e41f32d5`, recovery SHA
`c689a2ddb190510ba09452ad9c755446bf605351d22b41017c75187834e90ee0`.
The transition preserves **every non-contract checkpoint leaf**: native update1,
cursor14, policy/Adam, sampler, global and typed motor RNG, counters and pending
batch state. Existing native journals are copied with byte checksums, so the9,272
prefix remains inside the native stream cap exactly once. Old raw banks stay at
their immutable original paths. Only explicitly reviewed phase/source metadata
changes; frozen motor/features/noise/task/PPO semantics must match exactly.

EPISODE uses the already qualified generic R3 warmstart in episode mode, preserving
its original policy/Adam/sampler/global/typed-motor RNG. The legacy task cursor and
all exposure remain embedded unchanged; the explicitly new native task cycle
starts at0, as in the qualified full warmstart. It does not cold-start the policy.

Both use the existing trainer:14 complete K4 trials per global batch, native
220/280/300/520 horizons, outcome-independent all40 task cycle, real train-state
IDs0..39, two global Adam steps per batch, LR1e-4, valid-token loss weighting,
width128/layers2/heads4/context300 and seed2. The100-native-update target includes
FULL's already completed update1; it is not100 additional updates. Whole-batch
worst-case action admission, actual costs and time can stop either arm earlier.
Report actual complete/pending trials, states, task coverage and updates; scene
seeds are not new physical states and the state bank need not be exhausted.

## Preparing metadata plans

Run the metadata-only CLI from the exact reviewed code on Baige. It reads the
existing receipts and creates a new immutable proposal; it runs no model/test or
simulator and creates no allocation. The preserved producer worktree is needed
because the qualified full recovery binds the history file's exact absolute path.

```bash
TASK_HISTORY=/mowenjie/fluffy-bassoon-2/worktrees/libero40-batch14-warm-qualification-20261031-r1/outputs/libero40-batch14-warmstart-plan-20261031-r1
python -m fluffy_bassoon.meta_rl.vla_source40_batch14_production plan \
  --legacy-history "$TASK_HISTORY/history-full.json" \
  --legacy-history-sha256 e02db1d4de582c9b5694beaacc80f130206c4b7e1cc78eb49cdb12a0760396a6 \
  --mode full --native-root /mowenjie/fluffy-bassoon-2/outputs/libero40-batch14-warm-qualification-20261031-r1 \
  --output /absolute/new/plan-full.json
python -m fluffy_bassoon.meta_rl.vla_source40_batch14_production plan \
  --legacy-history "$TASK_HISTORY/history-episode.json" \
  --legacy-history-sha256 cfa398042213a54e99b65fc77507e817554e1dcf007287b68b5c57df2463c39e \
  --mode episode --output /absolute/new/plan-episode.json
```

## Funding and execution contract

Proposed additional GPU time per learner is44h =154,800s production +3,600s final
source validation. The latter is held separately and cannot be borrowed by
collection. Its action reserve covers8 fixed source goals × states40/41 × K4
(21,760 worst-case actions including settling),16 native episodes (5,440), plus
2,800 retry margin. Shared native queries get one unique joint charge ID if
actually shared. This adds no validation launcher or new scientific method.
The3,600s allowance is conservative against the qualified warm batch's ~10.2
resource actions per charged second; it remains a ceiling, not promised speed.
64GB per learner is a storage allowance, not a filesystem quota or permission to
prune old recoveries. Check current shared free space before launch.

After human approval, root supplies a funding JSON with `status="APPROVED"`, the
actual authorization, exact `run_id`, `source_commit`, `plan_sha256`, `mode`,
`production_gpu_seconds=154800`, `final_source_validation_gpu_seconds=3600`,
`maximum_total_gpu_seconds=158400`, and the exact row's
`maximum_native_stream_actions`. Preparation does not generate this approval.

Inside each funded exclusive GPU allocation:

```bash
bash scripts/diagnostics/vla_source40_batch14_production.sh \
  EXACT_COMMIT FRESH_RUN /absolute/plan.json PLAN_SHA /absolute/funding.json 21540
# Subsequent allocation, SAME run/source/plan/funding, original remaining time:
bash scripts/diagnostics/vla_source40_batch14_production.sh \
  EXACT_COMMIT SAME_RUN /absolute/plan.json PLAN_SHA /absolute/funding.json 21540 --resume
```

The dedicated production clock records all setup/failure/retry time, reserves60s
scheduler allowance per slice and shortens the final slice to remaining time.
An unclosed attempt stays fully charged. Normal slices reserve21,540+60 seconds,
so a six-hour scheduler window suffices; no slice resets the cumulative154,800s
production clock. Soft stop occurs between complete trials;300s remain before
the outer timeout for the longest native K4 and cleanup. The existing trainer
preserves partial complete batches and frozen behavior traces, never updates a
partial batch, and retains the ready checkpoint if an optimizer burst interrupts.
It restores all numerical state before collection and verifies exact operational
fingerprints through the existing warmstart restoration hook. Initial migration
publishes the new learner atomically after byte/fingerprint checks. A failed
migration's staging files remain preserved for diagnosis.

Actual allocation probing, the qualified OpenPI3.11/cu128 runtime and new focused
phase/clock tests are included. Existing50-test/JAX/batch14 qualifications are
reused; there is no additional optional diagnostic launch gate. Tests for this new
glue still require allocated Linux before production launch. No launch, reservation
or larger scientific conclusion follows from this preparation.
