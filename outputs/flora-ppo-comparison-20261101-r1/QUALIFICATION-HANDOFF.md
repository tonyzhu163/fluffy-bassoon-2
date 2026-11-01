# Task44 source handoff

Source `a3b6111a` is **CPU-qualified only**: allocated Baige
`job-dtz06yz20uw3` passed all67 cases without failure/error/skip. Owner evidence
`19bf173d` records97 inclusive CPU-worker-seconds, zero GPU/actions/models;
50 scheduler-seconds is a separate clock. [Byte-preserved publication and raw
JUnit references](QUALIFICATION-CARRY.json). The source author performed only
static/saved-artifact checks on macOS. `finish_assembly` remains the sole
allocation, qualification and launch owner. Human steering now selects the
published RLinf40 few-shot PyTorch release directly. Q1 conversion retry is held;
its failed pre-wrapper cost remains in the owner ledger.

## Completed CPU qualification and retained command

Inside a fresh allocated Linux CPU worker, with the existing qualified PPO
environment and exact RLinf `c605b9d55ed93a180119b099e6f836abff293b08`:

```sh
export PYTHONPATH="$SOURCE_WORKTREE/src:$RLINF_CHECKOUT"
export EMBODIED_PATH="$RLINF_CHECKOUT/examples/embodiment"
"$PPO_PYTHON" -B -m pytest -q "$SOURCE_WORKTREE/tests/test_rlinf_flora_ppo.py" \
  --junitxml="$FRESH_CPU_OUTPUT/junit.xml"
```

This selector covers continuous labels, full-horizon masks, simulator-truth
isolation, all-or-none/stale label refusal, actual author GAE equivalence for S,
pending model/Adam identity, plain logs/stopping and real Hydra composition for
1/2/4/8 workers. It constructs no model, scorer or simulator. Retain the actual
JUnit, import/deployment hashes and real scheduler charge. Do not import the old
RoboTwin 274-test qualification as a LIBERO pass. A proposed finite CPU reservation
is 300 worker-seconds, 8 CPUs, 32GiB, zero GPU; the execution owner binds actual
allocation/clock and may adjust ordinary CPU execution as needed.

## Runnable native episode path

After actual published-model artifact receipt, normalization and allocation/probe are
bound, smoke uses **one** rollout/env GPU and one fresh full-H400 episode, no
actor/optimizer initialization. Baseline uses 64 episodes from nominal states
0..39. Supply a fresh output and attempt for each; preserve the same canonical
phase ledger and immutable published model.

```sh
"$PPO_PYTHON" -B "$SOURCE_WORKTREE/scripts/diagnostics/rlinf_flora_ppo_entry.py" \
  --phase smoke --arm S --world 1 --profile "$ACTUAL_PROFILE" \
  --model "$PUBLISHED_RLINF40" --norm "$BUNDLED_NORM" \
  --output "$FRESH_OUTPUT" --exchange "$PHASE_EXCHANGE" \
  --monitoring "$ISOLATED_MONITORING" --accounting "$COMMON_ACTION_ROOT" \
  --attempt "$FRESH_ATTEMPT"
```

For baseline replace `--phase smoke` with `--phase baseline`; 1/2/4/8 layouts
are composed, actual capacity is unmeasured. The entry verifies source/RLinf Git
pins and actual published safetensors bytes against release
`RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`:
weight SHA `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`,
7,473,091,464 bytes; bundled norm SHA
`dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`, 4,507 bytes.
It refuses stray `full_weights.pt` and never asserts C565/JAX identity. Official
c605 loads the published legacy PyTorch keys in memory. Keep model H10/padded32,
physical7 and the published PPO 3-denoise/execute5 sampler; no offline converter.

`initializer_artifact_receipt` replaces the historical conversion receipt.
It binds `completed:true`, `repository_id`, `revision`, absolute
`model_directory`, `normalization_sha256`, `model_action_dim:32`,
`action_horizon:10`, and `files` with absolute `path`/`sha256`/`bytes` for the
published weights and bundled norm. The template contains no fake receipt.

Training's actual `loading_qualification` replaces the old `loading_parity`.
It binds `completed:true`, the same repository/revision, `model_sha256`,
`normalization_sha256`, `backbone_missing_keys:[]`, `unexpected_keys:[]`, and
`value_head_seed:42`. This is an actual native strict-load result, not JAX
parity. The fresh64 headroom/AUROC gate remains. No source loading or baseline
result is claimed here.

The affected CPU selector is:

```sh
"$PPO_PYTHON" -B -m pytest -q "$SOURCE_WORKTREE/tests/test_rlinf_flora_ppo.py" \
  -k 'initializer or compose_actual_official_config' \
  --junitxml="$FRESH_CPU_OUTPUT/initializer-junit.xml"
```

It has 13 prepared cases (nine initializer ABI cases plus four real compositions).
The full67 selector actually ran once on a3 and passed. The earlier a894 job
was never created; no invented58-case carry was used. The affected13-case
command is retained for future affected changes, not an instruction to rerun
passed code. Existing a3 GPU qualification remains immutable.

## Separate Flora process

Pin NVIDIA/Flora code `129a2ada74c526ee4692f3d1b111bea6ca19d5b7` and model
`031299e46e452fc0cedb4e0975eb9fa95fc37783` in the separate Python3.12 environment.
The actual staged model receipt binds `completed`, `model_revision`, absolute
`model_directory` and the hashed `files` inventory. A local directory does not
use Flora's `--revision`; identity comes from this actual staged receipt.

```sh
"$FLORA_PYTHON" -B "$SOURCE_WORKTREE/scripts/diagnostics/rlinf_flora_ppo_score.py" \
  --mode smoke --code "$FLORA_CODE" --model "$FLORA_MODEL" \
  --model-receipt "$STAGED_MODEL_RECEIPT" --episodes "$EPISODES_JSON" \
  --output "$FRESH_SCORE_OUTPUT" --deadline-unix "$ALLOCATION_DEADLINE"
```

For the 64-episode baseline, use `--mode baseline` and add
`--baseline-monitor "$BASELINE_MONITOR_DIRECTORY"`. It emits the fixed mapping
and AUROC/headroom go/no-go receipt without fitting a threshold. The official
terminal CLI initially uses TP1/max-seqs1/context16384/utilization0.78. Actual
memory and latency are unqualified; tune execution batching within measured
capacity without shortening the prescribed videos/context.

For training, `--mode train --exchange-root "$PHASE_EXCHANGE" --max-batches 13`
consumes only F's bounded `label-request.json` files until service closure or the
actual deadline. No simulator-truth argument is accepted. Partial scorer output
is preserved and never overwritten; missing labels hold the on-policy batch.

## Training and recovery

Use the same native entry with `--phase train --arm S` or F and actual training
profile. Both use official critic/GAE/Adam and the published effective batch2048,
one optimizer epoch, 32env x4 rounds. Initial microbatch32 gives accumulation16
on4 actors or8 on8 actors. No measured throughput or GPU-memory fit is claimed.
An initial `FLORA_PPO_ITERATIONS=1` is a **charged first iteration of the arm**,
not a separate unfunded benchmark: retain its full boundary and resume it rather
than starting another learner. The first reservation uses a conservative finite
iteration estimate; replace that estimate with measured rollout/label/update/
50-case-eval/closure time before projecting more iterations.

World4 and8 use the same **56** evaluation slots: 50 scored cases (ten DEV
placements40..49, five predeclared draws each), plus six explicitly unscored
padding episodes. All56 are physical and charged: `(128+56)*415=76,360`
estimated actions per complete iteration. The default12-iteration maximum plus
common preparation fits the 1M ceiling; never silently run13 on these layouts.
World1/2 can execute exactly50 evaluation slots. Neither padding nor repeated
placements are additional independent scored cases.

Before GAE, the runner freezes all128 actual episodes, original rank-local
rollout tensors/RNG, native sampler state and the preceding full model/critic/
Adam/schedule boundary. The actor derives the **actual author loss mask** from
dones and refuses any incomplete horizon, native reward leakage or partial label
bank. Only the terminal reward tensor changes. A failed labeling attempt cannot
perform an update or recollect the batch. Pending resume binds the exact
preceding boundary/file digest, not just iteration/optimizer counters. An
`UPDATE-STARTED` batch is never automatically replayed after a partial optimizer
failure; the execution owner diagnoses it. Whole-boundary resume requires its
saved evaluation history and unchanged canonical action root, with a fresh
attempt identity.

## Actual profile bindings

`runtime-profile.template.json` is deliberately unadmitted. Fill its actual
allocation ID/GPU probe/start/deadline/cost path and loading receipt. Training
also binds successful CPU qualification, actual loading qualification and fresh64 validation
references, common action attribution and a finite per-iteration reservation.
All refs use absolute `path` and `sha256`.

The source reads the owner's existing `ledger.json` cost fields; it creates no
second resource ledger. Rank journals record every physical reset/settle/chunk,
including unresolved intents and padding. Common preparation is charged once
and explicitly attributed before each arm's 1M admission. Maximum8 GPUs is per
scheduler allocation, not an aggregate-per-arm limit; a dedicated Flora
allocation may be separate. Every job remains under8 hours. The former10h
aggregate gate is removed only for these approved jobs and necessary work.

Remaining actual qualifications: published source strict loading,
native clock/case/video runtime, fresh64 gate, GPU capacity and actual advancing
PPO updates. A real failure in these checks needs diagnosis and a new identity;
this source handoff is not evidence they passed.
