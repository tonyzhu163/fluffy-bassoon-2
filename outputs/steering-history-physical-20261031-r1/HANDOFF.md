# Prefix admission and physical readout closure

Prepared follow-on to component pin `77119a5ecb3dea33bd943244b9fb43b7764b3d38`.
That component passed 29/29 on allocated CPU job `job-2yxgfd3wd9sd`, costing
86 CPU-worker seconds. The changes described here require a new composite CPU
qualification; macOS checks are AST, JSON and whitespace only. Tiny synthetic
worlds do not establish robot performance. The completed A1 screen admitted no
source arm; actual source collection, A2, scientific fitting and readout remain
pending. Search trajectories cannot substitute for RL learning histories.

The source integration owner composes this commit with source pin
`48e7834ed004d7ba8712b4c7dd39e6487da34ca6`, paired-reset pin
`969fb44c503fac409373abc88ce5687b71195068`, and original A2 helper
`0279f5a8141d953aa8707037cbe89b17b45ead8b`. Only the existing resource owner
dispatches qualification or model work. This code creates no allocation or
expanded grant: one 8h / 300,000-action / 64-GiB pilot and its original
1,800 CPU-worker-second allowance remain authoritative.

## Prefix API and scientific gates

`steering_history_prefix.build_prefix` accepts actual dictionaries returned by
the existing publisher for all six TRAIN/VAL lineages. It preserves original
raw root/relative-path references and rich immutable recovery references.
Actual source-validation input and published wrappers may have different file
hashes; their actor tensors, counters and contracts must be identical.

Call `build_prefix(..., condition_cross=ORIGINAL_HELPER_RICH_REF, ...)` only into
a fresh aggregate after A2. The stored binding is:

```
condition_cross = {
  "result": ORIGINAL_HELPER_RICH_REF,
  "plan": {"path": ORIGINAL_ABSOLUTE_PLAN_JSON, "sha256": ACTUAL_FILE_SHA256}
}
```

Explicit `{result, plan}` is also accepted. The same six original 24-episode
publications produce a NEW `balanced-024-with-cross.json`; existing pending
12/24 aggregates remain immutable. First scientific profile has prefix24.
Data completion never implies admission. Both AD and DPT require recomputed
original raw24 A2 contrasts: mismatch gap >=0.25 in each condition and pooled
fixed-blind gap >=0.125 over eight pairs. Pass flags alone are insufficient.
Actual source TRAIN1 full states/actor hashes/updates, sampled-Gaussian mode,
task/motor/mapping/scope, fixed paired seeds and raw native masks/costs are bound.
The single blind uses actual final source-VAL success scores and fixed tie order.
DPT labels must use those same A2-qualified TRAIN1 actors for both TRAIN and VAL.
AD additionally needs measured learning; DPT independently needs competence.

## Frozen student readout

`steering_history_readout.bind_plan` binds completed original assemblies/models,
condition contract, the same fixed blind and exclusions from source, validation
and A2. The CLI reuses the qualified coefficient driver, complete-episode
runner, canonical raw writer, shared action ledger and existing RuntimeGuard /
BoundedTeacherRecovery. It does not start a learner or allocate a resource.

The exact full grid is 56 supports + 88 queries = 144 complete episodes:
K0/1/2/4, K4 reset/reward blind/consistent shuffle/wrong CONDITION, separately
trained static, native, one fixed blind stationary retry, and same-calibration
cross-task transfer at K2 from eight total charged donor supports. All support
prefixes finish before any query. Queries restore independent memory snapshots;
wrong-condition and transfer replay actual public traces into fresh controllers.
The common paired-reset API pins the original effective instruction as well as
strict public observation/feature identities. Query feedback is not reused.
Both H600 / reset0 gives 86,400 actions, inside the original 105,000 student slice.

The scientific fit profile is fixed before outcomes: four full/static AD/DPT
fits, 2,500 updates, width128/layers2/heads4, batch8, memory256, LR3e-4, validation
every250 with16 fixed batches, final endpoint, <=360s each /1,440 GPU-s total.
Execution microbatches preserve the same sampled logical batch, valid-position
loss weighting and one clip/optimizer step. A wall-clock partial is resumable
but cannot become an evaluated candidate. Four blocks are a mechanism pilot;
the separate 32-case confirmation and independent seeds remain outstanding.

## Allocated CPU gate

From the exact clean composite pin, with zero visible GPUs:

```
python -m pytest -q tests/test_vla_history.py tests/test_steering_history.py tests/test_steering_history_pipeline.py tests/test_steering_history_cross_admission.py --junitxml=<receipt-dir>/junit.xml
```

The earlier component fixtures explicitly isolate the new A2 gate. Dedicated
tests call the production verifier with actual tiny SAC recoveries and canonical
24-cell raw records, including forged flags, failed contrasts, missing evidence,
wrong sampling/counters/teachers and duplicated cells. Pipeline tests use the
actual common runner/writer/sink for 144 synthetic episodes, actual validation
input-to-publication wrapper differences, original cross-ref retention, and
full-batch versus execution-microbatch equivalence. Retain exact source pin,
allocation/import/CUDA attestation, JUnit/logs/terminal status and elapsed cost.
