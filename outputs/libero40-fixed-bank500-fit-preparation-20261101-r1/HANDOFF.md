# Controlled current-bank100 ->500 continuation preparation

Root authority: `outputs/libero40-fixed-bank500-controlled-fit-20261101-r1/`.
Sole execution/resource owner: finish_assembly. This package starts no jobs and
changes no active run, original100 file, teacher, source admission or dataset.
No model/array/test/resource execution occurred on macOS. Actual saved-state
inspection and the nine NEW focused cases are prepared, not run.

## Numerical compatibility and exact fork

The four current100 archives are from source
`2bc15da3028ab6a3cbaed4b813f3f761a680f1d9`, job `job-wbvs39zbt0pw`, on the
current33729-TRAIN-row ALL40/fixed8 bank. Their exact endpoint/full-recovery
paths and eight byte hashes are in `PROFILE.json`, taken from the committed
terminal audit. The old527-case500 generation is a different bank and excluded.

Original architecture: prefix4104 -> noise32, width128/layers2/heads4, context300,
window128, seed0, global batch8. AdamW LR3e-4, weight_decay0, gradient clipping1;
no LR scheduler or optimization-depth-dependent loss. Validation every25 uses
four batches of8 with private sampler734001 reinitialized every call.

Static diff against2bc15da3: GTrXL/model/loss code is unchanged. Sampling only
adds an inactive complete-episode option and a value-preserving token copy.
The active full-batch update now multiplies loss by1 before backward; same clip
and Adam step. A focused tiny synthetic old-vs-current update regression is
prepared. It has not run; no GPU bit-equivalence is asserted from static review.

Direct current `vla_history_train --resume` is incompatible at its metadata
contract/output-identity checks: source commit and schedule fields changed.
The new module `vla_source40_fixed_bank_depth` therefore makes an explicit
read-only-original state fork into a fresh controlled output. It does not
weaken generic resume checks or introduce a learner.

The fork preserves policy/normalizer, every Adam state, sampler and global/private
Python/NumPy/Torch/CUDA RNG tensors exactly. It preserves update100 and historical
best legacy NLL. Only source/run/schedule metadata changes and the NEW phase
elapsed clock starts at0; historical time remains in provenance. Existing
`vla_history_train` then restores the fork and executes101..500. Its temporary
normalizer initialization is overwritten by the exact saved100 buffers before
any training; those buffers are checked again at500. No100 refit occurs.

The original legacy VAL schedule remains in the training subprocess, including
its original DPT VAL byte hash. Selection is explicitly fixed-final500, so
legacy or corrected NLL cannot select a favorable checkpoint. Corrected VAL is
separate frozen student inference at100/500, with RNG snapshot/restore and no
state update. AD reports recorded-behavior NLL; DPT reports the same-TRAIN-teacher
metric. All original metrics/files remain untouched.

## Binding and actual prelaunch inspection

Derived assembly FILE:
`1d7a93c8c0ec64a92687003058dac372bf6c19ab7c871aa1e9d218e624b4c29e`;
CONTENT:
`05cb614a21805e2dee390fadc36a278f05704169f1c8c764942fc107f0a4bf0b`.
TRAIN AD/DPT bytes and VAL AD are unchanged. Only VAL DPT becomes
`32eda28573610d1a2bd9fcdb0f332b8075ef6a2676ff03f949bcccd6541a0243`.
This is honest legacy-derived candidate data, not V2 or expert qualification.

The audit binds complete original file bytes, not individual normalizer leaves.
Those identities remain explicitly pending until the actual `inspect` operation
on allocated Linux. It checks the four original fixed100 policies equal their
complete recoveries, actual Adam groups, RNG structure, normalizer shape/finite
values and exact content hashes. It also requires all four normalizers match.
No model construction, forward call or optimizer update occurs in inspection.
Its JSON is portable (including Adam betas), avoiding tuple/list false mismatches.

The parent explicitly permits this actual saved model/state inspection plus
only affected migration tests inside the conditional180CPU-worker-second cap.
Record their independent return codes; a passed synthetic suite is not actual
state qualification. Reuse `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`,
declared Torch2.7.1+cu128/CUDA12.8. Runtime mismatch is a reported incompatibility,
never permission to repair numerics or replace/retrain the100 models.

Training requires the actual inspection file/SHA and compares its canonical
model/Adam/normalizer/RNG identities again. CUDA restore is then checked with
the existing model, Adam loader and RNG restore on that model's own GPU ticket,
with zero SGD. It is explicitly unverified by CPU inspection.
The receipt binds all eight relevant runtime files by SHA. Both source commits
are recorded; metadata-only main advances can carry the inspection only when
those exact code bytes and the frozen profile remain identical. Changed code
cannot silently reuse the actual-state receipt.

## Existing clock and concrete runtime API

Use the qualified `vla_source40_ppo_worker.begin/finish` API; there is no new
clock or launcher. Its shared contract is returned by
`clock_contract(profile, profile_sha, source_commit)`. Root owns calls to begin:
worker_seconds7200, slice_seconds1740, lineage=model ID, lineage_seconds1800.
All four independent GPU tickets can run concurrently. The model driver only
checks its already-owned attempt and starts its one existing trainer subprocess.
Root closes each attempt once with the actual independent return code, including
startup/load/restore/validation/failure/cleanup. No cross-model budget borrowing.

New output identity: `<phase-root>/models/<ad_full|ad_static|dpt_full|dpt_static>`.
Initial and partial full state is atomic; a partial endpoint is not500. A manual
same-phase continuation can use `--resume-phase` only with another actually
admitted remaining original model ticket. It does not reset the model clock,
repeat the100 fork or add updates after500. No automatic retry is requested.
Final500 policy/recovery use existing immutable-copy helper for stable endpoints.

`RECIPE.txt` gives exact CLI templates. `RESOURCES.json` separates measured old
timings from prospective capacity. The original AUTHORITY/GRANT byte hashes are
in PROFILE: freeze those bytes during admission, or regenerate/freeze PROFILE
before inspection if root edits them. Job/adoption status can live in ledger,
clock and launch receipts. Exact inspected profile/code must be adopted before
sole-owner GPU launch; no further human ACK is needed within the adopted grant.

## Bounds and limitations

New grant7200 cumulative GPU-seconds, four fixed1800 subcaps including each
1740-second slice+60 allowance; one GPU/8CPU/32GiB per model,29 scheduler minutes.
New outputs <=16GiB; new robot actions and teacher/source updates0. Old source
457851 actions remain provenance and are not charged or duplicated again.
Actual shared-clock and output checks are included. Model/optimizer/loss/global
batch/sampler/normalizer and data remain fixed; no capacity tuning on failure.

The actual source100 elapsed times are91.09..93.04s/model. A linear400-update
estimate is364.35..372.16s plus load/restore/two corrected panels/output overhead;
budget conservatively450..650s expected per model, bounded by its actual ticket.
Data/controller dimensions are unchanged, so per-step capacity uses the already
completed recipe. New peak RAM/CUDA measurements are still pending actual
admission; no extra SGD or VLA episode is used as a capacity probe.

Controlled offline depth and imitation NLL do not establish robotic success,
memory benefit, teacher competence or best steering. A matched frozen100/500
robot readout with fresh native controls needs a separate concrete grant.
The old25h package and expanded-generation proposals remain unfunded and unsplit.
