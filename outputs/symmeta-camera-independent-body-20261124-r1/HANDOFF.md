# Release contract for the independent Kinova3 comparison

The parent owns both GPU decisions. The worker's qualification consists of the physical bank, public reader inputs and a single-case numerical fixture on Linux CPU. The frozen reader comparison and the native comparison have separate releases. Do not substitute nominal, digital or UR5e predictions. A/B and both PR7 controls each deploy their own three outputs from the assigned Kinova3 view. See PLAN.md and cases.csv for the estimand and membership.

## Frozen checkpoint comparison

The immutable registration and renderer revision is 69f44b996b4c22ba5ffa3133f0079fa671c57bc8. The parent deployed GPU scoring at 6030a74ba800011b650aaedf7a83fb62a80c7418. The file paths below are on Baige shared storage. Supply the primary source-only fit directory and the independently released SHA-256 of its sealed.json. It must contain all six A/B weights at the shared selected update. PR7 uses the three historical train02pro checkpoints and published hashes. The full evaluation uses the assigned physical view per support for true and swapped calibration, retaining all three training seeds offline. Grouped output includes the three head errors and paired whole-task intervals. No nominal RGB enters the inference batch.

```bash
run=symmeta-camera-independent-body-20261124-r1
base=/mowenjie/fluffy-bassoon-2
store="$base/outputs/$run"
score_pin=6030a74ba800011b650aaedf7a83fb62a80c7418
score_src="$base/setups/symmeta-perceiver-followups-20261123-r1/src-${score_pin:0:12}"
# Inside the parent-released single-card Distributed job; bank hash in RESULT.md.
bash "$score_src/outputs/$run/offline.sh" release "$score_pin" score01 \
  "$store/cpu01/bank64.json" "$BANK_SHA" 64 "$PRIMARY_FIT" "$PRIMARY_SEAL_SHA"
```

The immutable receipt is the SHA-256 of `$store/score01/full-head-release.json`. It binds the case schedule, source seal/step, full prediction file, bank, checkpoint hashes, observed pixels and supplied H to every native triple. The critical comparisons are A vs B and A vs shipped+warp. The raw shipped reader and true map remain declared controls. `offline.json` contains inference diagnostics only.

## Parent-only native job

Use a new attempt directory on one Baige Distributed RTX 4090. The parent supplies a verified detached checkout containing native.sh and native_panel.py, the independent full-head receipt SHA, the CPU query reset file digest, and the remaining budget in wall seconds. Deduct all charged time from the 2.5 additional GPU-hour envelope, including independent scoring and failed attempts; set the scheduler wall limit accordingly. Fresh pair files are written only after all five complete episodes pass action/reward, shared reset, opening proposal, noise and camera checks. Inspect the first real episode's qualification and monitor the submitted allocation. The primary native result is produced only after every arm has completed every registered case.

```bash
bash "$NATIVE_SRC/outputs/$run/native.sh" "$NATIVE_PIN" native01 \
  "$store/score01/full-head-release.json" "$HEAD_RELEASE_SHA" \
  "$store/cpu01/query-resets.json" "$RESETS_SHA" "$REMAINING_GPU_SECONDS"
```

The script runs the original Kinova3 evaluator and frozen π₀.₅ at its historical collector revision, retains both original motor cameras, and uses the shared native physical-command/reward capture code. Its explicitly seven-joint validation and paired analysis are in native_panel.py. A CPU fixture cannot qualify the real motor, renderer, first retry, or possible speed of this future job. If the parent declines or a complete physical cohort fails validation, release the bank and any authorized offline result with the native endpoint unmeasured.
