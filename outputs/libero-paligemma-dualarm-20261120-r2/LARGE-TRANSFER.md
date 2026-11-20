# Does prior motion history help a source-qualified motor on UR5e?

Use this entry point for the large rerun. The outcome is native sparse success on simulated LIBERO Goal 5 BASE. The primary history comparison is correct UR5e practice versus empty context under the same fitted reader, residual and motor. Raw measures frozen motor transfer. Wrong-body practice tests whether the identity of the measured response matters. Analytic calibration measures performance with separate target pulse data. This is a runnable protocol. Native source scores, fitted weights and UR5e outcomes must come from an allocated run. [CLI](large_transfer.py) · [Source panel](LARGE-PANEL.md)

## Qualification and source-only fitting

| Evidence | Required for target access |
| --- | --- |
| Source development | The producer compares the registered candidates on the same 50 generated layouts per arm. The accepted motor needs at least 38 native successes on Panda and 38 on Sawyer. The development comparison and accepted checkpoint are hashed. |
| Sealed source confirmation | The test manifest binds the earlier selection. The identical motor and replan stride must achieve at least 75/100 separately on each arm. The 100 ordered scene, movable-placement and state identities must match across source arms and be disjoint from development. |
| Source dynamics | Use the complete R1 motion pretraining directory, or regenerate it on Panda and Sawyer. Stage 1 loads and verifies all ten real command–motion tapes and the source checkpoint. Supports are released layouts 20–21, motion queries 22–23 and the separate check is 28. |
| Joint source fit | Reuse the verified source expert bank loader and Stage 2 fitter. Generate frozen proposals at five-control expert prefixes and jointly update reader and action residual for 150 Adam steps, 16 samples per source arm per step. Require nonzero parameter change in both modules. Score crossed source histories on offline validation. |
| Source receipt | Recount all three candidates’ 300 development episodes and the winner’s 200 confirmation episodes from their native tapes. Bind every tape digest, both panel manifests and reports, the pre-confirmation seal, the motor and reader SHA-256 values, source fit and code, and the exact target assignment to one JSON file. The evaluator checks the published byte SHA-256 and reconstructs the receipt from the source evidence before target initialization. |

**Use the source-only fit channel as soon as the development winner is available.** It can overlap sealed confirmation. The phase-1 checkpoint can also be shared across source candidates. A joint checkpoint can be reused only with the identical motor bytes. The evaluator reopens the source data, excludes all evaluation seeds, donor rows, scene and placement hashes, verifies expert actions against the saved proposal banks, and checks the frozen motor tensors before and after execution. Source bank paths keep their supplied order for duplicate attribution. The original bank indices, episode hashes and coverage are in the joint report. The selected large-panel batch-shaped inference is used for both proposal collection and target control. [Source fitter](large_transfer_fit.py) · [Shared fitting and native bank readback](transfer_fit.py) · [Native practice functions](../libero-paligemma-crossarm-20261120-r1/adapt_eval.py)

## Cluster commands

Run from the repository root on the pinned deployment. Set `PIN` to the deployed commit, `WINNER_FIT` to the winning source motor fit directory, and the Bash array `banks` to the immutable accepted source expert bank directories. For the registered SFT campaign keep sft01/fit, sft02/fit and sft03/fit under the common campaign directory. Older e2f794c4 receipts omit the fit paths. Keep that commit available in the deployment Git object database for its exact tape reader and reset bridge. The common R2 launcher configures the simulator and VLM interpreters. The source panel worker owns development, selection and sealed confirmation. Leave the original campaign launcher at its source-only stage when orchestrating this rerun. Its legacy automatic transfer path uses the small panel. [Source producer commands](LARGE-PANEL.md) · [R2 launcher](run.sh)

```bash
run=outputs/libero-paligemma-dualarm-20261120-r2
campaign=/mowenjie/fluffy-bassoon-2/$run
panel=${PANEL_ROOT:?set the existing source producer directory}
prior=/mowenjie/fluffy-bassoon-2/outputs/libero-paligemma-crossarm-20261120-r1
# Example for the winner's existing snapshot layout.
banks=( "$(dirname "$WINNER_FIT")"/data/bank* )

bash "$run/run.sh" "$PIN" ltjoint01 vlm large_transfer.py fit \
  --checkpoint "$panel/accepted/checkpoint.pt" --stage1 "$prior/reader01/pretrain" \
  --demos "${banks[@]}" --out joint
joint=$campaign/ltjoint01/joint
```

For an earlier source-only fit, pass the exact candidate file, for example `$WINNER_FIT/candidates/step-001600.pt`. Omit `--stage1` to run the authentic source-only motion pretrainer within the fit output. Phase 2 accepts registered source bank directories, including validated generated/source training data and released validation episodes for both arms. Source fitting and evidence readback need scratch space for the bank loader's temporary uncompressed source camera streams.

After the source producer publishes both panels and the source fit completes, run qualification in the Linux CPU NumPy environment. The command verifies the per-arm thresholds, registered ranking, accepted checkpoint and sealed layout identities. It publishes the receipt atomically and prints its SHA-256. Repeating the command on identical evidence returns the same digest. Pass that recorded digest to each attempt.

```bash
receipt=$campaign/large-transfer-receipt.json
digest=$(python3 -B "$run/large_transfer.py" qualify \
  --root "$panel" --fit "$WINNER_FIT" --joint "$joint" --out "$receipt") || exit 1
common=(--root "$panel" --fit "$WINNER_FIT" --joint "$joint" \
        --receipt "$receipt" --receipt-sha256 "$digest")
bash "$run/run.sh" "$PIN" ltur5e01 vlm large_transfer.py evaluate "${common[@]}" --out ur5e
python3 -B "$run/large_transfer.py" aggregate --run "$campaign/ltur5e01/ur5e"
```

### Job time and retries

A full run has up to 500 sequential episodes, 150,000 controls and 30,000 policy calls at stride five. For a 95-minute allocation the query budget is below 11.4 seconds per episode once source readback, admission and practice are included. GPU wall time is unmeasured here. Use the per-episode elapsed times and setup durations in `report.json` to estimate the remaining allocation. A stopped attempt retains validated episodes. Use the same panel root and receipt in a fresh attempt to resume from the shared cache. `--reuse DIR` also accepts a prior query-stage run with identical bindings.

For separate allocations, give each a unique attempt name and one zero-based shard index. Shard `i` takes slots `i, i + K, …` with all five conditions kept together. Admissions, practice and episode receipts are cached under the binding. For two shards, run one command per allocation, then combine on Linux CPU.

```bash
bash "$run/run.sh" "$PIN" ltpart0 vlm large_transfer.py evaluate "${common[@]}" --shard-count 2 --shard-index 0 --out ur5e
bash "$run/run.sh" "$PIN" ltpart1 vlm large_transfer.py evaluate "${common[@]}" --shard-count 2 --shard-index 1 --out ur5e
python3 -B "$run/large_transfer.py" aggregate --run "$campaign/ltpart0/ur5e" \
  --reuse "$campaign/ltpart1/ur5e" > "$campaign/large-transfer-summary.json"
```

Aggregation reopens native tapes, rejects inconsistent bindings and requires all 100 layout slots. The single-job path also saves `summary.json`.

## Paired design and readout

1. After source verification, attempt all 100 sealed layouts on the stock UR5e through the shared generated reset implementation. Record the generated Panda donor row, native body identity, full initial-state and scene fingerprints, camera/state hashes and instruction. Save all target admission decisions before collecting target query rewards. Named physical failures retain their slot and reason. Other simulator or evidence errors interrupt the run. [Physical reset bridge](large_panel_sim.py) · [Target process](large_transfer_native.py)
2. Reuse the registered 16-control practice functions on separate released layouts 8–15, cycling in receipt order. Correct and wrong practices run the same probe on the same scene. The 100 wrong donors comprise 50 Panda and 50 Sawyer assignments. The eight target supports are shared across cases. Independent calibration uses Panda and UR5e pulse poses 48 and 49 and a motion check on 39. A physically unavailable support marks its dependent conditions missing.
3. On each admitted query, reset the simulator separately for raw, correct, empty, wrong and calibration. Before issuing the first transformed action, require exact equality with raw on the full reset receipt, opening public observation, seed and entire first 5×7 proposal. Later replans use the selected stride and the deterministic seed `900000 + 1000 × layout_seed + elapsed_controls`. Each condition continues closed loop to native termination or 300 controls. Each hashed NPZ contains full proposals, issued controls, measured motions, native rewards, done/success flags and a receipt bound to the pinned plan and measured practice bank. [Query runner](large_transfer_query.py)
4. Report successes, observed denominator, missing count and planned denominator of 100 for each condition. Every contrast reports complete paired denominator, gains, losses, concordances, absolute success-rate difference, conservative 95% paired interval and exact two-sided McNemar p-value. The interval combines simultaneous exact binomial bounds for gain and loss probabilities. Missing-case bounds show the possible full-panel rate or difference under all assignments of missing outcomes. An interrupted job has a partial record and cannot produce a completed scientific summary. [CPU readback](large_transfer_stats.py)

Interpret `prior_history_answer` from correct-minus-empty. Use correct-minus-raw for practical gain over the motor and correct-minus-wrong for body specificity. A positive raw score alone answers only whether the motor transferred. The intervals concern sampled layouts under this fixed motor, reader, practice library and task. Secondary contrast intervals are marginal. The four-start source screen and earlier transfer pilots contribute zero observations to these denominators.


## Engineering verification

Synthetic Linux CPU checks exercised the original producer’s native tape reader on 300 two-arm episodes and the paired outcome reader on 490 synthetic query tapes with two excluded layouts. The latter recovered 16 gains and zero losses across 98 pairs, and rejected a changed tape and a mismatched first proposal. Combining two disjoint 50-slot shards reproduced the full raw count. A lone shard failed coverage. Linux also checked receipt-byte tampering, the pinned reset-bridge imports, and the extended motor-bank partition and fingerprint interfaces using synthetic generated and released recovery records. The bank-interface check supplied synthetic loader output. Native bank decoding, GPU timing and scientific outcomes require native runs.
