# Native evaluation handoff

## Initial Linux CPU readback

On November 23, 2026, baige-dev read and hashed all 384 passive NPZs, matched every path to its shard manifest and found one archived support physics fingerprint per task (eight per panel). The ordered SHA-256 digests, computed over one lowercase tape SHA-256 plus newline per case, matched the existing FK reader receipts.

| Panel | Runtime | Tape set SHA-256 |
| --- | --- | --- |
| gpu09test | Goal6 ff2ddf0a, UR5e π₀.₅ | 8729fa68343e2a7a0cc6471ab25e7c707555a866e682a3dc999079a7b9dc3554 |
| gpu09iiwa | Wave2d 490f25c5, IIWA π₀.₅ | f889bc291b695132dab6ef290d86887c94883e431fa61be7be27cd44bd455602 |
| gpu06kinova3 | Wave2b 465f33dc, Kinova3 π₀.₅ | 13c41c879edb89b4d139dc73b20ed78ec3da39b611caf0b36969662c10e1ae88 |
| gpu08pi0fix | Wave2d 490f25c5, UR5e π₀ | 5b092f507a9ccfd614f56b5ce7a3046451d9c86ca2b46c8df8565ce94934588d |

The original shard summaries store rounded corrections, success and action counts. They lack a per-reader opening-proposal digest. All 2,304 scored episodes therefore run fresh unless an exact pair from a matching attempt of this experiment is reused. Historic FK availability filters apply to an active-probe kinematics estimate. Every passive reader input here is available.

## Pairing and recovery

Each panel starts its own pinned historical evaluator, simulator worker and frozen policy. The evaluator joins each prediction to the official schedule by panel, case index and reader seed and checks suite, integer task ID, embodiment, regime when supplied, and passive tape SHA-256. A complete input has 384 distinct panel–case IDs, three reader seeds per ID and exactly 1,152 keys. Both correction triples pass through at their supplied precision. The preparation verifies the archived support physics, layout 11, tape path and bytes, plus the native policy weights. On UR5e π₀.₅, the Goal6 recovery evaluator and the earlier FK panel provide the query contract. The three other panels use the FK cross-arm archive checks. [Input verification](paired_inputs.py)

The native query uses layout 13, BASE, the released prompt and the task's native horizon. Its seed is 41,800,000 plus 10,000 times the sorted suite index, 100 times the task ID and 13. A float32 noise bank seeded at query seed + 817,117 supplies one draw per five controls, with motor shape (10, 32) for π₀.₅ and (50, 32) for π₀. Prepare reconstructs every bank and verifies its archived digest. Both π₀ reader arms use the original quaternion conversion at each motor observation. Each task has an unscored warm create. Each reader uses the native callback with up to three pre-action creates. The trajectory runs once after the callback accepts its reset.

An accepted pair has identical full reset receipts and identical hashes of the first full motor proposal before correction. Stable conditions match the historical task, including horizon, noise, scene, perturbation plan, controller schema, model class and released prompt. A validated record retains each reader's exact correction, native success, rewards, return, controls, terminal flags, reset, and opening digest. It reaches its final filename only after both readers pass. The final control's native simulator success supplies the outcome. [Native harness](live.py) · [Record validation](paired_records.py)

Opening proposals are scoped to the pair. I checked the UR5e FK and matched-flow native archives on Baige. The resets were equal for all eight first cases across the two jobs and the opening hashes differed for all eight. For task 0 the openings start with `78a420b3fa1b` and `b70e38f0d873`. Each job satisfied its own two-reader pairing. The reducer reports the number of distinct reset, opening and execution fingerprints by task.

A fresh attempt accepts one or more prior attempt names. Reuse verifies the complete contract, prepared panel hashes, GPU model and capacity, driver/framework versions, per-pair content hash, full row identity and both native records. Adopted pairs keep their original execution receipt and exact bytes. Each task logs adopted keys and source attempts. Missing, corrupt, incompatible or independently conflicting pair records receive fresh paired evaluation. Each comparison has a separate reset reference. Changing the card count or lane does not change its identity.

## Parent launch

Use the parent-supplied production prediction file and its independently recorded SHA-256. Entry points below run from the parent's verified Linux checkout. The CPU-only prepare uses a fresh attempt name and zero cards. The GPU run creates a separate fresh attempt and repeats input verification inside the allocation.

```bash
run=outputs/symmeta-perceiver-retry-20261123-r1
bash "$run/live.sh" prepare "$PIN" "$CPU_ATTEMPT" "$PREDICTIONS" "$PREDICTIONS_SHA256" 0 "$CPU_WALL_SECONDS"
bash "$run/live.sh" run "$PIN" "$ATTEMPT" "$PREDICTIONS" "$PREDICTIONS_SHA256" "$GPU_COUNT" "$WALL_SECONDS"
```

For a retry, append the prior attempt name or names after the wall time. One Baige Distributed Training master container holds four or eight homogenous RTX 4090 or RTX PRO 6000D cards. The launcher runs the deployment GPU probe for the full inherited allocation, joins each CUDA token to its physical PCI device and resolves EGL enumeration with the NVIDIA CUDA device attribute. A lane narrows its motor's CUDA visibility and verifies CUDA device zero's PCI address. Its native simulator subprocess renders on that lane's explicit EGL index. All processes stay inside the scheduler container. Lane d handles tasks d, d + card count, and so on through task 31. Each nonempty task loads the appropriate frozen policy once and runs up to 36 matched comparisons. [Launcher](live.sh) · [Device binding](paired_devices.py)

For a sparse checkout, include this run, `scripts/[train]gpu_probe.py`, `src/fluffy_bassoon/`, the matched-flow `live.py`, FK UR5e `paired.py`, the FK cross-arm run including `evidence/` and `QUERY_CONFIG.json`, and the ReGround rotation reference `reference.py`. Their run directories are `symmeta-matched-flow-20261123-r1`, `symmeta-fk-ur5e-20261123-r1`, `symmeta-fk-crossarm-20261123-r1` and `reground-rotation-reference-20261123-r1` under `outputs/`. The historical runtime checkouts, model weights and data remain on the original Baige shared mount.

## Monitoring and reduction

The attempt directory is `/mowenjie/fluffy-bassoon-2/outputs/symmeta-perceiver-retry-20261123-r1/$ATTEMPT`. Start with `config.txt`, `gpu.json`, the content-addressed execution receipt, and the first `pairs/PANEL/sSEED/II.json` on each lane. Shard logs and return codes are under each panel directory. The parent owns the 20-minute progress checks, budgets and scheduling. A failed or interrupted task preserves all previously sealed pairs. At any point after input preparation, including after an external kill, Linux CPU can reconstruct progress from those files.

```bash
python3 "$run/paired_records.py" reduce "$ATTEMPT_DIR" --partial
python3 "$run/paired_records.py" reduce "$ATTEMPT_DIR"
```

`partial.json` gives observed denominators and missing (index, seed) keys per panel. The strict reducer validates all 1,152 files, rejects unexpected result paths and writes `summary.json`. Its per-panel overall/OOD totals are 288/192, or 96/64 for each seed. Each split includes native counts and percentages for both readers, wins/losses/ties, the difference in percentage points and sample SD across the three seed percentages for both readers and their paired difference. A 95% percentile interval for tuned minus SymMeta resamples the eight whole (suite, task ID) clusters, retaining every seed and transform together. The integer convolution covers all 8⁸ ordered resamples exactly. Incomplete splits report descriptive counts and defer the seed SD and task interval until their denominators are complete.
