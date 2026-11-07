# Outputs-only archive candidates — November 7, 2026

**Execution follow-up (2026-11-07 14:53:57 Shanghai):** The approved 11-selection batch is complete, with **207.700 GiB reclaimed**, full cold readback/restoration verified and active/protected inputs retained. [Final results and restore mapping](../outputs-retention-20261107-r1/RESULT.md). Figures and no-mutation statements below describe the earlier discovery pass.

**Recommended first batch: 207.79 GiB across 11 disjoint output-directory selections.** These are archive candidates with scientific evidence preserved, not disposable results. The exact selectors and proposed restore commands are in [the machine-readable manifest](OUTPUTS-ARCHIVE-CANDIDATES.json). No archive, hot-payload deletion or GPU work was performed in this discovery pass.

Sizes use today's full census. All 14 first-batch/deferred roots were rechecked present as ordinary PFS directories, with no symlink roots, at **2026-11-07 13:48:32 Shanghai**. A separate development-container FD scan found zero readers of the reviewed roots; it does not see isolated scheduler pods, so execution must refresh scheduler/reader checks immediately before reclamation.

| First batch | Allocated GiB | What goes cold |
| --- | ---: | --- |
| Earlier RoboTwin qualification payloads | 62.20 | Old `qual02/train`, `qual03/train`, `grad02/train` recovery/rollout payloads. |
| Finished short-chunk benchmark checkpoints | 34.62 | The complete `attempt-01/OpenDrawer/checkpoints` tree from the finished one-update timing experiment. |
| Finished DEV64 staged policy copies | 25.89 | Four private inference `.pt` files in `staging-r1`; retain their manifest and conversion recipe. |
| Completed ablation arms, suffix input and endpoint bulk | 85.08 | Finished E/C/D arms, suffix initializer, original40 evaluation copy and endpoint bank payloads; keep the shared prefix initializer hot. |

The total is gross directory allocation. Keep compact code/config/log/result records and numerical qualification tensors hot; actual reclaimed bulk will be slightly smaller and must be measured after verified archive release.

## Exact first-batch paths


### Earlier RoboTwin qualification payloads — 62.20 GiB

- `outputs/robotwin-sidney-matched-baselines-20261106-r1/qual02/train` — 21.74 GiB
- `outputs/robotwin-sidney-matched-baselines-20261106-r1/qual03/train` — 31.16 GiB
- `outputs/robotwin-sidney-matched-baselines-20261106-r1/grad02/train` — 9.30 GiB

Why eligible: robotwin-sidney-matched-baselines-20261106-r1/qual02/EXIT.json: exit255; qual03 and grad02 exit0; qual03/RESULT.md and matched baseline RESULT.md: grouped recovery and the later completed pair.

- Keep numerical gradient/parity records, logs, configs, integrity hashes and result JSON hot.
- Archive old native DCP/Adam/worker/RNG boundaries in full, together with their original raw traces and reset inputs.
- Current matched ppo02/grpo01 and endpoint-readout checkpoints are excluded.

### Finished short-chunk benchmark checkpoints — 34.62 GiB

- `outputs/robocasa-ppo-short-chunk-benchmark-20261104-r1/attempt-01/OpenDrawer` — 34.62 GiB

Why eligible: robocasa-ppo-short-chunk-benchmark-20261104-r1/RESULT.md; BENCH-AUDIT.json: one update complete; all22checkpointfiles verified; post-update evaluation outside scope.

- Retain both step0 and step1 complete boundaries in the cold archive, not only inference weights.
- Keep benchmark timings, replay/gradient checks, markers, costs and resolved recipe hot.
- The run's rollout/video/environment/sampling directories were already archived and are not counted again.

### Finished DEV64 staged policy copies — 25.89 GiB

- `outputs/robocasa-ppo40-matched-dev64-20261104-r1/staging-r1` — 25.89 GiB

Why eligible: robocasa-ppo40-matched-dev64-20261104-r1/RESULT.md; READY-FOUR.json: four policy file hashes, original source boundaries and conversion recipe.

- Archive all four exact files: sft.pt, old.pt, new-ppo40-eval.pt, grpo40-eval.pt.
- Keep READY.json, READY-FOUR.json, prepare.py, original source markers and per-case results hot.
- Converted evaluation files are not byte-identical to their original actors; do not discard on filename/size equivalence.

### Completed ablation arms, suffix input and endpoint bulk — 85.08 GiB

- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/E/attempt-02` — 17.04 GiB
- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/C/attempt-01` — 17.98 GiB
- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/D/attempt-01` — 18.01 GiB
- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/input-suffix-r1` — 15.00 GiB
- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/eval-input-original40` — 6.47 GiB
- `outputs/robocasa-opendrawer-ppo-ablation-3m-20261104-r1/endpoint512` — 10.59 GiB

Why eligible: robocasa-opendrawer-ppo-ablation-3m-20261104-r1/EXECUTION.json: complete; RESULT.md, BUDGET-CLOSEOUT.json, E/C/D-COMPLETE.json and ENDPOINT-AUDIT.json; robocasa-ppo4-grpo40-test512-20261105-r1/RESULT.md: later consumer also complete.

- Keep the shared input-prefix-r1 initializer hot; it is explicitly excluded.
- Preserve every cited step40 checkpoint and its complete recovery state in the cold archive.
- Keep result/config/log files, per-case outcomes, native audits and all COLD-STORAGE/checkpoint sidecar records hot.
- The later PPO4-vs-GRPO evaluator references E; its completed experiment must have an explicit restore pointer to the archived actor.

## Defer 38.77 GiB of predecessor state

- **22.81 GiB `robotwin-sidney-rlinf-ppo-20261105-r1/train03` + `train06`:** train03 was stopped before SGD and train06 failed on host RAM before SGD; train07 subsequently completed 40 Adam updates. However, train03 still carries the explicitly protected step0 recovery boundary and is read by historical archive qualification code. Keep this ancestry in place for the first batch.
- **15.96 GiB `robocasa-ppo-grpo-meta-stage1-20261104-r1/grpo/attempt-01/OpenDrawer`:** this is the actual step0/step10 ancestor from which the final GRPO40 arm resumed. It can eventually be cold-preserved, but it needs its own complete-recovery mapping; do not treat it as a redundant failed run.

## Keep hot

The largest matched RoboTwin production arms (`ppo02`/`grpo01`), pilot `train07`, held 116 GiB critic comparison and joint-steering recoveries are excluded. Also excluded are all canonical Flow10M/FlowR4 teachers/corpora, the shared ablation `input-prefix-r1` initializer, source5 TRAIN/inference assets, and active Mikasa/AnyViewPhys output namespaces. The user requested output candidates; no input/cache/environment reclamation is proposed here.

The ablation's E/C/D step40 actors are cited by reported scientific results. Archiving them must preserve all bytes and recovery state. A completed later PPO4-versus-GRPO512 panel also references E; retain its exact checkpoint hash and restore mapping. This is cold storage, not removal of the reported evidence.

## Concrete archive/restore layout

Use a fresh private namespace:

`/mnt/bos/adgfyl/strong-plant-archives/outputs-retention-20261107-r1/`

Create one `.tar.zst` object per manifest selection, containing the original `outputs/<run>/...` relative hierarchy. Leave compact records and original-path cold pointers on PFS. First restore into a fresh empty staging directory and compare every inventory member. Verified missing payloads can then be restored under `/mowenjie/fluffy-bassoon-2`, recreating the original paths without overwriting changed files or the compact records retained hot. All restore commands are in the manifest. GPU readers must stage restored data back to PFS before launch rather than read a cold object through a new symlink.

Before releasing any source bytes: check the current scheduler/readers, inventory the unchanged source, verify full archive membership/content and independent BOS SHA readback, then fully restore and compare the selected original bytes. Use bounded `/tmp` staging, preserve complete actor/Adam/sampler/worker/RNG states, recheck source stability and readers, and only then release verified bulk files. Retain configs, logs, per-case results, checkpoint boundary markers, numerical reference records and archive receipts hot. Existing rollouts already archived by earlier retention runs are excluded from the totals.

[Fresh directory preflight](output-archive-path-preflight.json) · [Live marker/reader evidence](output-candidate-live-metadata.json) · [Full storage audit](RESULT.md).
