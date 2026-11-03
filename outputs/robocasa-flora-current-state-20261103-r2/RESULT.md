# Current-state diagnostic: all four candidates rejected

Baige Pro6K job `job-6zq2cpuq1c7c`, source `db2ef630a8cb7e2e5d2b8e2b047427841b3c2806`, completed all76 outputs and succeeded/released at15:30:24 Shanghai on2026-11-03. No production rubric or reward map changes.

| Evidence and output | False accepts /12 current negatives | Accepted /6 current positives | Expected-grade AUROC |
| --- | ---: | ---: | ---: |
|64-frame causal prefix, current-state grade |7 |5 |0.4167 |
|Reset/current temporal pairs, current-state grade |0 |0 |0.4444 |
|64-frame causal prefix, condition checklist |10 |6 |N/A |
|Reset/current temporal pairs, condition checklist |0 |0 |N/A |

All four old reference hard grades reproduce (1,5,1,4). All37 unique actual vLLM processor inputs match their CPU bfloat16 references. The official pipeline completed with one model load, no private-truth read by the inference worker, and zero new robot actions or optimizer steps. The four arms fail their predeclared development gate. This reused18-endpoint sample is neither held-out nor population qualification.

The checklist reports target identified and opening short on every reset/current pair, including all six successful states. Prefix recognition still confuses short openings with the native threshold; for example128@400 has native current success but gets grade1 while its condition checklist says meets. Thus a condition checklist under the terminal no-thinking protocol does not solve recognition. Reset/current evidence also changes frame count and temporal context together, so its rejection collapse is not proof that isolated endpoint views lack the required information. Actual learned semantic retention remains uninspected.

The current-state labels differ from earlier ever-success labels:64@750 and216@750 are closed again. These AUROCs must not be compared directly with the H600 terminal-task AUROCs or used to refit a threshold.

## Evidence and cost

Raw `RESULT.json`, `PAIRED-SAMPLE.json`, all13 label/receipt/audit batches, `scoring/PROCESSOR-AUDIT.json`, completion receipt, GPU probe and `terminal-charge.json` remain at `/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-current-state-20261103-r2` on Baige. The run-local summary verifies row receipts and all76 unique IDs before writing its result. The raw55-file archive is111,177 bytes; Chrome blocked its download under organization policy. No alternate transfer or bypass was attempted. [Observed results](OBSERVED-RESULT.json) transcribes inspected aggregate evidence; it is explicitly not a locally hash-verified copy of all raw labels. Earlier CPU inputs were independently downloaded and hash-verified before launch.

The wrapper records1,003 GPU-seconds including startup; the scoring worker records941.792s; the scheduler records1,010s from15:13:34 to15:30:24. Use scheduler cost0.2806 GPU-hours for allocation accounting, not worker time alone. r1 failed on CPU before any weights or GPU use. Adding this job to the prior documented7,925 workerGPU-s subtotal gives8,928 workerGPU-s; this subtotal excludes earlier full-terminal-video work and is not the entire project spend.

## Next boundary

Do not continue fitting wording, grade thresholds, or latching rules to these cases. The official pinned model source exposes a distinct rubric protocol: camera descriptions, per-step point values, a thinking trace, and an `ANSWER:` array parsed with the official parser. Its CLI defaults (tensor parallel2,98,304 context,9 samples,20,480 output tokens) are not the qualified terminal recipe. Qualify causal per-camera inputs, memory/token bounds, strict parser validity, and deterministic bounded settings on allocated Linux before any new GPU diagnostic. Compare a few frozen recognition controls through that interface; reasoning text is diagnostic output, not proof of the model's internal cause. A pass still requires a predeclared unused-case study, then the other four tasks and temporal/reward-map validation before PPO.
